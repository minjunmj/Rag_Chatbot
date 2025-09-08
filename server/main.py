import os
import uuid
import json
import logging
import traceback
import asyncio
# import tempfile
from functools import lru_cache
from shutil import which
# import threading
import secrets  

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Literal, Optional, Dict, Any  # [FIX] 타입 사용

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from sentence_transformers import CrossEncoder
import torch
import numpy as np

from faster_whisper import WhisperModel
import subprocess

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "25"))
stt_model: WhisperModel | None = None

# ─────────────────────────────────────────────
# 로깅 & 앱
# ─────────────────────────────────────────────
log = logging.getLogger("uvicorn.error")
log.setLevel(logging.INFO)

app = FastAPI()

# 모든 예외 로그 (HTTPException 포함해서 별도로도 찍음)
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    log.error("💥 Unhandled exception at %s %s", request.method, request.url.path)
    log.error("Headers: %s", dict(request.headers))
    log.error("Detail: %s", str(exc))
    log.error("Traceback:\n%s", "".join(traceback.format_exc()))
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "exc_type": type(exc).__name__, "exc_msg": str(exc)},
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

from fastapi.exceptions import HTTPException as FastAPIHTTPException

@app.exception_handler(FastAPIHTTPException)
async def http_exception_logger(request: Request, exc: FastAPIHTTPException):
    # HTTPException도 콘솔에 이유를 남김
    log.error("⚠️ HTTPException %s %s -> %s", request.method, request.url.path, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail},
                        headers={"Content-Type": "application/json; charset=utf-8"})

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# ─────────────────────────────────────────────
# 준비
# ─────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    log.warning("⚠️ OPENAI_API_KEY가 설정되어 있지 않습니다.")

# gpt api사용
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)
llm_router = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)

TOP_K = 3
MODEL_PATH    = os.getenv("MODEL_PATH", "/app/model_bs32")
INDEX_PATH    = os.getenv("INDEX_PATH", "/app/store")
RAW_DATA_ROOT = os.getenv("RAW_DATA_ROOT", "/app/data")

# 🔸 (신규) RAG 유사도 임계치 환경변수
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.20"))

def assert_ffmpeg():
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Windows: 'winget install Gyan.FFmpeg' 후 PATH 등록 필요")

assert_ffmpeg()

# 임베딩/FAISS
embedding_model = HuggingFaceEmbeddings(model_name=MODEL_PATH)
vectorstore = FAISS.load_local(INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
log.info("✅ FAISS 인덱스 로드 완료: %s", INDEX_PATH)

# Whisper: CPU 강제 토글(문제 진단용)
USE_CPU_WHISPER = os.environ.get("USE_CPU_WHISPER", "0") == "1"

def init_whisper():
    device = "cpu"
    compute_type = "int8"
    if not USE_CPU_WHISPER:
        try:
            import torch
            if torch.cuda.is_available() and torch.backends.cudnn.is_available():
                device = "cuda"
                compute_type = "float16"
        except Exception as e:
            log.warning("CUDA 체크 실패. CPU로 진행: %s", e)
    log.info("🎧 Whisper 초기화: device=%s, compute_type=%s", device, compute_type)
    return WhisperModel("base", device=device, compute_type=compute_type)

@lru_cache(maxsize=1)
def get_whisper():
    global stt_model
    if stt_model is None:
        stt_model = init_whisper()
    return stt_model

# ─────────────────────────────────────────────
# 유틸: ffmpeg로 WAV 변환
# ─────────────────────────────────────────────
def ffmpeg_to_wav(in_path: str, out_path: str, sr: int = 16000):
    # -vn: 영상 제거, -ac 1: 모노, -ar: 샘플레이트, -f wav: 포맷
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", in_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", out_path]
    log.info("🛠️ ffmpeg 변환 시작: %s", " ".join(cmd))
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0 or not os.path.exists(out_path):
        err = cp.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg 변환 실패(code={cp.returncode}): {err[:800]}")
    log.info("✅ ffmpeg 변환 성공 → %s", out_path)

# Chat_History 그전 대화를 기록 
_STORE = {}
def get_session_history(session_id: str):
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]

def get_session_id(request: Request) -> str:
    return (
        request.headers.get("X-Session-Id")
        or request.query_params.get("session_id")
        or "anon"
    )

# Direct or Rag 라우터
class RouteDecision(BaseModel):
    route : Literal["RAG", "DIRECT"]
    reason : Optional[str] = None

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 '법률 판례 Q&A' 시스템의 라우터입니다. 출력은 반드시 JSON만 반환하세요. "
            "사용자의 질문이 내부 법률 지식베이스(벡터스토어: 국내 판례 원문 JSON(사건번호/메타 포함), "
            "판시사항·판결요지·판례내용 발췌, 내부 해설/요약 문서) 조회가 필요한지 결정하세요.\n"
            "다음의 경우에는 기본적으로 RAG를 선택하세요 (모호하면 RAG):\n"
            " 1) 특정 법령/조문/요건/구성요건 충족 여부 판단\n"
            " 2) 사실관계를 제시하고 그 사실에 부합하는 판례/사례 요구\n"
            " 3) '관련/유사 판례', '근거/출처', '최신/최근 판례', '인용' 요구\n"
            " 4) 판결문 내용·판시사항·판결요지 원문 확인 필요\n"
            " 5) 행정/형사/민사 제재·처벌·과태료·벌금·무효/취소 판단 질의\n"
            " 6) 특정 법령명이나 조문을 직접 언급\n"
            "- 필요하면 정확히: {{\"route\":\"RAG\",\"reason\":\"...\"}}\n"
            "- 필요 없으면 정확히: {{\"route\":\"DIRECT\",\"reason\":\"...\"}}\n"
            "추가 설명 금지. 모호하면 RAG."
        ),
    ),
    ("human", "질문:\n{question}"),
])
router_chain = router_prompt | llm_router.with_structured_output(RouteDecision)

# DIRECT 프롬프트
direct_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 도움을 주는 조수입니다. 외부/사설 자료(사용자 파일·코드·문서 등)를 조회하지 말고, "
            "일반 지식과 주어진 맥락만으로 간결하고 정확하게 답변하세요. "
            "만약 사설 자료 없이는 확정하기 어려운 부분이 있다면, 그 사실을 한 줄로만 짧게 밝혀주세요."
            "답변은 한국어로 해주세요."
        ),
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

direct_chain = direct_prompt | llm

direct_chain_with_history = RunnableWithMessageHistory(
    direct_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 비동기 헬퍼
async def llm_route_decision(question: str) -> RouteDecision:
    return await router_chain.ainvoke({"question": question})

async def llm_direct_answer(question: str, *, session_id: str) -> str:
    hist = get_session_history(session_id)
    log.info("🕘 BEFORE DIRECT history_len=%d", len(hist.messages))
    resp = await direct_chain_with_history.ainvoke({"question": question}, config={"configurable": {"session_id": session_id}})
    log.info("🕙 AFTER  DIRECT history_len=%d", len(hist.messages))
    return getattr(resp, "content", str(resp))

# ─────────────────────────────────────────────
# RAG
# ─────────────────────────────────────────────
try:
    _DEVICE = "cuda" if (not USE_CPU_WHISPER and torch.cuda.is_available() and torch.backends.cudnn.is_available()) else "cpu"
except Exception:
    _DEVICE = "cpu"
cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3", device=_DEVICE, max_length=512)

class Query(BaseModel):
    question: str

def chat_rag(query: Query):
    q_text = query.question
    candidates = vectorstore.similarity_search(q_text, k=50)

    # 중복 제거 (사건번호 기준)
    seen = set()
    uniq_candidates = []
    for d in candidates:
        case_no = d.metadata.get("사건번호")
        if case_no in seen:
            continue
        seen.add(case_no)
        uniq_candidates.append(d)
    candidates = uniq_candidates[:50]

    q_emb = np.asarray(embedding_model.embed_query(q_text), dtype=np.float32)
    doc_texts = [d.page_content for d in candidates]
    doc_embs  = np.asarray(embedding_model.embed_documents(doc_texts), dtype=np.float32)

    def _l2norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
        np.maximum(n, 1e-12, out=n)
        return x / n

    q = _l2norm(q_emb.reshape(1, -1))[0]
    D = _l2norm(doc_embs)

    lambda_mult = 0.9
    top_mmr = 10
    selected = []
    unselected = set(range(len(D)))
    sims = D @ q

    while len(selected) < min(top_mmr, len(D)):
        if not selected:
            i = int(np.argmax(sims))
            selected.append(i); unselected.remove(i)
            continue
        max_div = np.max(D[list(selected)] @ D[list(unselected)].T, axis=0)
        mmr_scores = lambda_mult * sims[list(unselected)] - (1 - lambda_mult) * max_div
        pick_idx_in_un = int(np.argmax(mmr_scores))
        i = list(unselected)[pick_idx_in_un]
        selected.append(i); unselected.remove(i)

    stage2_docs = [candidates[i] for i in selected]

    # rerank
    pairs  = [(q_text, d.page_content) for d in stage2_docs]
    scores = cross_encoder.predict(pairs, batch_size=64, show_progress_bar=False)
    order  = np.argsort(scores)[::-1][:TOP_K]

    used_docs = [stage2_docs[i] for i in order]
    top_scores = [float(scores[i]) for i in order] if len(order) > 0 else []
    best_score = max(top_scores) if top_scores else -1e9

    # 임계치: 낮으면 '관련 없음'
    if not np.isfinite(best_score) or best_score < RERANK_THRESHOLD or len(used_docs) == 0:
        # 질문만으로 간단 답변
        prompt = f"""다음 질문에 대해 내부 판례에서 뚜렷한 관련 근거를 찾지 못했습니다.
가능한 범위에서 일반적 설명을 한국어로 간단히 제시하세요.

[질문]
{q_text}
"""
        resp = llm.invoke(prompt)
        answer = getattr(resp, "content", str(resp))
        # 서버가 명확히 표기
        answer += "\n\n관련된 판례는 없습니다."
        return {"answer": answer, "used_files": []}

    # 파일명 수집
    docs_name = []
    for d in used_docs:
        cn = d.metadata.get("사건번호")
        if cn:
            docs_name.append(f"{cn}.json")

    # 원문 로딩
    contexts = []
    if docs_name:
        for subdir in os.listdir(RAW_DATA_ROOT):
            subdir_path = os.path.join(RAW_DATA_ROOT, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for filename in os.listdir(subdir_path):
                if filename in docs_name:
                    file_path = os.path.join(subdir_path, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        판시사항 = (data.get("판시사항") or "").strip()
                        판결요지 = (data.get("판결요지") or "").strip()
                        판례내용 = (data.get("판례내용") or "").strip()
                        if any([판시사항, 판결요지, 판례내용]):
                            contexts.append(
                                f"[파일명: {filename}]\n[판시사항]\n{판시사항}\n[판결요지]\n{판결요지}\n[판례내용]\n{판례내용}\n"
                            )
                    except Exception as e:
                        log.warning("❌ 원문 로드 실패: %s → %s", file_path, e)

    final_context = "\n\n---\n\n".join(contexts) if contexts else "(관련 문서 없음)"
    # 더 이상 LLM에게 파일명 출력 강제하지 않음 (서버가 직접 붙임)
    prompt = f"""다음은 질문에 관한 판례 발췌입니다. 이 내용을 활용해 질문에 한국어로 간결히 답하세요.
자료가 충분하지 않으면 추정/일반 설명도 허용됩니다.

{final_context}

[질문]
{q_text}
"""
    resp = llm.invoke(prompt)
    answer = getattr(resp, "content", str(resp))

    # 서버에서 실제 파일명을 직접 덧붙임
    if docs_name:
        answer += "\n\n관련된 판례는 다음과 같습니다: [" + ", ".join(docs_name) + "]"
    else:
        answer += "\n\n관련된 판례는 없습니다."

    return {"answer": answer, "used_files": docs_name}

# ─────────────────────────────────────────────
# 라우트
# ─────────────────────────────────────────────
JOBS: Dict[str, Dict[str, Any]] = {}  # job_id -> {status, question, answer, used_files, message}

os.makedirs("temp", exist_ok=True)

@app.post("/voice-chat")
async def voice_chat(request: Request, file: UploadFile = File(...), stt_model: WhisperModel = Depends(get_whisper)):
    """
    업로드 → 즉시 202 + job_id 반환.
    백그라운드에서 전체 파이프라인(ffmpeg → STT → 라우팅 → DIRECT/RAG)을 수행.
    """
    session_id = get_session_id(request)
    log.info("👤 session_id=%s", session_id)
    log.info("📥 /voice-chat 업로드: filename=%s, content_type=%s", file.filename, file.content_type)

    uid = uuid.uuid4().hex
    orig_ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
    in_path = os.path.join("temp", f"{uid}{orig_ext}")
    wav_path = os.path.join("temp", f"{uid}.wav")

    # 1) 파일 저장
    raw = await file.read()
    with open(in_path, "wb") as f:
        f.write(raw)
    log.info("📄 저장 완료: %s (%d bytes)", in_path, len(raw))

    # 2) 작업 등록 + 비동기 실행
    job_id = secrets.token_hex(8)
    # 처음에는 중립 문구
    # JOBS[job_id] = {"status": "processing", "message": "처리 중..."}
    JOBS[job_id] = {"status": "processing"}
    async def run_job():
        try:
            # 변환
            ffmpeg_to_wav(in_path, wav_path, sr=16000)

            # STT
            log.info("🗣️ STT 시작")
            segments, info = stt_model.transcribe(wav_path)
            texts = []
            for seg in segments:
                t = getattr(seg, "text", None)
                if t is None and isinstance(seg, dict):
                    t = seg.get("text")
                if t:
                    texts.append(t.strip())
            question_text = " ".join(texts) if texts else "(음성에서 텍스트를 추출하지 못했습니다.)"
            log.info("🗣️ STT 결과 길이=%d", len(question_text))

            # 라우팅
            decision = await llm_route_decision(question_text)
            log.info("🧭 라우팅: %s (%s)", decision.route, decision.reason)

            # 진행 상태 문구 업데이트: RAG만 '자료 검색중...'
            if decision.route == "RAG":
                JOBS[job_id]["message"] = "자료 검색중..."
            # else:
            #     JOBS[job_id]["message"] = "답변 생성중..."

            if decision.route == "DIRECT":
                log.info("💬 DIRECT 답변 생성")
                answer = await llm_direct_answer(question_text, session_id=session_id)
                JOBS[job_id] = {
                    "status": "done",
                    "question": question_text,
                    "answer": answer,
                    "used_files": []
                }
                return

            # RAG
            log.info("🔎 RAG 검색 시작")
            rag_resp = await asyncio.to_thread(chat_rag, Query(question=question_text))
            used = rag_resp.get("used_files", [])
            answer = rag_resp.get("answer", "")

            # 히스토리 반영
            hist = get_session_history(session_id)
            log.info("🕘 BEFORE RAG   history_len=%d", len(hist.messages))
            hist.add_user_message(question_text)
            hist.add_ai_message(answer)
            log.info("🕙 AFTER  RAG   history_len=%d", len(hist.messages))

            JOBS[job_id] = {
                "status": "done",
                "question": question_text,
                "answer": answer,
                "used_files": used
            }
        except Exception as e:
            log.error("❌ 작업 실패: %s", e, exc_info=True)
            JOBS[job_id] = {"status": "error", "message": f"서버 오류: {str(e)}"}
        finally:
            # 임시 파일 정리
            for p in (in_path, wav_path):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except:
                    pass

    asyncio.create_task(run_job())

    # 즉시 202 반환 (Cloudflare/모바일 타임아웃 회피)
    return JSONResponse({"status": "processing", "job_id": job_id},
                        status_code=202,
                        headers={"Location": f"/voice-chat/result?job_id={job_id}",
                                 "Content-Type": "application/json; charset=utf-8"})

@app.get("/voice-chat/result")
async def voice_chat_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"status": "error", "message": "unknown job_id"}, status_code=404,
                            headers={"Content-Type": "application/json; charset=utf-8"})
    return JSONResponse(job, status_code=200, headers={"Content-Type": "application/json; charset=utf-8"})

# [FIX] 아래 동기 처리 라우트는 비동기 잡 방식과 경로 충돌하므로, 경로명을 변경
# @app.post("/voice-chat/sync")
# async def voice_chat_sync(request: Request, file: UploadFile = File(...), stt_model: WhisperModel = Depends(get_whisper)):
#     session_id = get_session_id(request)
#     log.info("👤 session_id=%s", session_id)
#     log.info("📥 /voice-chat/sync 업로드: filename=%s, content_type=%s", file.filename, file.content_type)

#     uid = uuid.uuid4().hex
#     orig_ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
#     in_path = os.path.join("temp", f"{uid}{orig_ext}")
#     wav_path = os.path.join("temp", f"{uid}.wav")
#     try:
#         # 1) 저장
#         raw = await file.read()
#         with open(in_path, "wb") as f:
#             f.write(raw)
#         log.info("📄 저장 완료: %s (%d bytes)", in_path, len(raw))

#         # 2) 오디오 → WAV(ffmpeg)
#         ffmpeg_to_wav(in_path, wav_path, sr=16000)

#         # 3) STT
#         log.info("🗣️ STT 시작")
#         segments, info = stt_model.transcribe(wav_path)
#         texts = []
#         for seg in segments:
#             t = getattr(seg, "text", None)
#             if t is None and isinstance(seg, dict):
#                 t = seg.get("text")
#             if t:
#                 texts.append(t.strip())
#         question_text = " ".join(texts) if texts else "(음성에서 텍스트를 추출하지 못했습니다.)"
#         log.info("🗣️ STT 결과 길이=%d", len(question_text))

#         # Decision
#         decision = await llm_route_decision(question_text)
#         log.info("🧭 라우팅: %s (%s)", decision.route, decision.reason)

#         if decision.route == "RAG":
#             log.info("🔎 RAG 검색 시작")
#             rag_resp = chat_rag(Query(question=question_text))
#             used = rag_resp.get("used_files", [])
#             answer = rag_resp.get("answer", "")
#             log.info("✅ RAG 완료, 사용 파일: %s", used)

#             hist = get_session_history(session_id)
#             log.info("🕘 BEFORE RAG   history_len=%d", len(hist.messages))
#             hist.add_user_message(question_text)
#             hist.add_ai_message(answer)
#             log.info("🕙 AFTER  RAG   history_len=%d", len(hist.messages))

#             return {"question": question_text, "answer": rag_resp["answer"], "used_files": used}

#         if decision.route == "DIRECT":
#             log.info("💬 DIRECT 답변 생성")
#             answer = await llm_direct_answer(question_text, session_id=session_id)
#             return {"question": question_text, "answer": answer, "used_files": []}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"voice-chat failed: {e}")

#     finally:
#         for p in (in_path, wav_path):
#             try:
#                 if p and os.path.exists(p):
#                     os.remove(p)
#             except:
#                 pass
