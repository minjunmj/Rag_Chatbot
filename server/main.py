import os
import uuid
import json
import logging
import traceback
import asyncio
from functools import lru_cache
from shutil import which
import secrets

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Intel MKL 라이브러리 중복 로드 오류 방지

from fastapi import FastAPI, UploadFile, File, Request, Depends
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Literal, Optional, Dict, Any

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

# Whisper STT 모델
stt_model: WhisperModel | None = None

# 로깅 & 앱 초기화
log = logging.getLogger("uvicorn.error")
log.setLevel(logging.INFO)

app = FastAPI()

# 처리되지 않은 모든 예외를 잡아 JSON 형식으로 반환 (500 Internal Server Error)
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

# HTTPException 발생 시 콘솔에 로그를 남기고 JSON으로 반환
@app.exception_handler(FastAPIHTTPException)
async def http_exception_logger(request: Request, exc: FastAPIHTTPException):
    log.error("⚠️ HTTPException %s %s -> %s", request.method, request.url.path, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail},
                        headers={"Content-Type": "application/json; charset=utf-8"})

# Docker / Cloudflare 생존 확인용
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# 루트 접속 시 Swagger 문서로 리다이렉트
@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")

# 브라우저의 favicon 자동 요청 무시 (204 No Content)
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# 환경변수 및 모델 초기화
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    log.warning("⚠️ OPENAI_API_KEY가 설정되어 있지 않습니다.")

# 답변 생성용 LLM (temperature=0.7: 자연스럽고 유연한 답변)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)
# 라우팅 판단용 LLM (temperature=0.0: 일관된 결정적 분류)
llm_router = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)

TOP_K = 3  # 최종 답변 생성에 사용할 판례 수
MODEL_PATH    = os.getenv("MODEL_PATH", "/app/model_bs32")   # 로컬 fine-tuned 임베딩 모델 경로
INDEX_PATH    = os.getenv("INDEX_PATH", "/app/store")         # FAISS 벡터 인덱스 경로
RAW_DATA_ROOT = os.getenv("RAW_DATA_ROOT", "/app/data")       # 판례 원문 JSON 루트 디렉토리

# CrossEncoder rerank 점수 임계치: 이 값 미만이면 "관련 판례 없음" 처리
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.20"))

def assert_ffmpeg():
    """ffmpeg 설치 여부 확인 (오디오 변환에 필수)"""
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Windows: 'winget install Gyan.FFmpeg' 후 PATH 등록 필요")

assert_ffmpeg()

# 임베딩 모델 로드 (로컬 fine-tuned HuggingFace 모델)
embedding_model = HuggingFaceEmbeddings(model_name=MODEL_PATH)
# 사전 구축된 FAISS 벡터 인덱스 로드
vectorstore = FAISS.load_local(INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
log.info("✅ FAISS 인덱스 로드 완료: %s", INDEX_PATH)

# GPU 사용 강제 제어 (USE_CPU_WHISPER=1 이면 CPU, 0 이면 자동 감지)
USE_CPU_WHISPER = os.environ.get("USE_CPU_WHISPER", "0") == "1"

def init_whisper():
    """Whisper STT 모델 초기화 (GPU 가능 시 CUDA float16, 아니면 CPU int8)"""
    device = "cpu"
    compute_type = "int8"
    if not USE_CPU_WHISPER:
        try:
            if torch.cuda.is_available() and torch.backends.cudnn.is_available():
                device = "cuda"
                compute_type = "float16"
        except Exception as e:
            log.warning("CUDA 체크 실패. CPU로 진행: %s", e)
    log.info("🎧 Whisper 초기화: device=%s, compute_type=%s", device, compute_type)
    return WhisperModel("base", device=device, compute_type=compute_type)

@lru_cache(maxsize=1)
def get_whisper():
    """첫 호출 시 Whisper 모델을 초기화하고 이후 캐시에서 재사용 (FastAPI Depends용)"""
    global stt_model
    if stt_model is None:
        stt_model = init_whisper()
    return stt_model

# 유틸: ffmpeg로 오디오 → WAV 변환, Whisper가 제대로 인식하게하기 위해서
def ffmpeg_to_wav(in_path: str, out_path: str, sr: int = 16000):
    """
    ffmpeg를 사용해 임의 오디오 파일을 Whisper에 맞는 WAV로 변환.
    옵션: -vn(영상 제거), -ac 1(모노), -ar 16000(샘플레이트)
    """
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", in_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", out_path]
    log.info("🛠️ ffmpeg 변환 시작: %s", " ".join(cmd))
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0 or not os.path.exists(out_path):
        err = cp.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg 변환 실패(code={cp.returncode}): {err[:800]}")
    log.info("✅ ffmpeg 변환 성공 → %s", out_path)

# 세션 히스토리 관리
# session_id → InMemoryChatMessageHistory 매핑 (서버 메모리에 대화 이력 보관)
_STORE = {}

def get_session_history(session_id: str):
    """세션 ID로 대화 히스토리를 조회하거나 없으면 새로 생성"""
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]

def get_session_id(request: Request) -> str:
    """
    요청에서 세션 ID 추출 (우선순위: 헤더 > 쿼리파라미터 > 'anon')
    Flutter 앱은 X-Session-Id 헤더로 세션 ID를 전달
    """
    return (
        request.headers.get("X-Session-Id")
        or request.query_params.get("session_id")
        or "anon"
    )

# 라우터: 질문을 RAG / DIRECT로 분류
class RouteDecision(BaseModel):
    route : Literal["RAG", "DIRECT"]  # RAG: 판례 검색 필요, DIRECT: LLM 직접 답변
    reason : Optional[str] = None     # 분류 이유 

# 라우터 프롬프트: 판례 검색이 필요한지 판단하는 시스템 지시문
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
# structured_output으로 RouteDecision 스키마에 맞게 자동 파싱
router_chain = router_prompt | llm_router.with_structured_output(RouteDecision)

# DIRECT 답변 체인 (판례 검색 없이 LLM 직접 답변)
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
    MessagesPlaceholder("history"),  # 이전 대화 이력을 프롬프트에 자동 삽입
    ("human", "{question}"),
])

direct_chain = direct_prompt | llm

# RunnableWithMessageHistory: 세션별 대화 이력을 자동으로 관리하는 래퍼
direct_chain_with_history = RunnableWithMessageHistory(
    direct_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 비동기 헬퍼 함수
async def llm_route_decision(question: str) -> RouteDecision:
    """질문을 RAG / DIRECT 중 하나로 비동기 분류"""
    return await router_chain.ainvoke({"question": question})

async def llm_direct_answer(question: str, *, session_id: str) -> str:
    """DIRECT 경로: 히스토리를 포함해 LLM으로 직접 답변 생성"""
    hist = get_session_history(session_id)
    log.info("🕘 BEFORE DIRECT history_len=%d", len(hist.messages))
    resp = await direct_chain_with_history.ainvoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    log.info("🕙 AFTER  DIRECT history_len=%d", len(hist.messages))
    return getattr(resp, "content", str(resp))

# RAG 검색 파이프라인 
# CrossEncoder Reranker 초기화 (BAAI/bge-reranker-v2-m3: 한국어 강력 지원)
try:
    _DEVICE = "cuda" if (not USE_CPU_WHISPER and torch.cuda.is_available() and torch.backends.cudnn.is_available()) else "cpu"
except Exception:
    _DEVICE = "cpu"
cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3", device=_DEVICE, max_length=512)

class Query(BaseModel):
    question: str

def chat_rag(query: Query):
    """
    3단계 RAG 검색 파이프라인:
      Stage 1) FAISS 유사도 검색 (k=50) + 사건번호 기준 중복 제거
      Stage 2) MMR(Maximal Marginal Relevance)으로 다양성 있는 10개 선택
      Stage 3) CrossEncoder Rerank → 최종 TOP_K(3)개 정밀 정렬
    임계치(RERANK_THRESHOLD) 미달 시 "관련 판례 없음" 처리
    """
    q_text = query.question

    # Stage 1: FAISS 유사도 검색 + 사건번호 기준 중복 제거 
    candidates = vectorstore.similarity_search(q_text, k=50)
    seen = set()
    uniq_candidates = []
    for d in candidates:
        case_no = d.metadata.get("사건번호")
        if case_no in seen:
            continue
        seen.add(case_no)
        uniq_candidates.append(d)
    candidates = uniq_candidates[:50]

    # Stage 2: MMR (Maximal Marginal Relevance)
    # 관련성은 높으면서 서로 겹치지 않는 다양한 문서 10개 선택
    q_emb = np.asarray(embedding_model.embed_query(q_text), dtype=np.float32)
    doc_texts = [d.page_content for d in candidates]
    doc_embs  = np.asarray(embedding_model.embed_documents(doc_texts), dtype=np.float32)

    def _l2norm(x):
        """코사인 유사도 계산을 위한 L2 정규화"""
        n = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
        np.maximum(n, 1e-12, out=n)  # 0 나눗셈 방지
        return x / n

    q = _l2norm(q_emb.reshape(1, -1))[0]
    D = _l2norm(doc_embs)

    lambda_mult = 0.9  # 관련성(λ) vs 다양성(1-λ) 가중치 (0.9 = 관련성 중심)
    top_mmr = 10
    selected = []
    unselected = set(range(len(D)))
    sims = D @ q  # 질문과 각 문서의 코사인 유사도 벡터

    # MMR Greedy: 관련성 높고 기선택 문서와 중복 최소인 문서를 순차 선택
    while len(selected) < min(top_mmr, len(D)):
        if not selected:
            # 첫 번째: 질문과 가장 유사한 문서 선택
            i = int(np.argmax(sims))
            selected.append(i); unselected.remove(i)
            continue
        # 이미 선택된 문서들과의 최대 유사도 (중복 패널티)
        max_div = np.max(D[list(selected)] @ D[list(unselected)].T, axis=0)
        # MMR 점수 = λ × 관련성 - (1-λ) × 중복도
        mmr_scores = lambda_mult * sims[list(unselected)] - (1 - lambda_mult) * max_div
        pick_idx_in_un = int(np.argmax(mmr_scores))
        i = list(unselected)[pick_idx_in_un]
        selected.append(i); unselected.remove(i)

    stage2_docs = [candidates[i] for i in selected]

    # Stage 3: CrossEncoder Rerank 
    # 질문-문서 쌍을 직접 비교해 정밀한 관련성 점수 산출 후 TOP_K 선택
    pairs  = [(q_text, d.page_content) for d in stage2_docs]
    scores = cross_encoder.predict(pairs, batch_size=64, show_progress_bar=False)
    order  = np.argsort(scores)[::-1][:TOP_K]

    used_docs = [stage2_docs[i] for i in order]
    top_scores = [float(scores[i]) for i in order] if len(order) > 0 else []
    best_score = max(top_scores) if top_scores else -1e9

    # 임계치 미달: 관련 판례 없음 → LLM으로 일반 답변 생성 
    if not np.isfinite(best_score) or best_score < RERANK_THRESHOLD or len(used_docs) == 0:
        prompt = f"""다음 질문에 대해 내부 판례에서 뚜렷한 관련 근거를 찾지 못했습니다.
가능한 범위에서 일반적 설명을 한국어로 간단히 제시하세요.

[질문]
{q_text}
"""
        resp = llm.invoke(prompt)
        answer = getattr(resp, "content", str(resp))
        answer += "\n\n관련된 판례는 없습니다."
        return {"answer": answer, "used_files": []}

    # 원문 JSON 로딩 및 컨텍스트 구성 
    # 사건번호로 원문 JSON 파일명 목록 생성
    docs_name = []
    for d in used_docs:
        cn = d.metadata.get("사건번호")
        if cn:
            docs_name.append(f"{cn}.json")

    # RAW_DATA_ROOT 하위 디렉토리에서 해당 파일을 찾아 판시사항/판결요지/판례내용 추출
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

    # 검색된 판례 원문을 컨텍스트로 GPT에 최종 답변 생성 요청
    prompt = f"""다음은 질문에 관한 판례 발췌입니다. 이 내용을 활용해 질문에 한국어로 간결히 답하세요.
자료가 충분하지 않으면 추정/일반 설명도 허용됩니다.

{final_context}

[질문]
{q_text}
"""
    resp = llm.invoke(prompt)
    answer = getattr(resp, "content", str(resp))

    # 실제 사용된 판례 파일명을 서버에서 직접 답변에 추가 (LLM에 의존하지 않음)
    if docs_name:
        answer += "\n\n관련된 판례는 다음과 같습니다: [" + ", ".join(docs_name) + "]"
    else:
        answer += "\n\n관련된 판례는 없습니다."

    return {"answer": answer, "used_files": docs_name}

# API 엔드포인트

# 비동기 Job 저장소: {job_id: {status, question, answer, used_files, message}}
# Cloudflare 100초 타임아웃 우회를 위해 Job 큐 방식 사용
JOBS: Dict[str, Dict[str, Any]] = {}

os.makedirs("temp", exist_ok=True)  # 임시 오디오 파일 저장 디렉토리 생성

@app.post("/voice-chat")
async def voice_chat(request: Request, file: UploadFile = File(...), stt_model: WhisperModel = Depends(get_whisper)):
    """
    음성 파일 업로드 엔드포인트.
    즉시 202 + job_id를 반환하고, 백그라운드에서 전체 파이프라인을 비동기 실행:
      ffmpeg(변환) → Whisper(STT) → LLM Router(라우팅) → DIRECT or RAG(답변)

    클라이언트는 반환된 job_id로 /voice-chat/result를 폴링해 결과를 확인.
    """
    session_id = get_session_id(request)
    log.info("👤 session_id=%s", session_id)
    log.info("📥 /voice-chat 업로드: filename=%s, content_type=%s", file.filename, file.content_type)

    # 업로드 파일을 UUID 기반 임시 경로에 저장
    uid = uuid.uuid4().hex
    orig_ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
    in_path = os.path.join("temp", f"{uid}{orig_ext}")
    wav_path = os.path.join("temp", f"{uid}.wav")

    raw = await file.read()
    with open(in_path, "wb") as f:
        f.write(raw)
    log.info("📄 저장 완료: %s (%d bytes)", in_path, len(raw))

    # Job 등록 후 백그라운드 태스크로 처리 시작
    job_id = secrets.token_hex(8)
    JOBS[job_id] = {"status": "processing"}

    async def run_job():
        try:
            # Step 1: 오디오 파일 → WAV 변환 (Whisper 입력 형식)
            ffmpeg_to_wav(in_path, wav_path, sr=16000)

            # Step 2: Whisper STT → 텍스트 추출
            log.info("🗣️ STT 시작")
            segments, _ = stt_model.transcribe(wav_path)  # info(언어/확률 등 메타) 미사용
            texts = []
            for seg in segments:
                t = getattr(seg, "text", None)
                if t is None and isinstance(seg, dict):
                    t = seg.get("text")
                if t:
                    texts.append(t.strip())
            question_text = " ".join(texts) if texts else "(음성에서 텍스트를 추출하지 못했습니다.)"
            log.info("🗣️ STT 결과 길이=%d", len(question_text))

            # Step 3: LLM이 질문을 분석해 RAG / DIRECT 경로 결정
            decision = await llm_route_decision(question_text)
            log.info("🧭 라우팅: %s (%s)", decision.route, decision.reason)

            # RAG 경로일 때만 클라이언트에 "자료 검색중..." 메시지 전달
            if decision.route == "RAG":
                JOBS[job_id]["message"] = "자료 검색중..."

            # Step 4-A: DIRECT 경로 (일반 질문 → 히스토리 포함 LLM 직접 답변)
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

            # Step 4-B: RAG 경로 (법률 판례 검색 → 컨텍스트 기반 답변)
            log.info("🔎 RAG 검색 시작")
            # chat_rag는 동기 함수 → asyncio.to_thread로 이벤트 루프 블로킹 방지
            rag_resp = await asyncio.to_thread(chat_rag, Query(question=question_text))
            used = rag_resp.get("used_files", [])
            answer = rag_resp.get("answer", "")

            # RAG 답변을 히스토리에 수동 추가 (RunnableWithMessageHistory 미사용)
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
            # 처리 완료 후 임시 파일 정리 (성공/실패 무관하게 항상 실행)
            for p in (in_path, wav_path):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except:
                    pass

    asyncio.create_task(run_job())

    # 즉시 202 반환 (Cloudflare/모바일 타임아웃 회피)
    return JSONResponse(
        {"status": "processing", "job_id": job_id},
        status_code=202,
        headers={
            "Location": f"/voice-chat/result?job_id={job_id}",
            "Content-Type": "application/json; charset=utf-8"
        }
    )

@app.get("/voice-chat/result")
async def voice_chat_result(job_id: str):
    """
    비동기 작업 결과 조회 엔드포인트 (클라이언트가 1초 간격으로 폴링).
    반환 status:
      - 'processing': 아직 처리 중
      - 'done'      : 완료 (question, answer, used_files 포함)
      - 'error'     : 처리 실패 (message 포함)
    """
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse(
            {"status": "error", "message": "unknown job_id"},
            status_code=404,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    return JSONResponse(job, status_code=200, headers={"Content-Type": "application/json; charset=utf-8"})
