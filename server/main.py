import os
import uuid
import json
import logging
import traceback
import tempfile
from shutil import which

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Literal, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from faster_whisper import WhisperModel
import subprocess

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
    )

from fastapi.exceptions import HTTPException as FastAPIHTTPException

@app.exception_handler(FastAPIHTTPException)
async def http_exception_logger(request: Request, exc: FastAPIHTTPException):
    # HTTPException도 콘솔에 이유를 남김
    log.error("⚠️ HTTPException %s %s -> %s", request.method, request.url.path, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# ─────────────────────────────────────────────
# 준비
# ─────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    log.warning("⚠️ OPENAI_API_KEY가 설정되어 있지 않습니다.")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)
llm_router = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)

TOP_K = 3
# MODEL_PATH = "./model_bs32"
# INDEX_PATH = r"C:\Users\vkslr\Rag-Agent\voice_chatbot\server\store"
# RAW_DATA_ROOT = r"C:\Users\vkslr\Rag-Agent\voice_chatbot\server\data\Training\01.원천데이터"
MODEL_PATH    = os.getenv("MODEL_PATH", "/app/model_bs32")
INDEX_PATH    = os.getenv("INDEX_PATH", "/app/store")
RAW_DATA_ROOT = os.getenv("RAW_DATA_ROOT", "/app/data")

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
    return WhisperModel("large-v2", device=device, compute_type=compute_type)

stt_model = init_whisper()

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

# Chat_History
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

# Direct or Rag
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
            " 1) 특정 법령/조문/요건/구성요건 충족 여부 판단(예: '~법상 ~한 경우 처벌 가능?', '~죄 성립?')\n"
            " 2) 사실관계를 제시하고 그 사실에 부합하는 판례/사례를 요구\n"
            " 3) '관련/유사 판례', '근거/출처', '최신/최근 판례', '인용'을 명시적으로 요구\n"
            " 4) 사건번호·사건명·선고일·법원 등 판례 식별정보를 요구/언급\n"
            " 5) 판결문 내용·판시사항·판결요지 원문 확인이 필요한 질문\n"
            " 6) 행정/형사/민사에서 제재·처벌·과태료·벌금·무효/취소 가능성 판단 질의\n"
            " 7) 특정 법령명(예: 식품위생법, 약사법, 염관리법 등)이나 조문을 직접 언급\n"
            "위에 해당하지 않고 일반 개념 설명(용어 정의, 절차 개요 등)만으로 충분하면 DIRECT를 선택하세요.\n"
            "- 필요하면 정확히 다음 형식으로만 출력: {{\"route\":\"RAG\",\"reason\":\"...\"}}\n"
            "- 필요 없으면 정확히 다음 형식으로만 출력: {{\"route\":\"DIRECT\",\"reason\":\"...\"}}\n"
            "추가 설명 금지. 모호하거나 확신이 없으면 RAG를 선택하세요."
        ),
    ),
    # (선택) 라우터도 직전 문맥을 참고하려면 history를 포함하세요:
    # MessagesPlaceholder("history"),
    ("human", "질문:\n{question}"),
])
router_chain = router_prompt | llm_router.with_structured_output(RouteDecision)

# ✅ 직접 답변 프롬프트 (한국어)
direct_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "당신은 도움을 주는 조수입니다. 외부/사설 자료(사용자 파일·코드·문서 등)를 조회하지 말고, "
            "일반 지식과 주어진 맥락만으로 간결하고 정확하게 답변하세요. "
            "만약 사설 자료 없이는 확정하기 어려운 부분이 있다면, 그 사실을 한 줄로만 짧게 밝혀주세요."
        ),
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

direct_chain = direct_prompt | llm

direct_chain_with_history = RunnableWithMessageHistory(
    direct_chain,
    get_session_history,          # 세션ID → 히스토리 객체
    input_messages_key="question",
    history_messages_key="history",
)

# 비동기 헬퍼 (그대로 사용)
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
class Query(BaseModel):
    question: str

def chat_rag(query: Query):
    docs = vectorstore.similarity_search(query.question, k=TOP_K)
    docs_name = [f"{d.metadata.get('사건번호')}.json" for d in docs if d.metadata.get("사건번호")]

    contexts = []
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
    prompt = f"""다음은 질문에 관한 최대 {TOP_K}개 판례 발췌입니다. 이 내용을 활용해 질문에 답하세요.
답변 마지막 줄에 반드시 다음 형식으로 출력하세요:
"관련된 판례는 다음과 같습니다: [파일명1, 파일명2, ...]"

{final_context}

[질문]
{query.question}
"""
    resp = llm.invoke(prompt)
    return {"answer": resp.content, "used_files": docs_name}

# ─────────────────────────────────────────────
# 라우트
# ─────────────────────────────────────────────
os.makedirs("temp", exist_ok=True)

@app.post("/voice-chat")
async def voice_chat(request: Request, file: UploadFile = File(...)):
    session_id = get_session_id(request)
    log.info("👤 session_id=%s", session_id)

    log.info("📥 /voice-chat 업로드: filename=%s, content_type=%s", file.filename, file.content_type)

    uid = uuid.uuid4().hex
    orig_ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
    in_path = os.path.join("temp", f"{uid}{orig_ext}")
    wav_path = os.path.join("temp", f"{uid}.wav")

    try:
        # 1) 저장
        raw = await file.read()
        with open(in_path, "wb") as f:
            f.write(raw)
        log.info("📄 저장 완료: %s (%d bytes)", in_path, len(raw))

        # 2) 오디오 → WAV(ffmpeg)
        ffmpeg_to_wav(in_path, wav_path, sr=16000)

        # 3) STT
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

        # Decision
        decision = await llm_route_decision(question_text)
        log.info("🧭 라우팅: %s (%s)", decision.route, decision.reason)

        # 4) RAG
        if decision.route == "RAG":
            log.info("🔎 RAG 검색 시작")
            rag_resp = chat_rag(Query(question=question_text))
            used = rag_resp.get("used_files", [])
            answer = rag_resp.get("answer", "")
            log.info("✅ RAG 완료, 사용 파일: %s", used)

            hist = get_session_history(session_id)
            log.info("🕘 BEFORE RAG   history_len=%d", len(hist.messages))
            hist.add_user_message(question_text)
            hist.add_ai_message(answer)
            log.info("🕙 AFTER  RAG   history_len=%d", len(hist.messages))

            return {"question": question_text, "answer": rag_resp["answer"], "used_files": used}

        if decision.route == "DIRECT":
            log.info("💬 DIRECT 답변 생성")
            answer = await llm_direct_answer(question_text, session_id=session_id)
            return {"question": question_text, "answer": answer, "used_files": []}
    
    except Exception as e:
        # 여기서 잡히면 위의 all_exception_handler가 스택까지 찍어줌
        raise HTTPException(status_code=500, detail=f"voice-chat failed: {e}")

    finally:
        for p in (in_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass
