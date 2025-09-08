# import os
# from pathlib import Path

# base_dir = Path("data/Training/02.라벨링데이터")
# cnt = 0
# # base_dir 안의 모든 하위 디렉토리 순회
# for folder in sorted(base_dir.iterdir()):
#     if folder.is_dir():
#         # 해당 폴더 안의 .json 파일만 필터링
#         json_files = list(folder.glob("*.json"))
#         print(f"{folder.name}: {len(json_files)}개")
#         cnt += len(json_files)
# print(cnt)

import os
import json
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
import torch
import numpy as np

# 평가 데이터 로드
with open("./data/evaluate/evaldata.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# 평가 설정
k = 3
eval_count = 20000 # (필요시 조정)

# 모델 및 chunk 설정
INDEX_PATH = r"C:\Users\vkslr\Rag-Agent\voice_chatbot\server\store"
embedding_model = HuggingFaceEmbeddings(model_name=r"C:\Users\vkslr\Rag-Agent\voice_chatbot\server\model_bs32")
vectorstore = FAISS.load_local(INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)

all_docs = list(vectorstore.docstore._dict.values())  # InMemoryDocstore 기준
bm25 = BM25Retriever.from_documents(all_docs)
bm25.k = 200  # BM25 상위 50

# 2) Dense Retriever (FAISS)
# dense = vectorstore.as_retriever(search_kwargs={"k": 50})



# 4) Cross-Encoder (rerank 용)
device = "cuda" if torch.cuda.is_available() else "cpu"
cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=512)

def reciprocal_rank(results, true_ids, k=3, meta_key="사건번호"):
    """
    Top-k 결과에서 첫 번째로 등장하는 정답의 순위를 찾아 1/rank를 반환.
    정답이 없으면 0을 반환.
    """
    # enumerate는 1위부터 시작하도록 start=1
    for rank, doc in enumerate(results[:k], start=1):
        doc_id = doc.metadata.get(meta_key)
        if doc_id is None:
            continue
        if doc_id in true_ids:
            return 1.0 / rank
    return 0.0

# 평가 루프 (MRR@3)

print("start")

mrr_sum = 0.0
hit_at_k = 0  # 참고용: Recall@k도 같이 보고 싶으면 유지
for item in tqdm(eval_data[:eval_count], desc="model", ncols=80):
    query = item["query"]

    # 정답 ID 집합 만들기 (문자열/리스트 모두 대응)
    if isinstance(item["case_ids"], str):
        true_ids = {item["case_ids"]}
    else:
        true_ids = set(item["case_ids"])

    # Top-k 검색
    candidates = vectorstore.similarity_search(query, k=30)

    # 중복 제거 (사건번호 + 내용 기준)
    seen = set()
    uniq_candidates = []
    for d in candidates:
        key = (d.metadata.get("사건번호"))
        if key in seen:
            continue
        seen.add(key)
        uniq_candidates.append(d)
    candidates = uniq_candidates[:50]  # 방어적으로 50개 제한

    # Stage 2) MMR로 10개 다양성 선택 (dense 임베딩 사용)
    q_emb = np.asarray(embedding_model.embed_query(query), dtype=np.float32)
    doc_texts = [d.page_content for d in candidates]
    doc_embs = np.asarray(embedding_model.embed_documents(doc_texts), dtype=np.float32)

    # cosine 대비 정규화
    def _l2norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
        np.maximum(n, 1e-12, out=n)
        return x / n

    q = _l2norm(q_emb.reshape(1, -1))[0]
    D = _l2norm(doc_embs)

    lambda_mult = 0.8  # 관련성/다양성 트레이드오프
    top_mmr = 10
    selected = []
    unselected = set(range(len(D)))

    # MMR greedy
    sims = D @ q  # (N,)
    while len(selected) < min(top_mmr, len(D)):
        if not selected:
            i = int(np.argmax(sims))
            selected.append(i)
            unselected.remove(i)
            continue
        # 이미 선택된 것들과의 최대 유사도
        max_div = np.max(D[list(selected)] @ D[list(unselected)].T, axis=0)
        # MMR 점수 = λ * relevance - (1-λ) * diversity
        mmr_scores = lambda_mult * sims[list(unselected)] - (1 - lambda_mult) * max_div
        pick_idx_in_un = int(np.argmax(mmr_scores))
        i = list(unselected)[pick_idx_in_un]
        selected.append(i)
        unselected.remove(i)

    stage2_docs = [candidates[i] for i in selected]

    # Stage 3) Cross-Encoder로 rerank → 최종 top-3
    pairs  = [(query, d.page_content) for d in candidates]
    # 배치 크게 주면 속도 개선
    scores = cross_encoder.predict(pairs, batch_size=64, show_progress_bar=False)
    order  = np.argsort(scores)[::-1][:k]
    results = [candidates[i] for i in order]

    # RR 계산
    rr = reciprocal_rank(results, true_ids, k=k, meta_key="사건번호")
    mrr_sum += rr

    # 참고용 Hit@k (있으면 1, 없으면 0)
    if rr > 0:
        hit_at_k += 1

mrr_at_k = mrr_sum / eval_count
recall_at_k = hit_at_k / eval_count  # 참고용 출력

print(f"✅ [결과]  MRR@{k}: {mrr_at_k:.4f}")
print(f"ℹ️  (참고) Recall@{k}: {recall_at_k:.4f}")

