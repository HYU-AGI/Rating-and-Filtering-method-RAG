import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Moved to command line argument
import asyncio
import torch

# Reranker 라이브러리 임포트
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc

# LightRAG의 핵심 클래스
from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger

# Sentence-BERT 같은 임베딩 모델 로딩용
from transformers import AutoTokenizer, AutoModel

# LightRAG 내부 로깅 설정
setup_logger("lightrag", level="INFO")

# 작업 디렉토리 설정
WORKING_DIR = "./rag_storage_qwen"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# --- 1. 사용자 정의 Reranker 로직 구현 (Async/GPU 버전) ---

W_ROUGE = 0.3
W_BERTSCORE = 0.7
rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def reranker_sync_job(query: str, documents: list[str]) -> list[dict]: # 반환 타입 힌트 변경
    """[동기 함수] ROUGE + BERTScore 계산을 수행 (별도 스레드에서 실행됨)"""
    
    print(f"\n[Custom Reranker (Sync Job)] ROUGE+BERTScore로 {len(documents)}개 청크 재정렬...")
    
    if not documents:
        return []

    chunk_contents = documents
    
    # --- ROUGE 점수 계산 ---
    rouge_scores_f1 = []
    for content in chunk_contents:
        score = rouge_scorer_instance.score(query, content)
        rouge_scores_f1.append(score['rougeL'].fmeasure) 

    # --- BERTScore 계산 ---
    queries_list = [query] * len(chunk_contents)
    
    print("[Custom Reranker (Sync Job)] BERTScore를 GPU(cuda)에서 실행합니다...")
    P, R, F1 = bert_score_calc(queries_list, chunk_contents, 
                               model_type="bert-base-multilingual-cased",
                               lang="en", 
                               verbose=False,
                               device="cuda") # GPU 사용
    
    bert_scores_f1 = F1.tolist()

    # --- 최종 점수 계산 및 정렬 ---
    scored_chunks = []
    print("[Custom Reranker (Sync Job)] 개별 점수:")
    for i, content in enumerate(chunk_contents): 
        rouge_f1 = rouge_scores_f1[i]
        bert_f1 = bert_scores_f1[i]
        total_score = (W_ROUGE * rouge_f1) + (W_BERTSCORE * bert_f1)
        
        print(f"  Chunk {i}: ROUGE-L={rouge_f1:.4f}, BERT-F1={bert_f1:.4f} -> Total={total_score:.4f}")
        scored_chunks.append((total_score, content))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    print("[Custom Reranker (Sync Job)] 재정렬 완료.")
    
    # ---------------------------------------------------------------------
    # FIX: 'str' 리스트 대신 [{'content': '...'}, ...] 형태의 'dict' 리스트 반환
    # ---------------------------------------------------------------------
    print("[Custom Reranker (Sync Job)] 정렬된 dict 리스트 반환 중...")
    return [{"content": content} for score, content in scored_chunks]

async def my_custom_reranker(query: str, documents: list[str], **kwargs) -> list[dict]:
    """[Async 래퍼] LightRAG가 호출할 비동기 Reranker 함수"""
    # reranker_sync_job이 이제 list[dict]를 반환합니다.
    return await asyncio.to_thread(reranker_sync_job, query, documents)


async def initialize_rag():
    """Qwen + Sentence-BERT 기반 LightRAG 인스턴스 초기화 (GPU)"""

    # 1) 임베딩 모델 설정
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model: {embed_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name).to("cuda") # GPU로

    # 2) LLM 모델 선택
    llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"LLM model: {llm_model_name}")

    # 3) LightRAG 인스턴스 생성
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name=llm_model_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=tokenizer,
                embed_model=embed_model
            )
        ),
        
        # Reranker 함수 주입
        rerank_model_func=my_custom_reranker,

        # LLM이 GPU를 사용하도록 "auto" 설정
        llm_model_kwargs={"device_map": "auto"},
    )

    # 4) 스토리지 초기화
    await rag.initialize_storages()

    # 5) 파이프라인 상태 초기화
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # 0) RAG 시스템 초기화
        print("Initializing LightRAG...")
        rag = await initialize_rag()

        # 1) 샘플 문서 삽입 (light rag 예시)
        sample_text = """
        LightRAG is a fast and simple RAG system.
        It efficiently processes documents using graph-based retrieval.
        It supports various LLMs and embedding models.
        You can run it locally without OpenAI API.
        The system uses knowledge graphs to enhance retrieval accuracy.
        """

        print("\nInserting document...")
        await rag.ainsert(sample_text)

        # 2) 질의 실행
        print("\nExecuting query...")
        question = "What are the main features of LightRAG?"

        # Reranker 활성화
        param = QueryParam(
            mode="hybrid",
            enable_rerank=True,
            chunk_top_k=2
        )

        result = await rag.aquery(
            question,
            param=param
        )

        # 3) 결과 출력
        print("\n" + "="*50)
        print("Question:", question)
        print(f"(QueryParam: mode={param.mode}, chunk_top_k={param.chunk_top_k}, rerank=True)")
        print("="*50)
        print("Answer:", result)
        print("="*50)

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 4) 스토리지 정리
        if 'rag' in locals():
            await rag.finalize_storages()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LightRAG Quick Local Example")
    parser.add_argument(
        "--cuda_device",
        type=str,
        default=None,
        help="CUDA device to use (e.g., '0' or '0,1'). If not specified, uses system default."
    )
    args = parser.parse_args()
    
    # Set CUDA device if specified
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    asyncio.run(main())