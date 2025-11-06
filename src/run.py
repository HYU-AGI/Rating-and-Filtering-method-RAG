import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  
import asyncio
import torch
import argparse # 1. argparse 모듈 임포트

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


# --- 1. CustomReranker 클래스 정의 (가중치를 인자로 받음) ---

class CustomReranker:
    def __init__(self, rouge_weight: float, bert_weight: float):
        self.rouge_weight = rouge_weight
        self.bert_weight = bert_weight
        self.rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        print(f"[CustomReranker] ROUGE 가중치: {self.rouge_weight}, BERTScore 가중치: {self.bert_weight}로 초기화됨")

    def _reranker_sync_job(self, query: str, documents: list[str]) -> list[dict]:
        """[동기 함수] ROUGE + BERTScore 계산을 수행 (별도 스레드에서 실행됨)"""
        
        print(f"\n[Custom Reranker (Sync Job)] ROUGE+BERTScore로 {len(documents)}개 청크 재정렬...")
        
        if not documents:
            return []

        chunk_contents = documents
        
        # --- ROUGE 점수 계산 ---
        rouge_scores_f1 = []
        for content in chunk_contents:
            score = self.rouge_scorer_instance.score(query, content)
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
            
            # ---------------------------------------------------------------------
            # FIX: 생성자에서 받은 가중치(self.xxx)를 사용
            # ---------------------------------------------------------------------
            total_score = (self.rouge_weight * rouge_f1) + (self.bert_weight * bert_f1)
            
            print(f"  Chunk {i}: ROUGE-L={rouge_f1:.4f}, BERT-F1={bert_f1:.4f} -> Total={total_score:.4f}")
            scored_chunks.append((total_score, content))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        print("[Custom Reranker (Sync Job)] 재정렬 완료.")
        print("[Custom Reranker (Sync Job)] 정렬된 dict 리스트 반환 중...")
        return [{"content": content} for score, content in scored_chunks]

    async def __call__(self, query: str, documents: list[str], **kwargs) -> list[dict]:
        """[Async 래퍼] LightRAG가 호출할 비동기 Reranker 함수 (클래스 인스턴스가 함수처럼 호출됨)"""
        return await asyncio.to_thread(self._reranker_sync_job, query, documents)


# 2. initialize_rag가 Reranker 인스턴스를 받도록 수정
async def initialize_rag(reranker_instance: CustomReranker):
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
        
        # Reranker 함수 대신 Reranker *인스턴스* 주입
        rerank_model_func=reranker_instance,

        # LLM이 GPU를 사용하도록 "auto" 설정
        llm_model_kwargs={"device_map": "auto"},
    )

    # 4) 스토리지 초기화
    await rag.initialize_storages()

    # 5) 파이프라인 상태 초기화
    await initialize_pipeline_status()

    return rag


# 3. main 함수가 파일 경로와 Reranker 인스턴스를 받도록 수정
async def main(txt_file_path: str, reranker_instance: CustomReranker):
    try:
        # 0) RAG 시스템 초기화 (Reranker 인스턴스 전달)
        print("Initializing LightRAG...")
        rag = await initialize_rag(reranker_instance)

        # 1) 인자로 받은 'txt_file_path' 사용
        print(f"\nLoading document from {txt_file_path}...")
        try:
            with open(txt_file_path, "r", encoding="utf-8") as f:
                sample_text = f.read()
        except FileNotFoundError:
            print(f"ERROR: '{txt_file_path}' 파일을 찾을 수 없습니다.")
            return

        print("\nInserting document...")
        await rag.ainsert(sample_text)

        # 2) 사용자로부터 직접 질문 입력받기
        print("\nExecuting query...")
        question = input("Please enter your question: ")

        # Reranker 활성화
        param = QueryParam(
            mode="hybrid",
            enable_rerank=True,
            chunk_top_k=3
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
    # 4. argparse를 사용하여 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description="LightRAG Reranking Example with Custom Weights")
    parser.add_argument(
        "--file", 
        type=str, 
        required=True, 
        help="Path to the .txt file to insert."
    )
    parser.add_argument(
        "--rouge_weight",
        type=float,
        default=0.3, # 기본값 0.3
        help="Weight for ROUGE-L score (default: 0.3)"
    )
    parser.add_argument(
        "--bert_weight",
        type=float,
        default=0.7, # 기본값 0.7
        help="Weight for BERTScore (default: 0.7)"
    )
    args = parser.parse_args()
    
    # 5. 파싱된 가중치로 Reranker 인스턴스 생성
    reranker = CustomReranker(
        rouge_weight=args.rouge_weight,
        bert_weight=args.bert_weight
    )
    
    # 6. 인스턴스와 파일 경로를 main 함수에 전달
    asyncio.run(main(txt_file_path=args.file, reranker_instance=reranker))