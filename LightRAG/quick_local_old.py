import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  
import asyncio

# LightRAG의 핵심 클래스
# - LightRAG: RAG 파이프라인 전체(삽입, 검색, KG, 벡터, rerank, LLM 호출 등)
# - QueryParam: 검색 시에 rating/filtering 관련 파라미터를 포함한 설정 객체
from lightrag import LightRAG, QueryParam

# HF(HuggingFace) 기반 LLM/Embedding 헬퍼
from lightrag.llm.hf import hf_model_complete, hf_embed

# Knowledge Graph 파이프라인에서 공유되는 상태 초기화용
# (history_messages 같은 것들) – 이걸 안 하면 query 시에 KeyError 뜰 수 있음
from lightrag.kg.shared_storage import initialize_pipeline_status

# EmbeddingFunc: LightRAG에 embedding 함수를 주입할 때 사용하는 래퍼
# setup_logger: LightRAG 전반의 로그 설정
from lightrag.utils import EmbeddingFunc, setup_logger

# Sentence-BERT 같은 임베딩 모델 로딩용
from transformers import AutoTokenizer, AutoModel

# LightRAG 내부 로깅 설정
setup_logger("lightrag", level="INFO")

# 작업 디렉토리 설정 (Knowledge Graph, VectorDB, KV Storage, Pipeline status 등
# 모든 스토리지가 이 디렉토리 아래에 파일 형태로 생성됨)
WORKING_DIR = "./rag_storage_qwen"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    """Qwen + Sentence-BERT 기반 LightRAG 인스턴스 초기화"""

    # 1) 임베딩 모델 설정
    # - 여기서 고른 모델의 차원 수(384)를 EmbeddingFunc에 전달해야 함
    # - 이 임베딩은 retrieval 단계에서 "semantic similarity 기반 rating"의 핵심 스코어로 사용됨
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"Loading embedding model: {embed_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name)

    # 2) LLM 모델 선택
    # - retrieve된 문서(혹은 chunk)를 최종 context로 만들어 답변 생성할 때 사용
    # - rating/filtering은 "어떤 chunk를 이 LLM에게 넘길 것인가?"를 결정하는 과정이라고 보면 됨
    llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"LLM model: {llm_model_name}")

    # 3) LightRAG 인스턴스 생성
    #    여기서 중요한 것은:
    #      - working_dir: RAG 전체 상태 저장 위치
    #      - llm_model_func, llm_model_name: LLM 호출 설정
    #      - embedding_func: 문서 chunk 및 쿼리를 벡터화할 함수
    #
    #  [Rating & Filtering 관점 설명]
    #   - 최초 retrieve 단계에서:
    #       * embedding_func로 문서 chunk와 query를 벡터화
    #       * cosine similarity 등으로 1차 rating을 수행 (벡터 검색)
    #       * KG 기반 검색(global/hybrid 모드)와 합쳐서 "후보 chunk 리스트"를 만듦
    #   - 그 다음 단계에서:
    #       * QueryParam의 top_k, chunk_top_k, enable_rerank 등에 따라
    #         후보를 정렬하고 상위 N개만 남기는 filtering이 수행됨
    #   - 실제 rating/rerank 알고리즘 구현은 LightRAG 내부(query 파이프라인, rerank 모듈)에 있음.
    rag = LightRAG(
        working_dir=WORKING_DIR,

        # Qwen 계열 HF 모델을 사용해서 답변을 생성하는 함수
        llm_model_func=hf_model_complete,
        llm_model_name=llm_model_name,

        # Sentence-BERT 기반 임베딩 함수 설정
        # embedding_dim=384: all-MiniLM-L6-v2 임베딩 차원 수
        # func: 텍스트 리스트 -> numpy array (N x 384) 로 변환
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=tokenizer,
                embed_model=embed_model
            )
        ),

        # [선택적으로 rating & filtering 강화를 위한 옵션 예시 - 참고용]
        # rerank_model_func=my_custom_rerank 또는 cohere_rerank/jina_rerank 등을 넘기면
        # 1차 retrieve 후에 "추가적인 rating & filtering" 단계(rerank)가 켜짐.
        #
        # rerank_model_func=my_custom_rerank,
    )

    # 4) 스토리지 초기화 (반드시 필요)
    # - KV, VectorDB, Graph, Pipeline status 등 모든 내부 저장소를 열고(또는 새로 만들고)
    #   사용가능한 상태로 만드는 단계
    await rag.initialize_storages()

    # 5) 파이프라인 상태 초기화
    # - shared_storage에 있는 history_messages, node 상태 등 초기값을 세팅
    # - query 시에 상태가 없는 문제를 방지하는 역할
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # 0) RAG 시스템 초기화
        print("Initializing LightRAG...")
        rag = await initialize_rag()

        # 1) 샘플 문서 삽입
        #   - 이 텍스트가 chunk 단위로 자르고
        #   - embedding_func로 벡터화되어 VectorDB에 저장되고
        #   - 동시에 KG 추출(엔티티/관계) 및 그래프에 반영됨
        #
        #   [Rating & Filtering 관점 설명]
        #   - 이 단계에서는 주로 "retrieval 대상 후보 풀"을 만드는 작업이고,
        #   - 실제 rating/filtering은 query 시점(aquery)에서 수행됨
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

        # [중요] QueryParam: rating & filtering이 실제로 관여하는 지점
        #
        # - mode="hybrid":
        #     * local(순수 벡터 검색) + global(KG 기반 검색)을 섞어서 후보를 뽑는 모드
        #     * 내부적으로는 두 소스에서 가져온 chunk에 스코어를 매기고 합쳐서 정렬
        #
        # - top_k, chunk_top_k, enable_rerank 등의 필드는 QueryParam에 정의되어 있고,
        #   여기서 값을 지정하지 않으면 LightRAG의 기본값(TOP_K, CHUNK_TOP_K, COSINE_THRESHOLD 등 env 기반)을 사용
        #
        #   예를 들어, 다음처럼 쓰면:
        #
        #   param=QueryParam(
        #       mode="hybrid",
        #       top_k=40,          # KG에서 가져올 엔티티/릴레이션 수 (1차 후보 수)
        #       chunk_top_k=8,     # rating/rerank 이후 최종적으로 남겨질 chunk 수
        #       enable_rerank=True # rerank_model_func가 설정되어 있으면 활성화
        #   )
        #
        #   이런 값들을 바꾸는 것만으로도
        #   "얼마나 많은 문서를 가져와서, 얼마나 공격적으로 필터링할 것인가"를 튜닝할 수 있음.
        #
        #   지금은 예제라서 mode만 지정하고 나머지는 기본값을 사용.
        result = await rag.aquery(
            question,
            param=QueryParam(mode="hybrid")
        )

        # 3) 결과 출력
        print("\n" + "="*50)
        print("Question:", question)
        print("="*50)
        print("Answer:", result)
        print("="*50)

    except Exception as e:
        # 에러 발생 시 로그 및 traceback 출력
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 4) 스토리지 정리
        # - 파일 핸들, DB 커넥션 등 정리
        if 'rag' in locals():
            await rag.finalize_storages()


if __name__ == "__main__":
    # 비동기 main 실행
    asyncio.run(main())