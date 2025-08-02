import logging
import json
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from tenacity import Retrying, stop_after_attempt, wait_exponential
from config.settings import Settings
from utils.cache import Cache
from services.audit.file_logger import FileLogger
from core.monitoring.metrics import retriever_requests, retriever_latency
from schemas.models import Regulation
from utils.exceptions import RetrieverError

class Retriever:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        self.cache = Cache()
        self.audit_logger = FileLogger()
        self.vectorstore = None
        self.retry_policy = Retrying(
            stop=stop_after_attempt(self.settings.retriever_max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True
        )
        self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> None:
        try:
            self.vectorstore = FAISS.from_texts(
                texts=[""],
                embedding=OllamaEmbeddings(
                    model=self.settings.ollama_embeddings_model,
                    base_url=self.settings.ollama_base_url
                ),
                metadatas=[{}]
            )
            try:
                self.vectorstore = FAISS.load_local(
                    self.settings.faiss_index_path,
                    embeddings=OllamaEmbeddings(
                        model=self.settings.ollama_embeddings_model,
                        base_url=self.settings.ollama_base_url
                    ),
                    allow_dangerous_deserialization=True
                )
                self.logger.info(f"Loaded FAISS index from {self.settings.faiss_index_path}")
            except FileNotFoundError:
                self.logger.info("No existing FAISS index found; initializing new index")
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS vector store: {str(e)}")
            self.audit_logger.log_event(
                event_type="retriever_init_error",
                details={"error": str(e)}
            )
            raise RetrieverError(f"Failed to initialize FAISS: {str(e)}")

    async def get_relevant_rules(self, query: str, k: int = 3) -> List[Regulation]:
        retriever_requests.inc()
        with retriever_latency.time():
            cache_key = f"retriever:{hash(query)}:{k}"
            
            cached = await self.cache.get(cache_key)
            if cached:
                self.logger.debug(f"Cache hit for query: {query}")
                return [Regulation.parse_raw(item) for item in json.loads(cached)]
            
            try:
                for attempt in self.retry_policy:
                    with attempt:
                        results = self.vectorstore.similarity_search_with_score(
                            query=query,
                            k=k
                        )
                        regulations = [
                            Regulation(
                                text=doc.page_content,
                                metadata=doc.metadata,
                                score=score
                            )
                            for doc, score in results
                        ]
                        self.logger.info(f"Retrieved {len(regulations)} regulations for query: {query}")
                        self.audit_logger.log_event(
                            event_type="retriever_query",
                            details={
                                "query": query,
                                "results": [reg.dict() for reg in regulations]
                            }
                        )
                        
                        await self.cache.set(
                            cache_key,
                            json.dumps([reg.json() for reg in regulations]),
                            ttl=self.settings.retriever_cache_ttl
                        )
                        return regulations
            except Exception as e:
                self.logger.error(f"Retrieval failed for query '{query}': {str(e)}")
                self.audit_logger.log_event(
                    event_type="retriever_query_error",
                    details={"query": query, "error": str(e)}
                )
                raise RetrieverError(f"Retrieval failed: {str(e)}")

    async def add_regulations(self, texts: List[str], metadatas: List[Dict]) -> None:
        try:
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            self.vectorstore.save_local(self.settings.faiss_index_path)
            self.logger.info(f"Added {len(texts)} regulations to FAISS index")
            self.audit_logger.log_event(
                event_type="retriever_add_regulations",
                details={"count": len(texts)}
            )
        except Exception as e:
            self.logger.error(f"Failed to add regulations: {str(e)}")
            self.audit_logger.log_event(
                event_type="retriever_add_error",
                details={"error": str(e)}
            )
            raise RetrieverError(f"Failed to add regulations: {str(e)}")