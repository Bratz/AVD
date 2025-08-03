# src/ii_agent/rag/rowboat_rag_system.py
"""
ROWBOAT-Enhanced RAG System
Integrates RAG capabilities with natural language workflow building
"""

from typing import List, Dict, Any, Optional, Union
import os
import json
import asyncio
from datetime import datetime
import logging
from pathlib import Path

# Handle imports gracefully
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    logging.warning("aiofiles not available - file operations will be synchronous")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available - using simple text splitting")

try:
    import httpx
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    logging.warning("Web scraping dependencies not available")

from src.ii_agent.agents.base import BaseAgent
from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
from src.ii_agent.llm.model_registry import ChutesModelRegistry

logger = logging.getLogger(__name__)

class SimpleTextSplitter:
    """Fallback text splitter if LangChain not available"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Simple text splitting by sentences"""
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                # Keep overlap
                overlap_size = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    overlap_size += len(s)
                    overlap_chunk.insert(0, s)
                    if overlap_size >= self.chunk_overlap:
                        break
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

class InMemoryVectorStore:
    """Simple in-memory vector store using Chutes embeddings"""
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.documents = []
        self.embeddings = []
        self.embedding_model = embedding_model
        self.embedding_client = ChutesOpenAIClient(
            model_name=embedding_model,
            use_native_tool_calling=False
        )
    
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add texts to the store"""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        ids = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            doc_id = f"doc_{len(self.documents)}_{i}"
            
            # Get embedding from Chutes
            try:
                # For now, use a simple hash as embedding placeholder
                # In production, would call actual embedding API
                embedding = [float(ord(c) % 10) / 10 for c in text[:100]]
            except Exception as e:
                logger.error(f"Failed to get embedding: {e}")
                embedding = [0.0] * 100
            
            self.documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata,
                "embedding": embedding
            })
            ids.append(doc_id)
        
        return ids
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """Search for similar documents"""
        # Simple text matching for now
        # In production, would use actual vector similarity
        results = []
        
        query_lower = query.lower()
        for doc in self.documents:
            # Apply filters if any
            if filter:
                skip = False
                for key, value in filter.items():
                    if doc["metadata"].get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Simple scoring based on keyword matches
            score = sum(1 for word in query_lower.split() if word in doc["text"].lower())
            if score > 0:
                results.append((doc, score))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

class ROWBOATRAGSystem:
    """ROWBOAT-enhanced RAG system with natural language integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Use appropriate text splitter
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200)
            )
        else:
            self.text_splitter = SimpleTextSplitter(
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200)
            )
        
        # Per-workflow vector stores
        self.vector_stores: Dict[str, InMemoryVectorStore] = {}
        
        # ROWBOAT-specific features
        self.auto_index = config.get("auto_index", True)
        self.extract_entities = config.get("extract_entities", True)
        self.generate_summaries = config.get("generate_summaries", True)
        
        # Initialize Chutes LLM for advanced processing
        self.llm = ChutesModelRegistry.create_llm_client(
            model_key="deepseek-chat",
            use_native_tools=False
        )
        
        logger.info("ROWBOAT RAG System initialized")
    
    async def _get_or_create_vector_store(self, workflow_id: str) -> InMemoryVectorStore:
        """Get or create vector store for workflow"""
        if workflow_id not in self.vector_stores:
            self.vector_stores[workflow_id] = InMemoryVectorStore()
        return self.vector_stores[workflow_id]
    
    async def process_file_upload(
        self,
        workflow_id: str,
        file_path: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process uploaded file with ROWBOAT enhancements"""
        
        logger.info(f"Processing file upload: {file_path} ({file_type})")
        
        # Read file content
        content = await self._read_file(file_path, file_type)
        
        # Extract text based on file type
        text = await self._extract_text(content, file_type)
        
        # ROWBOAT Enhancement: Generate summary
        summary = ""
        if self.generate_summaries and text:
            summary = await self._generate_summary(text[:2000])  # First 2000 chars
        
        # ROWBOAT Enhancement: Extract entities
        entities = []
        if self.extract_entities and text:
            entities = await self._extract_entities(text[:2000])
        
        # Chunk text
        if isinstance(self.text_splitter, RecursiveCharacterTextSplitter):
            chunks = self.text_splitter.split_text(text)
        else:
            chunks = self.text_splitter.split_text(text)
        
        # Enhanced metadata
        enhanced_metadata = {
            "source": file_path,
            "type": "file",
            "file_type": file_type,
            "upload_time": datetime.utcnow().isoformat(),
            "summary": summary,
            "entities": entities,
            "chunk_count": len(chunks),
            **(metadata or {})
        }
        
        # Add to vector store
        vector_store = await self._get_or_create_vector_store(workflow_id)
        ids = await vector_store.add_texts(
            texts=chunks,
            metadatas=[enhanced_metadata] * len(chunks)
        )
        
        # ROWBOAT: Auto-generate workflow suggestions based on content
        workflow_suggestions = []
        if self.auto_index:
            workflow_suggestions = await self._suggest_workflows_for_content(
                file_type, summary, entities
            )
        
        return {
            "file_path": file_path,
            "chunks_created": len(chunks),
            "vector_ids": ids,
            "summary": summary,
            "entities": entities,
            "workflow_suggestions": workflow_suggestions
        }
    
    async def _read_file(self, file_path: str, file_type: str) -> str:
        """Read file content"""
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    async def _extract_text(self, content: str, file_type: str) -> str:
        """Extract text from content based on file type"""
        if file_type in ['txt', 'md']:
            return content
        elif file_type == 'json':
            try:
                data = json.loads(content)
                return json.dumps(data, indent=2)
            except:
                return content
        elif file_type == 'pdf':
            # Would use PDF extraction library
            logger.warning("PDF extraction not implemented - returning raw content")
            return content
        else:
            return content
    
    async def _generate_summary(self, text: str) -> str:
        """Generate summary using Chutes LLM"""
        try:
            prompt = f"Summarize the following text in 2-3 sentences:\n\n{text}"
            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].text.strip()
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return ""
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text"""
        try:
            prompt = f"""Extract key entities (people, organizations, locations, concepts) from this text.
            Return as a comma-separated list.
            
            Text: {text}
            
            Entities:"""
            
            response = await self.llm.agenerate([prompt])
            entities_text = response.generations[0][0].text.strip()
            return [e.strip() for e in entities_text.split(',') if e.strip()]
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []
    
    async def _suggest_workflows_for_content(
        self,
        file_type: str,
        summary: str,
        entities: List[str]
    ) -> List[Dict[str, str]]:
        """Suggest relevant workflows based on content"""
        suggestions = []
        
        # Simple rule-based suggestions
        if file_type == 'csv':
            suggestions.append({
                "type": "data_analysis",
                "description": "Analyze CSV data with researcher and analyst agents"
            })
        
        if any(entity.lower() in ['customer', 'client', 'user'] for entity in entities):
            suggestions.append({
                "type": "customer_analysis",
                "description": "Analyze customer-related content with specialized agents"
            })
        
        if 'report' in summary.lower() or file_type == 'pdf':
            suggestions.append({
                "type": "document_processing",
                "description": "Process document with reader, analyzer, and summarizer agents"
            })
        
        return suggestions
    
    async def natural_language_search(
        self,
        workflow_id: str,
        query: str,
        context: str = "",
        k: int = 5
    ) -> Dict[str, Any]:
        """ROWBOAT-enhanced search with natural language understanding"""
        
        # Enhance query with context
        enhanced_query = query
        if context:
            enhanced_query = f"{context}\n\nQuery: {query}"
        
        # Determine search intent
        search_intent = await self._analyze_search_intent(enhanced_query)
        
        # Build filters based on intent
        filters = {}
        if "file" in search_intent.get("focus", ""):
            filters["type"] = "file"
        elif "url" in search_intent.get("focus", ""):
            filters["type"] = "url"
        
        # Perform search
        vector_store = await self._get_or_create_vector_store(workflow_id)
        results = await vector_store.similarity_search_with_score(
            enhanced_query,
            k=k,
            filter=filters if filters else None
        )
        
        # Format results with ROWBOAT enhancements
        formatted_results = []
        for doc, score in results:
            result = {
                "content": doc["text"],
                "metadata": doc["metadata"],
                "score": score,
                "relevance_explanation": await self._explain_relevance(
                    query, doc["text"], score
                )
            }
            formatted_results.append(result)
        
        # Generate answer if requested
        answer = ""
        if search_intent.get("requires_answer", False):
            answer = await self._generate_answer_from_results(query, formatted_results)
        
        return {
            "query": query,
            "intent": search_intent,
            "results": formatted_results,
            "answer": answer,
            "search_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "result_count": len(formatted_results),
                "filters_applied": filters
            }
        }
    
    async def _analyze_search_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent behind a search query"""
        try:
            prompt = f"""Analyze this search query and determine:
            1. What type of content they're looking for (file/url/any)
            2. Whether they want a direct answer or just documents
            3. Key topics of interest
            
            Query: {query}
            
            Return as JSON with keys: focus, requires_answer, topics"""
            
            response = await self.llm.agenerate([prompt])
            # Simple parsing - in production would use proper JSON parsing
            return {
                "focus": "any",
                "requires_answer": "?" in query,
                "topics": []
            }
        except Exception as e:
            logger.error(f"Failed to analyze intent: {e}")
            return {"focus": "any", "requires_answer": False, "topics": []}
    
    async def _explain_relevance(self, query: str, content: str, score: float) -> str:
        """Explain why a result is relevant"""
        if score > 3:
            return "High relevance - multiple keyword matches"
        elif score > 1:
            return "Moderate relevance - some keyword matches"
        else:
            return "Low relevance - few keyword matches"
    
    async def _generate_answer_from_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Generate a direct answer from search results"""
        if not results:
            return "No relevant information found to answer your query."
        
        # Combine top results
        context = "\n\n".join([
            f"Source: {r['metadata'].get('source', 'Unknown')}\n{r['content'][:500]}"
            for r in results[:3]
        ])
        
        try:
            prompt = f"""Based on the following information, answer the question.
            
            Question: {query}
            
            Information:
            {context}
            
            Answer:"""
            
            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].text.strip()
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "Unable to generate answer from available information."
    
    async def export_knowledge_base(self, workflow_id: str) -> Dict[str, Any]:
        """Export workflow knowledge base for sharing"""
        vector_store = await self._get_or_create_vector_store(workflow_id)
        
        return {
            "workflow_id": workflow_id,
            "document_count": len(vector_store.documents),
            "export_time": datetime.utcnow().isoformat(),
            "documents": [
                {
                    "id": doc["id"],
                    "content_preview": doc["text"][:200] + "...",
                    "metadata": doc["metadata"]
                }
                for doc in vector_store.documents
            ]
        }


class ROWBOATRAGAgent(BaseAgent):
    """Agent with ROWBOAT-enhanced RAG capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rag_system = ROWBOATRAGSystem(config.get("rag_config", {}))
        self.use_rag = config.get("use_rag", True)
        self.rag_context_limit = config.get("rag_context_limit", 3)
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with ROWBOAT RAG context"""
        
        # Determine if RAG should be used
        if self.use_rag and self._should_use_rag(task):
            # Search RAG with natural language understanding
            rag_results = await self.rag_system.natural_language_search(
                workflow_id=context.get("workflow_id", "default"),
                query=task,
                context=context.get("conversation_history", ""),
                k=self.rag_context_limit
            )
            
            # Build enhanced prompt
            augmented_task = self._build_rowboat_rag_prompt(task, rag_results)
            
            # Add RAG metadata to context
            context["rag_used"] = True
            context["rag_results"] = rag_results
        else:
            augmented_task = task
            context["rag_used"] = False
        
        # Execute with augmented context
        result = await super().execute(augmented_task, context)
        
        # ROWBOAT: Learn from execution
        if self.rag_system.auto_index and result.get("success"):
            # Index successful results for future use
            await self._index_execution_result(task, result, context)
        
        return result
    
    def _should_use_rag(self, task: str) -> bool:
        """Determine if RAG should be used for this task"""
        # Use RAG for questions, analysis, or when specific data is mentioned
        rag_indicators = ['what', 'how', 'why', 'analyze', 'find', 'search', 'tell me about']
        return any(indicator in task.lower() for indicator in rag_indicators)
    
    def _build_rowboat_rag_prompt(self, task: str, rag_results: Dict[str, Any]) -> str:
        """Build ROWBOAT-style augmented prompt"""
        prompt_parts = [f"Task: {task}"]
        
        if rag_results.get("answer"):
            prompt_parts.append(f"\nRelevant information: {rag_results['answer']}")
        
        if rag_results.get("results"):
            prompt_parts.append("\nAdditional context:")
            for i, result in enumerate(rag_results["results"][:self.rag_context_limit]):
                source = result["metadata"].get("source", "Unknown")
                prompt_parts.append(f"\n{i+1}. From {source}:")
                prompt_parts.append(f"   {result['content'][:200]}...")
                if result.get("relevance_explanation"):
                    prompt_parts.append(f"   Relevance: {result['relevance_explanation']}")
        
        prompt_parts.append("\nPlease complete the task using the provided context.")
        
        return "\n".join(prompt_parts)
    
    async def _index_execution_result(
        self,
        task: str,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Index successful execution for future reference"""
        try:
            # Create a document from the execution
            execution_doc = f"Task: {task}\nResult: {result.get('output', '')}"
            
            vector_store = await self.rag_system._get_or_create_vector_store(
                context.get("workflow_id", "default")
            )
            
            await vector_store.add_texts(
                texts=[execution_doc],
                metadatas=[{
                    "type": "execution_history",
                    "task": task,
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.name
                }]
            )
        except Exception as e:
            logger.error(f"Failed to index execution result: {e}")