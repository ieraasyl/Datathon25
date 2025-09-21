import os
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Optional, Any, Literal
from google import genai
from dotenv import load_dotenv

# ----------------- Load .env -----------------
load_dotenv()

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Config -----------------
class Config:
    API_KEY = os.getenv("GEMINI_API_KEY")
    DATA_FOLDER = Path('TXT')
    EMBEDDING_MODEL = 'text-embedding-004'
    GENERATION_MODEL = 'gemini-2.0-flash-lite'
    SIMILARITY_THRESHOLD = 0.15
    BATCH_SIZE = 50
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    MAX_DOC_SIZE = 1024 * 1024
    MAX_QUERY_LENGTH = 2000

    @classmethod
    def validate(cls):
        if not cls.API_KEY:
            raise ValueError("API key not found in .env (GEMINI_API_KEY or GOOGLE_API_KEY)")
        return True

Config.validate()

# ----------------- Initialize Client -----------------
try:
    client = genai.Client(api_key=Config.API_KEY)
    logger.info("GenAI client initialized successfully")
except Exception as e:
    logger.error("Failed to initialize GenAI client", exc_info=True)
    raise RuntimeError("GenAI client initialization failed") from e

# ----------------- Custom Exceptions -----------------
class DocumentLoadError(Exception):
    pass

class EmbeddingError(Exception):
    pass

class GenerationError(Exception):
    pass

# ----------------- Operator contacts -----------------
OPERATOR_CONTACTS = {
    "phone": "+7 (701) 123-45-67",
    "email": "support@company.kz",
    "telegram": "@company_support",
    "working_hours": "Пн-Пт: 9:00-18:00 (Астана)",
    "website": "https://company.kz/support"
}

# ----------------- Models -----------------
class RAGRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=Config.MAX_QUERY_LENGTH)
    top_k: int = Field(1, ge=1, le=5, description="Количество документов (по умолчанию 1)")
    style: Literal["detailed", "concise", "analytical", "friendly", "formal"] = "detailed"
    temperature: float = Field(0.1, ge=0.0, le=1.0)

class RAGResponse(BaseModel):
    response: str
    sources_count: int
    source_documents: Optional[List[str]] = None
    needs_operator: bool = False
    processing_time_ms: int
    confidence_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class SimilaritySearchRequest(BaseModel):
    message: str = Field(..., description="Текстовый запрос для поиска")
    top_k: int = Field(5, ge=1, le=10, description="Количество документов для поиска")

class BulkQueryItem(BaseModel):
    message: str = Field(..., description="Текст запроса")
    style: Literal["detailed", "concise", "analytical", "friendly", "formal"] = "detailed"
    temperature: float = Field(0.1, ge=0.0, le=1.0)

class BulkQueryRequest(BaseModel):
    queries: List[BulkQueryItem] = Field(..., description="Список запросов для пакетной обработки (макс. 10)")

# ----------------- Document Store -----------------
class DocumentStore:
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: Optional[NDArray[Any]] = None
        self.names: List[str] = []

doc_store = DocumentStore()

# ----------------- Utils -----------------
async def load_and_embed_documents():
    try:
        doc_store.documents = []
        doc_store.embeddings = None
        doc_store.names = []

        if not Config.DATA_FOLDER.exists():
            Config.DATA_FOLDER.mkdir(exist_ok=True)
            logger.warning("TXT folder created, add files there")
            return False

        txt_files = list(Config.DATA_FOLDER.glob("*.txt"))
        if not txt_files:
            logger.warning("No .txt documents found")
            return False

        texts = []
        for filepath in txt_files:
            if filepath.stat().st_size > Config.MAX_DOC_SIZE:
                continue
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                text = (await f.read()).strip()
                if text:
                    doc_store.documents.append(text)
                    doc_store.names.append(filepath.name)
                    texts.append(text)

        if not texts:
            raise DocumentLoadError("No valid documents found")

        all_embeddings = []
        for i in range(0, len(texts), Config.BATCH_SIZE):
            batch = texts[i:i + Config.BATCH_SIZE]
            for attempt in range(Config.MAX_RETRIES):
                try:
                    response = client.models.embed_content(
                        model=Config.EMBEDDING_MODEL,
                        contents=batch
                    )
                    if hasattr(response, 'embeddings') and response.embeddings:
                        batch_emb = [emb.values for emb in response.embeddings]
                        all_embeddings.extend(batch_emb)
                        break
                except Exception as e:
                    if attempt < Config.MAX_RETRIES - 1:
                        await asyncio.sleep(Config.RETRY_DELAY)
                    else:
                        raise EmbeddingError(f"Failed to embed batch: {e}")
            await asyncio.sleep(1)

        doc_store.embeddings = np.array(all_embeddings)
        logger.info(f"Loaded {len(doc_store.documents)} documents with embeddings")
        return True

    except Exception as e:
        logger.error("Error loading documents", exc_info=True)
        raise DocumentLoadError(str(e))

async def retrieve_documents(query: str, top_k: int = 1):
    if doc_store.embeddings is None:
        return [], [], []

    try:
        q_emb_resp = client.models.embed_content(
            model=Config.EMBEDDING_MODEL,
            contents=query
        )
        if not hasattr(q_emb_resp, 'embeddings') or not q_emb_resp.embeddings:
            return [], [], []

        q_emb = np.array(q_emb_resp.embeddings[0].values)
        sims = cosine_similarity(q_emb.reshape(1, -1), doc_store.embeddings)
        idxs = np.argsort(sims[0])[::-1][:top_k]

        docs, scores, names = [], [], []
        for idx in idxs:
            score = sims[0][idx]
            if score > Config.SIMILARITY_THRESHOLD:
                docs.append(doc_store.documents[idx])
                scores.append(float(score))
                names.append(doc_store.names[idx])
        return docs, scores, names
    except Exception as e:
        raise EmbeddingError(f"Error retrieving documents: {e}")

def operator_message(query: str) -> str:
    return (f"Информация не найдена.\n"
            f"Ваш запрос: {query}\n\n"
            f"Свяжитесь с поддержкой:\n"
            f"📞 {OPERATOR_CONTACTS['phone']}\n"
            f"📧 {OPERATOR_CONTACTS['email']}\n"
            f"💬 {OPERATOR_CONTACTS['telegram']}\n"
            f"🌐 {OPERATOR_CONTACTS['website']}")

async def generate_response(query: str, docs: List[str], style: str, temperature: float) -> str:
    if not docs:
        return operator_message(query)

    context = "\n\n".join([f"Документ {i+1}:\n{d}" for i, d in enumerate(docs)])
    prompts = {
        "detailed": f"""Ты — эксперт-консультант. Дай подробный, но компактный ответ (до 300 слов), используя только документы. Форматируй в Markdown.\n\nКОНТЕКСТ:\n{context}\n\nВОПРОС: {query}\n\nОТВЕТ:""",
        "concise": f"""Ответь кратко (до 5 предложений). Форматируй в Markdown.\n\nКОНТЕКСТ:\n{context}\n\nВОПРОС: {query}\n\nОТВЕТ:""",
        "analytical": f"""Сделай структурированный анализ: 1. **Факты** 2. **Анализ** 3. **Заключение**. Форматируй в Markdown.\n\nКОНТЕКСТ:\n{context}\n\nВОПРОС: {query}\n\nОТВЕТ:""",
        "friendly": f"""Дай дружелюбный и позитивный ответ (до 200 слов), добавь пару эмодзи. Форматируй в Markdown.\n\nКОНТЕКСТ:\n{context}\n\nВОПРОС: {query}\n\nОТВЕТ:""",
        "formal": f"""Составь официальный ответ в деловом стиле (до 250 слов). Форматируй в Markdown.\n\nКОНТЕКСТ:\n{context}\n\nВОПРОС: {query}\n\nОФИЦИАЛЬНЫЙ ОТВЕТ:"""
    }

    try:
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.8,
            top_k=40,
            max_output_tokens=1024
        )
        resp = client.models.generate_content(
            model=Config.GENERATION_MODEL,
            contents=prompts.get(style, prompts["detailed"]),
            config=config
        )
        if hasattr(resp, 'text') and resp.text:
            text = resp.text.strip()
            if len(text) > 2000:
                text = text[:2000].rsplit(" ", 1)[0] + " … (сокращено)"
            return text
        else:
            raise GenerationError("Empty response from model")
    except Exception as e:
        logger.error("Error generating response", exc_info=True)
        raise GenerationError(str(e))

# ----------------- FastAPI -----------------
app = FastAPI(
    title="RAG MVP",
    version="1.5",
    description="Lightweight RAG system with dotenv config and robust error handling"
)

# ----------------- Global Error Handler -----------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error at {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": exc.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
    )

# ----------------- API Routes -----------------
router = APIRouter()

@router.post("/api/rag", response_model=RAGResponse)
async def rag_endpoint(request: RAGRequest):
    if not doc_store.documents:
        raise HTTPException(status_code=503, detail="No documents loaded")

    start = datetime.now()
    docs, scores, names = await retrieve_documents(request.message, request.top_k)
    resp_text = await generate_response(request.message, docs, request.style, request.temperature)
    elapsed = int((datetime.now() - start).total_seconds() * 1000)

    return RAGResponse(
        response=resp_text,
        sources_count=len(docs),
        source_documents=names,
        needs_operator="Информация не найдена" in resp_text,
        processing_time_ms=elapsed,
        confidence_score=max(scores) if scores else None
    )

@router.post("/api/similarity-search")
async def similarity_search(request: SimilaritySearchRequest):
    docs, scores, names = await retrieve_documents(request.message, request.top_k)
    return {
        "query": request.message,
        "results": [
            {"rank": i+1, "document": names[i], "score": scores[i], "preview": docs[i][:200]}
            for i in range(len(docs))
        ]
    }

@router.post("/api/bulk-query")
async def bulk_query(request: BulkQueryRequest):
    if not request.queries:
        raise HTTPException(status_code=400, detail="No queries provided")
    if len(request.queries) > 10:
        raise HTTPException(status_code=400, detail="Too many queries (max 10)")

    results = []
    for q in request.queries:
        docs, scores, names = await retrieve_documents(q.message, 1)
        resp = await generate_response(q.message, docs, q.style, q.temperature)
        results.append({
            "query": q.message,
            "response": resp,
            "sources": names,
            "needs_operator": "Информация не найдена" in resp
        })
    return {"results": results, "total": len(request.queries)}

@router.get("/api/health")
async def health():
    return {
        "status": "ok" if doc_store.embeddings is not None else "empty",
        "documents": len(doc_store.documents),
        "timestamp": datetime.now()
    }

app.include_router(router)

# ----------------- Main -----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    asyncio.run(load_and_embed_documents())
    uvicorn.run(app, host=args.host, port=args.port)