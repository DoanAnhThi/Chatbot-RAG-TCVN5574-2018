import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.services.indexing.indexer import build_or_load_vectorstore
from src.services.pipelines.retrieval.retriever import get_retriever
from src.services.pipelines.generation.generation import build_graph
from src.api.schemas.query import ChatRequest, ChatResponse

router = APIRouter()

_vectorstore = None
_retriever = None
_graph = None


def ensure_graph():
    global _vectorstore, _retriever, _graph
    if _graph is not None:
        return
    _vectorstore = build_or_load_vectorstore()
    _retriever = get_retriever(_vectorstore, settings.retrieval_top_k)
    _graph = build_graph(_retriever)


@router.on_event("startup")
def _startup():
    # Do not crash if indexing fails (e.g., missing API key or documents). We'll lazily init on first request.
    try:
        ensure_graph()
    except Exception:
        pass


@router.get("/", response_class=HTMLResponse)
async def status_page():
    return HTMLResponse(
        """
        <html>
            <head>
                <title>RAG Chatbot</title>
            </head>
            <body style="font-family:Arial, sans-serif; margin:2rem;">
                <h1>RAG Chatbot backend is running</h1>
                <p>Use the <code>/chat</code> endpoint to interact with the API.</p>
                <p>Health check: <a href="/health">/health</a></p>
            </body>
        </html>
        """,
        status_code=200,
    )


# Serve minimal frontend
if os.path.isdir("/app/frontend"):
    router.mount("/static", StaticFiles(directory="/app/frontend"), name="frontend")

    @router.get("/")
    async def read_root():
        with open("/app/frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Chat endpoint for RAG chatbot"""
    try:
        ensure_graph()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Graph initialization failed: {e}")

    state = {"question": req.question}
    result = _graph.invoke(state)

    return ChatResponse(
        answer=result.get("answer", ""),
        confidence=result.get("confidence", 0.0),
        needs_clarification=result.get("needs_clarification", False),
    )
