from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.route.health import router as health_router
from app.api.route.query import router as query_router


app = FastAPI(title="RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(query_router)
