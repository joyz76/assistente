from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from rag.ask import RAG, format_final_answer


app = FastAPI(title="RAG Aziendale API")
rag = RAG()


class ChiediRequest(BaseModel):
    domanda: str
    modalita: str | None = "auto"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chiedi")
def chiedi(req: ChiediRequest):
    domanda = (req.domanda or "").strip()

    if not domanda:
        return {
            "ok": False,
            "errore": "Domanda vuota",
            "risposta": "",
            "sources": [],
        }

    result = rag.ask(domanda)
    risposta = format_final_answer(result["answer"], result["sources"], max_sources=2)

    return {
        "ok": True,
        "domanda": domanda,
        "risposta": risposta,
        "sources": result["sources"],
    }


@app.post("/ask")
def ask(req: ChiediRequest):
    domanda = (req.domanda or "").strip()

    if not domanda:
        return {"answer": "Domanda vuota."}

    result = rag.ask(domanda)
    risposta = format_final_answer(result["answer"], result["sources"], max_sources=2)

    return {"answer": risposta}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-local",
                "object": "model",
                "owned_by": "rag"
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(req: dict):
    messages = req.get("messages", [])

    domanda = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            domanda = (msg.get("content") or "").strip()
            break

    if not domanda:
        risposta = "Domanda vuota."
    else:
        result = rag.ask(domanda)
        risposta = format_final_answer(result["answer"], result["sources"], max_sources=2)

    return {
        "id": "chatcmpl-rag-local",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": risposta
                },
                "finish_reason": "stop"
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
