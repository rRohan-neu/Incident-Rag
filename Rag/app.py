"""Local server for the incident KB assistant.

    uvicorn app:app --port 8000

http://localhost:8000              agent screen: KB suggestions update as you type a ticket
http://localhost:8000/v1           OpenAI-compatible endpoint; add it to Open WebUI as a connection
"""
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import rag_pipeline as rag

MODEL_ID = "incident-kb"
app = FastAPI(title="Incident KB assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- agent screen
@app.get("/", include_in_schema=False)
def agent_screen():
    return FileResponse(Path(__file__).parent / "static" / "agent.html")


class TicketText(BaseModel):
    short_description: str = ""
    description: str = ""


def ticket_query(t: TicketText) -> str:
    # The headline is the strongest signal, so it is repeated; details refine it.
    sd, desc = t.short_description.strip(), t.description.strip()
    return f"{sd}. {sd}. {desc}" if desc else sd


@app.post("/suggest")
def suggest(t: TicketText):
    """Top KB articles for what the agent has typed so far. Fast: retrieval only, no LLM."""
    matches = rag.search_articles(ticket_query(t), top=3)
    confident = [m for m in matches if m["confident"]]
    return {"matches": confident, "best_unconfident": None if confident else (matches[0] if matches else None),
            "threshold": rag.MIN_SIMILARITY}


@app.get("/kb/{kb_id}")
def kb_article(kb_id: str):
    a = rag.get_article(kb_id)
    if not a:
        raise HTTPException(404, f"{kb_id} not found")
    return a


@app.post("/answer")
def answer(t: TicketText):
    """AI-written steps grounded in the matching articles, for the 'Summarize the fix' button."""
    text, matches = rag.answer(ticket_query(t))
    return {"answer": text, "sources": [m["kb"] for m in matches]}


# ---------------------------------------------------------------- Open WebUI (OpenAI-compatible)
@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}]}


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[dict]
    stream: bool = False


def _text(content) -> str:
    if isinstance(content, list):                       # newer clients send a list of parts
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _chunk(cid, text=None, finish=None):
    delta = {"content": text} if text is not None else {}
    return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                                  "model": MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    user_msgs = [_text(m.get("content")) for m in req.messages if m.get("role") == "user"]
    question = user_msgs[-1] if user_msgs else ""

    # Open WebUI sends its own housekeeping prompts (chat titles, tags, follow-ups).
    # Answer those with the plain model, without searching the KB.
    housekeeping = question.lstrip().startswith("### Task:")
    cid = "chatcmpl-" + uuid.uuid4().hex[:12]

    if req.stream:
        def gen():
            if housekeeping:
                for piece in rag.get_llm().stream(question):
                    yield _chunk(cid, piece.content)
            else:
                for piece in rag.stream_answer(question):
                    yield _chunk(cid, piece)
            yield _chunk(cid, finish="stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    if housekeeping:
        text = rag.get_llm().invoke(question).content
    else:
        text, matches = rag.answer(question)
        if matches:
            text += "\n\nSources: " + ", ".join(f"{m['kb']} {m['title']}" for m in matches)
    return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
