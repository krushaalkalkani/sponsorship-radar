"""Sponsorship Radar - FastAPI service.

Data endpoints work with no API key. The /api/ask agent endpoint needs
ANTHROPIC_API_KEY and degrades with a clear message when it is absent.
"""
import os, traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from . import db

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="H-1B Sponsorship Radar", version="1.0")


@app.get("/api/stats")
def api_stats():
    s = db.stats()
    s["agent_enabled"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    s["semantic_enabled"] = db.store().ok
    return s


@app.get("/api/company")
def api_company(name: str):
    return db.employer_profile(name)


@app.get("/api/suggest")
def api_suggest(q: str, limit: int = 8):
    return db.find_employers(q, limit=limit)


@app.get("/api/shortlist")
def api_shortlist(role: Optional[str] = None, state: Optional[str] = None,
                  city: Optional[str] = None, min_wage: Optional[int] = None,
                  cap_exempt: Optional[bool] = None, exclude_staffing: bool = False,
                  sort: str = "score", limit: int = 25):
    return db.search_employers(role=role, state=state, city=city, min_wage=min_wage,
                               cap_exempt=cap_exempt, exclude_dependent=exclude_staffing,
                               sort=sort, limit=min(limit, 100))


@app.get("/api/semantic")
def api_semantic(q: str, k: int = 10):
    if not db.store().ok:
        return JSONResponse({"error": "vector index not built; run ingest/embed.py"}, 503)
    return db.semantic_search(q, k=min(k, 25))


class Ask(BaseModel):
    question: str
    history: Optional[List[dict]] = None


@app.post("/api/ask")
def api_ask(body: Ask):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return JSONResponse({"error": "Agent disabled: set ANTHROPIC_API_KEY. "
                                      "Company lookup and shortlist still work."}, 503)
    try:
        from .agent import ask
        return ask(body.question)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": "%s: %s" % (type(e).__name__, e)}, 500)


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/health")
def health():
    return {"ok": True}
