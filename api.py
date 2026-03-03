# -*- coding: utf-8 -*-
"""
WatchTower API  
====================
All analysis goes through two endpoints:

  POST /api/analyze       — plain text string
  POST /api/analyze/file  — .txt or .json file upload
                            ↳ if the JSON contains a top-level "messages" array
                              it is automatically treated as a chat export;
                              the response includes flagged_messages + summary
                              sentences in addition to the normal result fields.

Supporting endpoints:
  POST /api/demo
  GET  /api/dictionary
  POST /api/dictionary/reload
  GET  /api/corrections
  GET  /api/health
"""

import io
import contextlib

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from constants import DEFAULT_DICTIONARY_FILE, SAMPLE_TEXTS
from text_analyzer import TextAnalyzer, CLASSLA_AVAILABLE
from report_generator import ReportGenerator
from spell_checker import PHUNSPELL_AVAILABLE
from watchtower import analyze_file


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="WatchTower API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "stranica"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/stranica", StaticFiles(directory=str(STATIC_DIR)), name="stranica")

_analyzer: Optional[TextAnalyzer] = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text:           str  = Field(..., min_length=1)
    use_spellcheck: bool = True
    auto_correct:   bool = False
    output_json:    bool = False

class DemoRequest(BaseModel):
    use_spellcheck: bool = True
    auto_correct:   bool = False



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def capture_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def _get_analyzer(use_spellcheck: bool = True,
                  auto_correct:   bool = False,
                  dictionary_file: str = DEFAULT_DICTIONARY_FILE) -> TextAnalyzer:
    global _analyzer
    if (
        _analyzer is None
        or _analyzer.use_spellcheck  != use_spellcheck
        or _analyzer.auto_correct    != auto_correct
        or _analyzer.dictionary_file != dictionary_file
    ):
        _analyzer = TextAnalyzer(
            dictionary_file     = dictionary_file,
            use_spellcheck      = use_spellcheck,
            auto_correct        = auto_correct,
            interactive_correct = False,
        )
    return _analyzer


def _get_corrections() -> list:
    """
    Return all learned corrections via the spell checker that already owns
    the parsed in-memory state — no need to re-parse the file here.
    """
    a = _get_analyzer()
    if a.spell_checker is None:
        return []
    return a.spell_checker.get_all_corrections()



# ===========================================================================
# Routes
# ===========================================================================

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "pocetna.html"))


# ---------------------------------------------------------------------------
# Plain text analysis
# ---------------------------------------------------------------------------

@app.post("/api/analyze", tags=["Analysis"])
async def analyze(req: AnalyzeRequest):
    """Analyze a plain-text string."""
    if not req.text.strip():
        raise HTTPException(422, "Text must not be empty.")
    with capture_stdout() as buf:
        a      = _get_analyzer(req.use_spellcheck, req.auto_correct)
        raw    = a.analyze_text(req.text)
        report = (ReportGenerator.generate_json_report(raw)
                  if req.output_json
                  else ReportGenerator.generate_text_report(raw, a))
    return {
        **raw,
        "console_output":   buf.getvalue(),
        "formatted_report": report,
        "term_weights":     {t: a.term_to_weight.get(t, 1)
                             for t in raw['analysis']['term_frequencies']},
        "chat_analysis":    None,   # always null for plain-text requests
    }


# ---------------------------------------------------------------------------
# File upload analysis  (.txt or .json — chat exports auto-detected)
# ---------------------------------------------------------------------------

@app.post("/api/analyze/file", tags=["Analysis"])
async def analyze_file_route(
    file:           UploadFile = File(...),
    use_spellcheck: str        = Form("true"),
    auto_correct:   str        = Form("false"),
    output_json:    str        = Form("false"),
):
    """
    Upload a .txt or .json file for analysis.

    .txt  → full content analysed as a single text.
    .json → auto-detected:
              • Chat export (top-level "messages" array):
                  each message is analysed individually; response includes
                  chat_analysis.flagged_messages with user_name, user_id,
                  date, text, risk score, risk level, and verdict.
              • Generic JSON:
                  text extracted by key priority then analysed as a whole.
    """
    filename  = file.filename or ""
    raw_bytes = await file.read()

    if len(raw_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large. Maximum allowed size is 20 MB.")

    sp = use_spellcheck.lower() != "false"
    ac = auto_correct.lower()   == "true"
    oj = output_json.lower()    == "true"

    with capture_stdout() as buf:
        a = _get_analyzer(sp, ac)
        try:
            result = analyze_file(raw_bytes, filename, a, output_json=oj)
        except ValueError as e:
            status_code = 415 if "Unsupported file type" in str(e) else 422
            raise HTTPException(status_code, str(e))

    result["console_output"] = buf.getvalue()
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

@app.post("/api/demo", tags=["Analysis"])
async def run_demo(req: DemoRequest):
    results = []
    with capture_stdout() as buf:
        a = _get_analyzer(req.use_spellcheck, req.auto_correct)
        for sample in SAMPLE_TEXTS:
            raw = a.analyze_text(sample['text'])
            results.append({
                "name": sample['name'],
                "text": sample['text'],
                **raw,
                "term_weights": {t: a.term_to_weight.get(t, 1)
                                 for t in raw['analysis']['term_frequencies']},
            })
    return {"results": results, "console_output": buf.getvalue()}


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

@app.get("/api/dictionary", tags=["Dictionary"])
async def get_dictionary():
    with capture_stdout():
        a = _get_analyzer()
    terms = sorted(
        [{"term": t, "weight": a.term_to_weight.get(t, 1)} for t in a.words_set],
        key=lambda x: x["weight"], reverse=True,
    )
    return {"size": len(a.words_set), "file_path": a.dictionary_file, "terms": terms}


@app.post("/api/dictionary/reload", tags=["Dictionary"])
async def reload_dict():
    with capture_stdout() as buf:
        a = _get_analyzer()
        a.load_dictionary()
    return {"message": f"Reloaded — {len(a.words_set)} terms.", "console_output": buf.getvalue()}


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

@app.get("/api/corrections", tags=["Corrections"])
async def get_corrections():
    corrections = _get_corrections()
    return {"corrections": corrections, "total": len(corrections)}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["System"])
async def health():
    with capture_stdout():
        a = _get_analyzer()
    return {
        "status":              "ok",
        "classla_available":   CLASSLA_AVAILABLE,
        "phunspell_available": PHUNSPELL_AVAILABLE,
        "dictionary_size":     len(a.words_set),
        "dictionary_file":     a.dictionary_file,
    }