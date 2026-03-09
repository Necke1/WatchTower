# -*- coding: utf-8 -*-
"""
WatchTower API  v1.3
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
import os
import logging
import threading
import contextlib

from pathlib import Path
from typing import Dict, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from constants import DEFAULT_DICTIONARY_FILE, SAMPLE_TEXTS
from text_analyzer import TextAnalyzer, CLASSLA_AVAILABLE
from report_generator import ReportGenerator
from spell_checker import PHUNSPELL_AVAILABLE
from watchtower import analyze_file, build_context_items


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# CORS origins: comma-separated list in WT_ALLOWED_ORIGINS, or "*" for dev.
# Example: WT_ALLOWED_ORIGINS="https://app.example.com,http://localhost:3000"
_raw_origins = os.getenv("WT_ALLOWED_ORIGINS", "*")
_ALLOWED_ORIGINS: list = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins != "*"
    else ["*"]
)

app = FastAPI(title="WatchTower API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "stranica"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/stranica", StaticFiles(directory=str(STATIC_DIR)), name="stranica")

# ---------------------------------------------------------------------------
# Thread-safe analyzer cache  (fix 4)
# ---------------------------------------------------------------------------
# Key: (use_spellcheck, auto_correct, dictionary_file)
# Each unique combination gets its own TextAnalyzer instance.
# A lock prevents two concurrent requests from both deciding to create the
# same instance simultaneously.

_AnalyzerKey = Tuple[bool, bool, str]
_analyzer_cache: Dict[_AnalyzerKey, TextAnalyzer] = {}
_analyzer_lock  = threading.Lock()


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

class FeedbackRequest(BaseModel):
    text:              str   = Field(..., min_length=1)
    term:              str   = Field(..., min_length=1)
    feedback:          str   = Field(...)
    custom_multiplier: float = Field(None, ge=0.0, le=10.0,
                                     description="Optional: override the default multiplier "
                                                 "(0.0=ignore, 1.0=neutral, 2.0=amplify). "
                                                 "If set, the feedback type is used only for the label.") # type: ignore

class CorrectionAddRequest(BaseModel):
    original:  str = Field(..., min_length=1, description="The misspelled word as it appeared.")
    corrected: str = Field(..., min_length=1, description="The correct spelling.")
    rating:    int = Field(3, ge=1, le=4, description="Quality rating 1–4 (default 3).")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def capture_output():
    """
    Capture both print() output and logging records for the duration of a
    request and return them as a single string.

    print()   → captured via contextlib.redirect_stdout
    logging   → captured via a temporary StreamHandler on the root logger
    """
    buf     = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        with contextlib.redirect_stdout(buf):
            yield buf
    finally:
        root_logger.removeHandler(handler)


def _get_analyzer(use_spellcheck: bool = True,
                  auto_correct:   bool = False,
                  dictionary_file: str = DEFAULT_DICTIONARY_FILE) -> TextAnalyzer:
    """
    Return a cached TextAnalyzer for the given configuration.
    Thread-safe: uses a lock so concurrent requests never race to create the
    same instance, and never see a partially-initialised one.
    """
    key: _AnalyzerKey = (use_spellcheck, auto_correct, dictionary_file)
    if key not in _analyzer_cache:
        with _analyzer_lock:
            # Re-check inside the lock — another thread may have created it
            # while we were waiting.
            if key not in _analyzer_cache:
                _analyzer_cache[key] = TextAnalyzer(
                    dictionary_file     = dictionary_file,
                    use_spellcheck      = use_spellcheck,
                    auto_correct        = auto_correct,
                    interactive_correct = False,
                )
    return _analyzer_cache[key]


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
    with capture_output() as buf:
        a      = _get_analyzer(req.use_spellcheck, req.auto_correct)
        raw    = a.analyze_text(req.text)
        report = (ReportGenerator.generate_json_report(raw)
                  if req.output_json
                  else ReportGenerator.generate_text_report(raw, a))
        ctx_items = build_context_items(raw, req.text, a)
    return {
        **raw,
        "console_output":   buf.getvalue(),
        "formatted_report": report,
        "term_weights":     {t: a.term_to_weight.get(t, 1)
                             for t in raw['analysis']['term_frequencies']},
        "chat_analysis":    None,
        "context_items":    ctx_items,
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

    with capture_output() as buf:
        a = _get_analyzer(sp, ac)
        try:
            result = analyze_file(raw_bytes, filename, a, output_json=oj)
        except ValueError as e:
            status_code = 415 if "Unsupported file type" in str(e) else 422
            raise HTTPException(status_code, str(e))
        raw_text   = raw_bytes.decode('utf-8', errors='replace')
        ctx_items  = build_context_items(result, raw_text, a)

    result["console_output"] = buf.getvalue()
    result["context_items"]  = ctx_items
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

@app.post("/api/demo", tags=["Analysis"])
async def run_demo(req: DemoRequest):
    results = []
    with capture_output() as buf:
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
    with capture_output():
        a = _get_analyzer()
    terms = sorted(
        [{"term": t, "weight": a.term_to_weight.get(t, 1)} for t in a.words_set],
        key=lambda x: x["weight"], reverse=True,
    )
    return {"size": len(a.words_set), "file_path": a.dictionary_file, "terms": terms}


@app.post("/api/dictionary/reload", tags=["Dictionary"])
async def reload_dict():
    with capture_output() as buf:
        a = _get_analyzer()
        a.load_dictionary()
    return {"message": f"Reloaded — {len(a.words_set)} terms.", "console_output": buf.getvalue()}


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

@app.get("/api/corrections", tags=["Corrections"])
async def get_corrections():
    """Return all learned spell-check corrections."""
    corrections = _get_corrections()
    return {"corrections": corrections, "total": len(corrections)}


@app.post("/api/corrections", tags=["Corrections"])
async def add_correction(req: CorrectionAddRequest):
    """
    Save a spell-check correction from the browser interactive mode.

    Equivalent to confirming a correction in the CLI --interactive-correct flow.
    The correction is saved to korekcija.txt and applied automatically in future
    analyses (same as the CLI learned-corrections mechanism).
    """
    a = _get_analyzer()
    if a.spell_checker is None:
        raise HTTPException(503, "Spell checker not available (phunspell not installed).")
    a.spell_checker.add_user_correction(req.original, req.corrected, req.rating)
    return {
        "saved":     True,
        "original":  req.original,
        "corrected": req.corrected,
        "rating":    req.rating,
        "message":   f"Korekcija sačuvana: '{req.original}' → '{req.corrected}' (ocena {req.rating}/4)",
    }


@app.delete("/api/corrections", tags=["Corrections"])
async def delete_corrections():
    """Delete all learned spell-check corrections (cannot be undone)."""
    a = _get_analyzer()
    if a.spell_checker is None:
        raise HTTPException(503, "Spell checker not available.")
    count = a.spell_checker.clear_corrections()
    return {"deleted": count, "message": f"Obrisano {count} naučenih korekcija."}


# ---------------------------------------------------------------------------
# Analyst feedback — pattern learning
# ---------------------------------------------------------------------------

@app.post("/api/feedback", tags=["Learning"])
async def submit_feedback(req: FeedbackRequest):
    """
    Submit analyst feedback on a flagged term to teach the system.

    Works exactly like the spell-checker's learned corrections, but for
    context patterns.  Each call saves a generalised rule that is
    automatically applied to future analyses — no code changes needed.

    The effect is confidence-blended:
      • 1 analyst vote  → pattern applied at 50% strength
      • 3 analyst votes → 75% strength
      • 5 analyst votes → fully trusted, applied at 100%

    This prevents a single mistake from permanently corrupting scores.
    """
    a = _get_analyzer()
    multiplier_map = {
        'false_positive':  0.0,
        'confirmed':       2.0,
        'lower_severity':  0.35,
        'higher_severity': 3.0,
    }
    if req.feedback not in multiplier_map:
        raise HTTPException(422, f"Unknown feedback type '{req.feedback}'. "
                                 f"Choose from: {list(multiplier_map)}")

    # Allow analyst to override the default multiplier with a custom value
    multiplier = req.custom_multiplier if req.custom_multiplier is not None \
                 else multiplier_map[req.feedback]

    try:
        fingerprint = a.learned_patterns.learn(
            req.term.lower(),
            a.extract_lemmas(a.normalize_text(req.text)),
            multiplier,
            label=f"[analyst:{req.feedback}]",
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    if fingerprint is None:
        return {
            "saved":   False,
            "message": (
                "Pattern not saved — the context window around the term "
                "contained only stop-words. Provide a longer message with "
                "more context words."
            ),
        }

    total = len(a.learned_patterns.get_all_patterns())
    return {
        "saved":          True,
        "term":           req.term.lower(),
        "feedback":       req.feedback,
        "fingerprint":    sorted(fingerprint),
        "total_patterns": total,
        "message": (
            f"Naučeno: '{req.term}' u kontekstu {sorted(fingerprint)} → '{req.feedback}'. "
            f"Ukupno naučenih paterna: {total}."
        ),
    }


@app.get("/api/patterns", tags=["Learning"])
async def get_patterns():
    """Return all learned context patterns with confidence scores."""
    a = _get_analyzer()
    patterns = a.learned_patterns.get_all_patterns()
    return {"total": len(patterns), "patterns": patterns}


@app.delete("/api/patterns", tags=["Learning"])
async def clear_patterns():
    """Delete all learned context patterns (cannot be undone)."""
    a = _get_analyzer()
    count = a.learned_patterns.clear()
    return {"deleted": count, "message": f"Obrisano {count} naučenih paterna."}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["System"])
async def health():
    with capture_output():
        a = _get_analyzer()
    return {
        "status":              "ok",
        "classla_available":   CLASSLA_AVAILABLE,
        "phunspell_available": PHUNSPELL_AVAILABLE,
        "dictionary_size":     len(a.words_set),
        "dictionary_file":     a.dictionary_file,
    }