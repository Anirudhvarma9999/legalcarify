# legal clarify
ClearClause — Streamlit demo app (modified fallback)

This file is a *fixed*, single-file prototype that preserves the original Streamlit UX **if Streamlit is installed**, but **gracefully falls back to a CLI** when Streamlit is not available (common in sandbox/test environments).

Why this change? You ran the file in an environment where `streamlit` was not installed and got:

    ModuleNotFoundError: No module named 'streamlit'

We cannot change the runtime environment from inside the script, so the safest fix is to avoid importing Streamlit at import time and provide a robust fallback mode. The updated file:
- attempts to import optional libs (streamlit, opencv, pytesseract, googletrans, openai), but never crashes if they're missing
- exposes two run modes: `run_streamlit_app()` (only used if streamlit is present) and `run_cli_app()` (fallback)
- has a simple heuristic/fake LLM mode (USE_FAKE_LLM) which is **enabled by default if OpenAI is not configured** so the demo still works offline
- provides a small test harness you can run locally with `--test` to validate behavior

How to run:
- Streamlit mode (if you have streamlit):
    streamlit run clearclause_streamlit_app.py

- CLI mode (works everywhere):
    python clearclause_streamlit_app.py --text "Tenant shall pay a late fee of 15%..."

- Run tests:
    python clearclause_streamlit_app.py --test

Notes:
- OCR only runs if PIL + pytesseract are available. If not, the CLI will ask you to pass text.
- For a real hackathon demo on Google Cloud, replace the fake-LLM with Vertex/OpenAI keys and set `USE_FAKE_LLM=False`.

"""

from typing import Dict, Any, List
import os
import json
import textwrap
import argparse
import sys
import logging

# optional imports — try and fail gracefully
try:
    import streamlit as st
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import numpy as np
    NP_AVAILABLE = True
except Exception:
    NP_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

try:
    from googletrans import Translator
    GT_AVAILABLE = True
except Exception:
    GT_AVAILABLE = False

try:
    import openai
    OPENAI_PY_AVAILABLE = True
except Exception:
    OPENAI_PY_AVAILABLE = False

# load dotenv if present (not required)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
USE_FAKE_LLM = os.environ.get("USE_FAKE_LLM", "auto").lower() in ("1", "true", "yes")
# If openai python library or API key missing, default to fake LLM
if not OPENAI_PY_AVAILABLE or not OPENAI_API_KEY:
    USE_FAKE_LLM = True

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")

LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
}

# Simple logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clearclause")

# Utilities

def safe_parse_json(s: str):
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # try to greedily extract JSON-like substring
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass
    return None


# OCR helpers (only if PIL + pytesseract available)

def preprocess_for_ocr_pil(pil_img):
    """Return a Pillow image that's been converted to high-contrast grayscale for OCR."""
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is not available for image processing")
    # Basic conversion — keep it simple for a hackathon demo
    img = pil_img.convert("L")  # to grayscale
    # small resize if image is tiny
    w, h = img.size
    if max(w, h) < 1000:
        img = img.resize((int(w * 1.5), int(h * 1.5)))
    return img


def ocr_from_image_path(path: str, lang_code: str = "eng") -> str:
    """Try to OCR an image file and return extracted text. Falls back cleanly if dependencies are missing."""
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for OCR but is not installed.")
    img = Image.open(path)
    pil_for_ocr = preprocess_for_ocr_pil(img)

    # If pytesseract is available, use it; otherwise return a helpful error
    if PYTESSERACT_AVAILABLE:
        try:
            text = pytesseract.image_to_string(pil_for_ocr, lang=lang_code)
            return text
        except Exception as e:
            raise RuntimeError(f"pytesseract OCR failed: {e}")
    else:
        raise RuntimeError("pytesseract not available: install pytesseract + tesseract-ocr for OCR support.")


# LLM helper (fake or live)

def call_llm_simplify(english_text: str) -> Dict[str, Any]:
    """Return dict with keys: summary, risks(list), next_steps(list), risk_score (Low/Medium/High).

    Uses:
    - Live OpenAI ChatCompletion if available and configured (not recommended in a locked sandbox)
    - Otherwise a simple keyword heuristic fake-LLM so the demo is deterministic and offline-friendly
    """
    if not english_text or not english_text.strip():
        return {
            "summary": "",
            "risks": [],
            "next_steps": [],
            "risk_score": "Low",
        }

    if USE_FAKE_LLM:
        lower = english_text.lower()
        risks: List[str] = []
        if any(k in lower for k in ["penalty", "fine", "late", "interest"]):
            risks.append("Financial penalties for late payments")
        if any(k in lower for k in ["automatic renewal", "renew automatically", "auto-renew"]):
            risks.append("Automatic renewal clause")
        if any(k in lower for k in ["arbitration", "waive", "waiver"]):
            risks.append("Arbitration / dispute resolution clause")
        if any(k in lower for k in ["liability", "indemnif", "hold harmless", "hold harmless"]):
            risks.append("Broad liability / indemnity language")
        if any(k in lower for k in ["confidential", "data", "privacy", "share"]):
            risks.append("Data / privacy obligations or sharing")

        if len(risks) >= 2:
            risk_score = "High"
        elif len(risks) == 1:
            risk_score = "Medium"
        else:
            risk_score = "Low"

        summary = textwrap.shorten(english_text.replace("\n", " "), width=300, placeholder="...")
        next_steps = [
            "Ask the counterparty to clarify or cap the penalty/obligation.",
            "If the clause affects your legal or financial position, get a professional review.",
        ]
        return {
            "summary": summary,
            "risks": risks or ["No obvious red flags detected in this short excerpt — check the full contract."],
            "next_steps": next_steps,
            "risk_score": risk_score,
        }

    # Live OpenAI path (only used if USE_FAKE_LLM is False and OpenAI is configured)
    if OPENAI_PY_AVAILABLE and OPENAI_API_KEY:
        try:
            openai.api_key = OPENAI_API_KEY
            system_msg = (
                "You are a helpful legal explainer. Given a short contract clause in English, return a JSON object with keys: 'summary', 'risks', 'next_steps', 'risk_score'. Keep responses concise."
            )
            user_prompt = f"Clause: {english_text}\n\nReturn only JSON."
            resp = openai.ChatCompletion.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            text = resp['choices'][0]['message']['content']
            parsed = safe_parse_json(text)
            if parsed:
                return parsed
            else:
                # fallback: return raw text in summary
                return {
                    "summary": text,
                    "risks": ["(Model returned non-JSON output)"] ,
                    "next_steps": ["Ask a lawyer for review."],
                    "risk_score": "Medium",
                }
        except Exception as e:
            logger.exception("OpenAI call failed")
            return {
                "summary": f"LLM error: {e}",
                "risks": ["LLM call failed - using offline heuristics"],
                "next_steps": ["Set OPENAI_API_KEY or run with USE_FAKE_LLM=True"],
                "risk_score": "Medium",
            }
    else:
        # Shouldn't reach here because USE_FAKE_LLM would be True above, but safe fallback
        return call_llm_simplify(english_text)


# Presentation helpers

def print_results_cli(llm_out: Dict[str, Any], original_text: str = None, local_translation: str = None):
    print("\n=== Plain English Summary ===\n")
    print(llm_out.get('summary', ''))
    if local_translation:
        print("\n=== Local Language Simplified ===\n")
        print(local_translation)
    print("\n=== Risks / Red Flags ===\n")
    risks = llm_out.get('risks', []) or []
    for r in risks:
        print(f"- {r}")
    print("\n=== Suggested Next Steps ===\n")
    for s in llm_out.get('next_steps', []):
        print(f"- {s}")
    print("\n=== Risk Score ===\n")
    print(llm_out.get('risk_score', 'Medium'))
    if original_text:
        print("\n=== Original / Extracted Text (truncated) ===\n")
        print((original_text or '')[:2000])
    print("\nNote: This is a prototype demo. Not legal advice.\n")


# Streamlit app (only builds UI if streamlit is present)

def run_streamlit_app():
    if not ST_AVAILABLE:
        raise RuntimeError("Streamlit is not available in this environment.")

    st.set_page_config(page_title="ClearClause — Legal Simplifier", layout="wide")
    st.title("ClearClause — Demystify legal text (prototype)")
    st.markdown("Upload a screenshot or paste a clause, choose language, and get a plain-English summary, red flags and suggested next steps.")

    with st.sidebar:
        st.header("Demo settings")
        lang_choice = st.selectbox("Document language", list(LANGUAGE_MAP.keys()), index=0, format_func=lambda k: LANGUAGE_MAP[k])
        input_mode = st.radio("Input mode", ["Paste text", "Upload image"], index=0)
        show_original = st.checkbox("Show extracted original text", value=True)
        use_google_translate = st.checkbox("Auto-translate (googletrans)", value=False)
        st.caption("Note: Streamlit mode requires additional packages (pytesseract, googletrans) for OCR/translation.")

    translator = Translator() if GT_AVAILABLE else None

    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Input")
        clause_text = ""
        if input_mode == "Paste text":
            clause_text = st.text_area("Paste the clause here", height=240)
        else:
            uploaded = st.file_uploader("Upload screenshot (png/jpg)", type=["png", "jpg", "jpeg"])
            if uploaded:
                image = Image.open(uploaded)
                st.image(image, caption="Uploaded image", use_column_width=True)
                with st.spinner("Running OCR..."):
                    try:
                        # tesseract lang mapping: simple approach
                        tesseract_lang = 'eng'
                        clause_text = pytesseract.image_to_string(preprocess_for_ocr_pil(image), lang=tesseract_lang) if PYTESSERACT_AVAILABLE else ''
                        st.success("OCR complete")
                    except Exception as e:
                        st.error(f"OCR failed: {e}")

        if st.button("Simplify & Analyze"):
            if not clause_text or clause_text.strip() == "":
                st.warning("Provide text or upload image with visible clause")
            else:
                with st.spinner("Translating (if needed) and calling LLM..."):
                    original_text = clause_text
                    if lang_choice != 'en' and GT_AVAILABLE:
                        try:
                            trans = translator.translate(clause_text, src=lang_choice, dest='en')
                            english_text = trans.text
                        except Exception:
                            english_text = clause_text
                    else:
                        english_text = clause_text

                    llm_out = call_llm_simplify(english_text)
                    simplified_en = llm_out.get('summary', '')
                    if lang_choice != 'en' and GT_AVAILABLE:
                        try:
                            back = translator.translate(simplified_en, src='en', dest=lang_choice)
                            simplified_local = back.text
                        except Exception:
                            simplified_local = simplified_en
                    else:
                        simplified_local = simplified_en

                    with col2:
                        st.header("Results")
                        st.subheader("Plain English Summary")
                        st.write(simplified_en)
                        if lang_choice != 'en':
                            st.subheader(f"Simplified ({LANGUAGE_MAP[lang_choice]})")
                            st.write(simplified_local)

                        st.subheader("Risks / Red Flags")
                        risks = llm_out.get('risks', [])
                        if isinstance(risks, list):
                            for r in risks:
                                st.markdown(f"- {r}")
                        else:
                            st.write(risks)

                        st.subheader("Suggested Next Steps")
                        for s in llm_out.get('next_steps', []):
                            st.markdown(f"- {s}")

                        st.subheader("Risk Score")
                        rscore = llm_out.get('risk_score', 'Medium')
                        if isinstance(rscore, str):
                            s = rscore.lower()
                            color = 'green' if s == 'low' else ('orange' if s == 'medium' else 'red')
                            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{rscore}</span>", unsafe_allow_html=True)
                        else:
                            st.write(rscore)

                        if show_original:
                            st.subheader("Original / Extracted Text")
                            st.code(original_text[:5000])

                        st.caption("Note: This is a demo prototype. Not legal advice. For binding guidance consult a lawyer.")


# CLI fallback (safe for sandboxed environments)

def run_cli_app(args: argparse.Namespace):
    # language translator (optional)
    translator = Translator() if GT_AVAILABLE else None

    clause_text = None
    if args.text:
        clause_text = args.text
    elif args.image:
        if not PIL_AVAILABLE:
            logger.error("Pillow is required to read image files for OCR. Provide text with --text instead.")
            sys.exit(1)
        if not PYTESSERACT_AVAILABLE:
            logger.error("pytesseract is not available. Please install pytesseract + tesseract-ocr for OCR or pass --text.")
            sys.exit(1)
        try:
            clause_text = ocr_from_path_with_fallback(args.image, args.lang)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            sys.exit(1)
    else:
        logger.info("No input provided. Use --text '...' or --image PATH. Running built-in tests by default.")
        run_tests()
        return

    if args.lang != 'en' and translator is not None:
        try:
            trans = translator.translate(clause_text, src=args.lang, dest='en')
            english_text = trans.text
        except Exception:
            english_text = clause_text
    else:
        english_text = clause_text

    llm_out = call_llm_simplify(english_text)

    simplified_en = llm_out.get('summary', '')
    simplified_local = None
    if args.lang != 'en' and translator is not None:
        try:
            back = translator.translate(simplified_en, src='en', dest=args.lang)
            simplified_local = back.text
        except Exception:
            simplified_local = simplified_en

    print_results_cli(llm_out, original_text=clause_text, local_translation=simplified_local)


# small wrapper for OCR path usage (keeps original helper name style)

def ocr_from_path_with_fallback(path: str, lang_code: str = 'eng') -> str:
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for OCR but not available")
    if PYTESSERACT_AVAILABLE:
        return ocr_from_image_path(path, lang_code=lang_code)
    else:
        raise RuntimeError("pytesseract not available")


# Tests

def run_tests():
    print("Running basic self-tests...")
    samples = [
        "Tenant shall pay a late fee of 15% of monthly rent for any rent not paid within five (5) days of the due date.",
        "This Agreement shall automatically renew for successive one-year terms unless either party provides written notice of termination at least 60 days prior to the end of the then-current term.",
        "Company shall not be liable for any indirect, incidental, or consequential damages arising from the use of the service.",
    ]
    for s in samples:
        out = call_llm_simplify(s)
        assert isinstance(out, dict), "LLM output should be a dict"
        for k in ["summary", "risks", "next_steps", "risk_score"]:
            assert k in out, f"Missing key {k} in output"
        assert isinstance(out['risks'], list), "risks should be a list"
        assert out['risk_score'] in ("Low", "Medium", "High"), f"Unexpected risk score: {out['risk_score']}"
    print("All tests passed — call_llm_simplify basic heuristics OK.")


# Argument parsing and entrypoint

def build_argparser():
    p = argparse.ArgumentParser(description="ClearClause CLI fallback (demo).")
    p.add_argument("--text", type=str, help="Paste clause text directly (wrap in quotes)")
    p.add_argument("--image", type=str, help="Path to screenshot/image to OCR (requires Pillow + pytesseract)")
    p.add_argument("--lang", type=str, default='en', help="Document language (en, hi, ta, te, bn)")
    p.add_argument("--test", action='store_true', help="Run self-tests and exit")
    p.add_argument("--streamlit", action='store_true', help="Force attempt to run streamlit UI (if available)")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    if args.streamlit and ST_AVAILABLE:
        # run streamlit UI if user asked for it and it's available
        run_streamlit_app()
    elif ST_AVAILABLE and not args.streamlit:
        # prefer CLI unless the user explicitly asked for Streamlit — avoids streamlit import-time UI boot here
        print("Streamlit is available in the environment. To open the browser UI run:\n\n  streamlit run clearclause_streamlit_app.py\n\nOr run this script with --streamlit to start the app inside this process (not recommended from terminals that are not Streamlit-ready).")
        # Fall back to CLI mode
        run_cli_app(args)
    else:
        # Streamlit not available — run CLI fallback
        run_cli_app(args)

# End of file

