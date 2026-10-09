import os, re
from pathlib import Path
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
KB_FILE = BASE_DIR / "travel_agent_data.txt"
FALLBACK = "I don't have that information in the travel knowledge base."
SCOPE_FALLBACK = "I can only help with travel-related questions based on the travel knowledge base."
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://travel2811.services.ai.azure.com/openai/v1")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-mini")
app = Flask(__name__)

def load_kb():
    return KB_FILE.read_text(encoding="utf-8") if KB_FILE.exists() else ""

def is_travel_related(q):
    terms = ["travel","trip","tour","tourism","tourist","vacation","holiday","destination","itinerary","package","booking","reservation","departure","arrival","hotel","flight","airport","visa","passport","insurance","transfer","breakfast","luggage","ticket","attraction","museum","cruise","safari","price","cost","duration","include","includes","excluded","exclude","cancellation","cancel","deposit","balance","change","fare difference","support","emergency","best time","dubai","bali","paris","singapore","burj khalifa","ubud","tanah lot","eiffel","louvre","seine","sentosa","gardens by the bay"]
    q = q.lower().strip()
    return any(t in q for t in terms)

def retrieve_context(question, kb):
    # Simple lexical retrieval keeps the app usable without downloading an embedding model.
    q_terms = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", kb) if p.strip()]
    scored = []
    for p in paragraphs:
        p_terms = set(re.findall(r"[a-z0-9]+", p.lower()))
        overlap = len(q_terms & p_terms)
        if overlap:
            scored.append((overlap / max(1, len(q_terms)), p))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [p for score, p in scored[:4] if score > 0], (scored[0][0] if scored else 0)

def answer_with_azure(question, context):
    key = os.getenv("AZURE_OPENAI_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None, "Azure AI key not configured"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=key)
        prompt = f"""You are a strict travel assistant. Answer only with facts explicitly stated in the knowledge-base context. Do not infer or use general knowledge. If the requested fact is absent, output exactly: "{FALLBACK}". If outside travel scope, output exactly: "{SCOPE_FALLBACK}". Keep concise.\n\nKNOWLEDGE-BASE CONTEXT:\n{chr(10).join(context)}\n\nCUSTOMER QUESTION:\n{question}\n\nFINAL ANSWER:"""
        resp = client.responses.create(model=AZURE_OPENAI_MODEL, input=prompt)
        answer = (resp.output_text or "").strip()
        # Conservative grounding check.
        if not answer or (answer != FALLBACK and answer != SCOPE_FALLBACK and not any(w in " ".join(context).lower() for w in re.findall(r"[a-z0-9]{4,}", answer.lower()))):
            return FALLBACK, None
        return answer, None
    except Exception as e:
        return None, str(e)

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"status":"ok", "knowledge_base_loaded": bool(load_kb()), "azure_configured": bool(os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))})

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"answer":"Please enter a question.", "status":"empty"})
    if not is_travel_related(question):
        return jsonify({"answer":SCOPE_FALLBACK, "status":"out_of_scope"})
    kb = load_kb()
    context, score = retrieve_context(question, kb)
    if not context:
        return jsonify({"answer":FALLBACK, "status":"no_context", "sources":[]})
    answer, err = answer_with_azure(question, context)
    if answer is None:
        # Safe deterministic fallback: return a relevant KB excerpt rather than invent facts.
        answer = "\n\n".join(context[:2])
        status = "knowledge_base_only"
        if err:
            app.logger.info("Azure response unavailable: %s", err)
    else:
        status = "answered"
    return jsonify({"answer":answer, "status":status, "sources":context[:3]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
