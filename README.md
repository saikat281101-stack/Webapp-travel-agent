# Waypoint — Guardrailed RAG Travel Agent Web App

A Flask web interface for the supplied travel-agent knowledge base. Includes a responsive chat UI, destination shortcuts, retrieved-context disclosure, travel-scope guardrail, and optional Azure AI Foundry model answers.

## Requirements
- Python 3.10+
- Optional: Azure AI Foundry endpoint/key and a deployed model compatible with the OpenAI Responses API.

## Run locally

### Windows PowerShell
```powershell
cd travel_agent_webapp
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:AZURE_OPENAI_API_KEY="YOUR_KEY"
$env:AZURE_OPENAI_ENDPOINT="https://travel2811.services.ai.azure.com/openai/v1"
$env:AZURE_OPENAI_MODEL="gpt-5-mini"
python app.py
```

### macOS / Linux
```bash
cd travel_agent_webapp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export AZURE_OPENAI_API_KEY="YOUR_KEY"
export AZURE_OPENAI_ENDPOINT="https://travel2811.services.ai.azure.com/openai/v1"
export AZURE_OPENAI_MODEL="gpt-5-mini"
python app.py
```

Open http://127.0.0.1:5000

## Key handling
Do not paste API keys into source files or commit them. Set `AZURE_OPENAI_API_KEY` as an environment variable. If a real key has been exposed in a chat, screenshot, or source file, rotate it in Azure AI Foundry.

## Behavior and guardrails
- Non-travel questions are rejected with the configured scope message.
- Responses use only retrieved snippets from `travel_agent_data.txt` as model context.
- If the Azure key is not set or the Azure request fails, the app returns relevant knowledge-base excerpts rather than inventing an answer.
- The included knowledge base explicitly says prices and policies are fictional lab data. Do not present them as real travel offers.
- `/api/health` reports whether the knowledge base is present and whether an API key environment variable is configured.

## API
- `GET /` — web interface
- `GET /api/health` — basic readiness
- `POST /api/chat` JSON body: `{"question":"What is included in the Dubai package?"}`

## Original project files
`rag_travel_agent_original.py` and `vector_db_lab_original.py` are preserved as supplied reference scripts. The web app uses lightweight lexical retrieval by default so it can start without a ChromaDB vector index/model download.
