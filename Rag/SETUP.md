# Incident KB assistant: setup on Windows

Project folder: `C:\Incident knowledge base articles\Rag`

```
Rag\
  rag_pipeline.py        indexing, search, cited answers (replaces your current file)
  app.py                 local server: agent screen + Open WebUI connection
  static\agent.html      the agent screen
  requirements.txt
  Documents\             put the KB Word files here (KB00001_....docx)
  Chroma_store\          created by --build
  kb_articles.json       created by --build
```

## 1. Ollama models (one time)

Install Ollama from ollama.com if it isn't already, then in a terminal:

```
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

`llama3.2:3b` is small and runs on most laptops. To use another model, change `CHAT_MODEL` at the top of `rag_pipeline.py`.

## 2. Python environment

```
cd "C:\Incident knowledge base articles\Rag"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Add the KB articles and build the index

Copy the 25 `KB000xx_....docx` files into `Documents\`. The index file
(`KB_Index_and_Known_Gaps.docx`) can sit there too; it is skipped automatically.

```
python rag_pipeline.py --build
```

You should see `Loaded 25 KB articles, split into ... chunks`. Re-run `--build` whenever articles change.
Stop the server first, because Windows locks the `Chroma_store` folder while it is in use.

## 4. Test from the terminal

```
python rag_pipeline.py --query "VPN drops right after I approve the Okta push"
python rag_pipeline.py --ask "VPN drops right after I approve the Okta push"
```

`--query` should list KB00023 first. `--ask` gives numbered steps citing [KB00023].

Then test a known gap, an issue with no article on purpose:

```
python rag_pipeline.py --query "laptop connects to wifi but lands on the guest VLAN"
```

The matches should be marked "below threshold". If a gap question passes the threshold, or a real
question falls below it, adjust `MIN_SIMILARITY` in `rag_pipeline.py` (start at 0.55 and move
in steps of 0.05), then re-test. No rebuild is needed for that change.

## 5. Agent screen (live suggestions as you type)

```
uvicorn app:app --port 8000
```

Open http://localhost:8000. Type a headline such as "Can't open shared drive" and watch the
suggestions appear. Then add detail in Description, for example "it keeps asking for my password
since I changed it". The best match switches from KB00017 to KB00018 and the screen says why.
Select a card to see the article; "Summarize the fix" asks the local LLM for cited steps.

## 6. Open WebUI chat

Keep the server from step 5 running.

- **Open WebUI installed with pip** (`pip install open-webui`, then `open-webui serve`; needs Python 3.11):
  go to Admin Panel > Settings > Connections > OpenAI API, add `http://localhost:8000/v1`, and put any text as the API key.
- **Open WebUI in Docker:** same place, but use `http://host.docker.internal:8000/v1`.

Start a new chat and pick the model **incident-kb**. Answers come from your KB index with sources
listed at the end. Don't upload the KB files to Open WebUI's own Knowledge feature as well; that
would give it a second, separate index.

## Adding the past incidents later

Put incident files in `Documents\` and give their chunks `doc_type: "incident"` in `build_index()`.
Keep KB articles ranked first and label incident matches "past incident, not an approved fix",
matching the governance controls.

## Troubleshooting

- **"No vector store found":** run `python rag_pipeline.py --build` first.
- **Connection errors:** make sure Ollama is running (`ollama list` should answer).
- **Open WebUI doesn't show incident-kb:** check http://localhost:8000/v1/models in a browser, and use the Docker address if Open WebUI runs in Docker.
- **Slow answers:** suggestions don't use the LLM and should be quick. Only `--ask`, "Summarize the fix" and chat call the model.
