---
title: elijah-digital-twin
app_file: main.py
sdk: gradio
sdk_version: 6.11.0
---

# Elijah Digital Twin

A RAG-augmented “digital twin” chat app: a [Gradio](https://gradio.app/) UI where a large language model answers as the configured profile (see [`config.py`](config.py)). Retrieval uses [Chroma](https://www.trychroma.com/) over documents you provide; the model is called through [OpenRouter](https://openrouter.ai/) using the OpenAI-compatible API. Optional tools can record contact details or unanswered questions and notify you via [Pushover](https://pushover.net/) when credentials are set.

## Features

- **Gradio 6** chat interface with suggested prompts ([`app/gradio.py`](app/gradio.py))
- **OpenRouter** for chat and lightweight “should we retrieve?” checks ([`src/agent.py`](src/agent.py), [`src/pipeline.py`](src/pipeline.py))
- **Chroma** persistent vector store under `data/vector_store`
- **Ingestion** from `data/raw` as `.txt` or `.pdf` ([`src/injest.py`](src/injest.py))
- **Optional Pushover** for notifications when tools run ([`src/tools.py`](src/tools.py), [`ultils/Pushover.py`](ultils/Pushover.py))

## Try it

When the Space is public, open:

`https://huggingface.co/spaces/YOUR_USERNAME/elijah-digital-twin`

Replace `YOUR_USERNAME` with your Hugging Face username or organization (the Space slug matches `title` in the README frontmatter: `elijah-digital-twin`).

## Requirements

- **Python 3.12+** for local development (see [`pyproject.toml`](pyproject.toml)). Hugging Face Spaces uses the runtime configured for the Space; this repo’s Space metadata pins **Gradio 6.11.0** in the header above.

## Local setup

1. Clone the repository and create a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (see below). Do not commit secrets; use a local `.env` file or your shell.

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | Yes | API key for OpenRouter ([`config.py`](config.py)) |
| `PUSHOVER_USER` | No | Pushover user key for tool notifications |
| `PUSHOVER_TOKEN` | No | Pushover application token |

## RAG and data

The repository includes a **prepopulated Chroma database** under `data/vector_store`, indexed from the maintainer’s own profile material so the app and [Hugging Face Space](https://huggingface.co/docs/hub/spaces) work without running ingestion first.

**To use your own profile or documents:** add `.txt` or `.pdf` files under `data/raw`, then from the **project root** run:

```bash
python scripts/injest_data.py
```

That rebuilds the vector index (see [`scripts/injest_data.py`](scripts/injest_data.py)).

Note: `.gitignore` lists `data/raw`, so new files there are not tracked by default. Add them explicitly if you want them in git, or keep them local only. For a Space deployment, ensure any raw files you rely on are present in the deployed tree if you are not using only the bundled `data/vector_store`.

## Run locally

From the project root:

```bash
python main.py
```

This matches `app_file: main.py` in the Space README header.

## Deployment

- **Hugging Face Spaces:** set **Secrets** (or variables) in the Space settings for `OPENROUTER_API_KEY` and, if used, Pushover keys. Same names as in the table above.
- **CI:** [`.github/workflows/update_space.yml`](.github/workflows/update_space.yml) runs on pushes to `main`, logs in with a GitHub Actions secret `hf_token`, and runs `gradio deploy`. Configure that secret with a Hugging Face token that can deploy to your Space.
