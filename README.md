# VoidMind Proxy

A small HTTP service that holds VoidMind's Groq API keys and rotates them
on rate limits. Protects keys from extraction in the shipped Android app.

## Endpoints

- `GET /health` — returns key availability status
- `POST /api/llm` — forwards prompts to Groq with key rotation

## Environment Variables

- `GROQ_KEY_1` — first Groq API key (required)
- `GROQ_KEY_2` — second key (optional, for rotation)
- `GROQ_KEY_3` — third key (optional)
- `HOST` — bind host (defaults to 127.0.0.1)
- `PORT` — bind port (defaults to 8001)

## Local Development

    GROQ_KEY_1=gsk_... python3 proxy.py

## Deploy to Render

This repo includes `render.yaml` for one-click deploy.
