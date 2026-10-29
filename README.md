![hide your llama](./assets/banner.png)

# Poncho - A sidecar for ML in the Pocket Network

The poncho is a performant Rust app that is designed to cover your vLLM endpoint and keep it warm and cozy doing its stuff while it connects to the Pocket Network.


### Configuration

You will only need a config file, please refer to the [example file](./apps/rust/poncho/config/config.sample.yaml).

### Deployment

Just create the `config.yaml` file next to the [example file](./apps/rust/poncho/config/config.sample.yaml) (with the correct data for you) and deploy with:

```sh
docker compose up
```

### Features

The main features of this app are:
- **Model name overriding on requests**: The Pocket Network will send requests for the model `pocket_network` (hopefully, if not random stuff), we take care of overriding this to match the actual backend model name.
- **Model name overriding on responses**: The model name shown in the response will match the one advertised to Pocket Network (`pocket_network`, or whatever you set). No model name leakage.
- **Logprobs limitation**: Sometimes you don't want (or can't) provide "logprobs" to your generations, so we can block any request that includes `logprobs` as parameter if you chose to.
- **Tokens to generate limitation**: You may want to limit the amount of tokens to generate to keep the backend working at a given load. This will replace the `max_tokens` or `max_completion_tokens` parameters to allow up to `config/max_tokens` tokens to generate (or limit the total context, input+output, using `config/max_position_embeddings`). We even account for input tokens and also when the number of tokens to generate is not explicit.
- **Embeddings endpoint** (`/v1/embeddings`): Accepts both OpenAI-style `input` and vLLM-style `texts`, but only text input (`input` as a string or array of strings, `texts` as an array of strings). Token-id input (numbers, integer arrays such as `[[-1, 1879]]`) is rejected with `400`, which blocks GHSA-25q3-v2hm-8vpf. Non-object bodies are rejected, the model name is always overridden, and batch size is validated against an optional `max_batched_texts_embedding` config limit.
- **Rerank endpoint** (`/v1/rerank`): Validates the number of `documents` against an optional `max_rerank_documents` config limit.
- **OpenAI-compatible models endpoint** (`/v1/models`): Returns a static models list with the configured `model_public_name`.
- **Engine role allowlist**: Only the endpoints for the configured engine role are exposed. Requests are matched on the decoded, normalized path; path traversal, `%2F`, `%5C`, `//`, and any other percent-encoded path are rejected with `404`.
- **Content-Type independent JSON bodies**: Every `POST` body is parsed as JSON regardless of the inbound `Content-Type`. Empty or malformed bodies return `400`, and only the re-serialized JSON is forwarded upstream.
- **Generation parameter validation**: `max_tokens` and `max_completion_tokens` must be positive integers and are capped by `model_config_data.max_tokens` (or rejected when `crop_max_tokens` is disabled). `temperature` must be finite and between `0` and `2`; `top_p` must be finite and between `0` and `1`. For `/v1/completions`, `prompt` must be a single string, and a request that omits `max_tokens` gets it set to `model_config_data.max_tokens` directly (no token-counting round-trip).
- **Strict chat/completions field allowlist**: `/v1/chat/completions` and `/v1/completions` accept only an explicit set of OpenAI-compatible top-level fields; anything else (e.g. `mm_processor_kwargs`, `chat_template_kwargs`, `allowed_token_ids`, `best_of`) is rejected with `400`. `n` must be `1` when present, preventing relay-side GPU multiplication.
- **Multimodal media validation**: `image_url`, `input_audio`, and optionally `video_url` parts are accepted only as inline base64 data. Declared MIME types are ignored; decoded magic bytes must be PNG/JPEG/WEBP for images, WAV/FLAC/MP3 for audio, and MP4/WebM for video when enabled. Per-item byte limits, per-request counts, image dimensions, and audio duration are enforced. Client-supplied `uuid` fields are stripped (GHSA-p92p).
- **Health check** (`/health`): Probes the backend `/health` endpoint with a 3-second timeout and returns `{"status": "healthy"}` on success.


### Engine roles

Configure `engine.role` as `chat`, `embedding`, or `reranker`. The `engine` block and its `role` field are mandatory; the sidecar fails at startup if either is missing:

| Role | Allowed `/v1` endpoints |
|------|--------------------------|
| `chat` | `GET /v1/models`, `POST /v1/chat/completions`, and `POST /v1/completions` only when `engine.serve_completions: true` |
| `embedding` | `GET /v1/models`, `POST /v1/embeddings` |
| `reranker` | `GET /v1/models`, `POST /v1/rerank`, `POST /v1/score` |

Anything else under `/v1`, including `/v1/responses`, `/v1/messages`, `*/render`, and `/v1/load_lora_adapter`, returns a JSON `404`.


### Media inputs

Media is rejected by default. The `media` block is mandatory when `engine.role` is `chat` (the sidecar fails at startup without it) and may be omitted for other roles. Set `media.limit_mm_per_prompt` to the same values used by the engine's `--limit-mm-per-prompt` configuration:

```yaml
media:
  allow_video: false
  limit_mm_per_prompt:
    image: 4
    audio: 1
    video: 0
  max_item_bytes:
    image: 5242880
    audio: 5242880
    video: 5242880
  max_image_width: 8192
  max_image_height: 8192
  max_audio_duration_seconds: 600
```

- Only inline `data:` base64 URIs are accepted; `http:`, `https:`, `file:`, and all other schemes are rejected.
- Images must be PNG, JPEG, or WEBP. Audio must be WAV, FLAC, or MP3. Video (when enabled) must be MP4 or WebM.
- Magic bytes are authoritative; declared MIME types are ignored. EPS, PostScript, PDF, TIFF, and unknown content are rejected.
- `input_audio` accepts OpenAI's `{ "data": ..., "format": ... }` form or a `data:` URL and is normalized to `data` + detected format.
- Per-item bytes, per-request item counts, image dimensions, and audio duration are enforced.
- `uuid` fields on content parts are removed before forwarding (GHSA-p92p).


### Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | `GET` | Probes backend health, returns `{"status": "healthy"}` |
| `/v1/models` | `GET` | Static OpenAI-compatible models list (showing only the overridden model name) |
| `/v1/chat/completions` | `POST` | Chat completions proxy (chat role) |
| `/v1/completions` | `POST` | Legacy completions proxy (chat role, only when `serve_completions` is enabled) |
| `/v1/embeddings` | `POST` | Embeddings with batch size validation & model override (embedding role) |
| `/v1/rerank` | `POST` | Rerank with document count validation & model override (reranker role) |
| `/v1/score` | `POST` | Score with rerank-style validation & model override (reranker role) |
| `/pokt/*` | `any` | Pocket-specific endpoints (see below) |
| anything else | `any` | Returns JSON `404` |

It also supports some pocket-specific endpoints (`/pokt/*`) to provide extra (optional) data to the public:
- `/pokt/config` : Give you the model config you want to show the public
- `/pokt/config-hash` : Returns the config hash (TODO).
- `/pokt/tokenizer` : Provides the model tokenizer (TODO)
- `/pokt/tokenizer-hash` : Returns the tokenizer hash (TODO).


