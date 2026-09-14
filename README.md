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
- **Embeddings endpoint** (`/v1/embeddings`): Accepts both OpenAI-style `input` and vLLM-style `texts` arrays. Validates batch size against an optional `max_batched_texts_embedding` config limit.
- **Rerank endpoint** (`/v1/rerank`): Validates the number of `documents` against an optional `max_rerank_documents` config limit.
- **OpenAI-compatible models endpoint** (`/v1/models`): Returns a static models list with the configured `model_public_name`.
- **Health check** (`/health`): Probes the backend `/health` endpoint with a 3-second timeout and returns `{"status": "healthy"}` on success.


### Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | `GET` | Probes backend health, returns `{"status": "healthy"}` |
| `/v1/models` | `any` | Static OpenAI-compatible models list (showing only the overridden model name) |
| `/v1/embeddings` | `POST` | Embeddings with batch size validation & model override |
| `/v1/rerank` | `POST` | Rerank with document count validation & model override |
| `/v1/*` | `any` | Proxy everything else to the vLLM backend |
| `/pokt/*` | `any` | Pocket-specific endpoints (see below) |
| anything else | `any` | Returns JSON `404` |

It also supports some pocket-specific endpoints (`/pokt/*`) to provide extra (optional) data to the public:
- `/pokt/config` : Give you the model config you want to show the public
- `/pokt/config-hash` : Returns the config hash (TODO).
- `/pokt/tokenizer` : Provides the model tokenizer (TODO)
- `/pokt/tokenizer-hash` : Returns the tokenizer hash (TODO).


