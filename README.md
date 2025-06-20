# Poncho - A sidecar for ML in the Pocket Network

The poncho is a performant Rust app that is designed to cover your vLLM endpoint and keep it warm and cozy doing its stuff while it connects to the POKT Network.
Currently it supports:
- Custom `/pokt` endpoints to provide extra (optional) data to the public:
    - `/pokt/config` : Give you the model config you want to show the public
    - `/pokt/tokenizer` : Provides the model tokenizer (TODO)
- Model name overriding on requests: The POKT Network will send requests for the model `pocket_network`, if your model is not deployed under this name, you will need to override that request and replace it with your real model name.
- Logprobs limitation:  Sometimes you don't want (or cant) provide "logprobs" to your generations, so we can block any request that includes `logprobs` as parameter.