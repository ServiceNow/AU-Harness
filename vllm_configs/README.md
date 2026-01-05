# Launching vLLM endpoints

This folder contains a Kubernetes deployment example (`k8.yaml`) and guidance
for launching vLLM endpoints that can serve LALMs. 


You can use any one of the two recommended approaches below:
- Local: run a vLLM server on your workstation or VM (good for development).
- Kubernetes: deploy the provided `k8.yaml` to a GPU-capable cluster.


Keep these high-level notes in mind:
- Do NOT commit real secrets (Hugging Face tokens) into source control. Use
	Kubernetes Secrets or environment variables stored securely.
- The `k8.yaml` file uses a placeholder image (`<YOUR_VLLM_IMAGE_WITH_AUDIO_DEPENDANCIES_INSTALLED>`).
	Replace that with an image that has required audio dependencies (ffmpeg, soundfile, librosa, torchaudio,
	any model-specific libs) before applying.
- The example exposes ports 8000..8007. If you only need a single instance,
	reducing the number of containers/ports in the Pod is fine.



**Useful links**

- vLLM docs (overview & quickstart): https://docs.vllm.ai/en/latest/getting_started/quickstart/
- vLLM CLI `serve` docs: https://docs.vllm.ai/en/latest/cli/serve/
- vLLM Kubernetes / deployment docs: https://docs.vllm.ai/en/latest/deployment/k8s/
- vLLM audio / multimodal docs and examples:
	- Audio assets API: https://docs.vllm.ai/en/latest/api/vllm/assets/audio/
	- Audio example (offline / language + audio): https://docs.vllm.ai/en/latest/examples/offline_inference/audio_language/

These audio-specific links describe how vLLM handles audio assets, required
dependencies and example code for audio-language workflows.






## **A. Local (development)**

1) Prerequisites

- GPU node or a machine with a compatible PyTorch/CUDA setup (or CPU only for small models).
- Python 3.10+ and a virtual environment is recommended.
- A Hugging Face token with access to the model, set in `HUGGING_FACE_HUB_TOKEN`.

2) Install vLLM (recommended minimal steps)

```bash
# create & activate a venv (example using uv as in vLLM docs, or use python -m venv)
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# install vllm and choose a torch backend if needed
pip install vllm --upgrade

# macOS (Homebrew):
brew install ffmpeg libsndfile
pip install soundfile librosa torchaudio

# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
pip install soundfile librosa torchaudio
```

3) Start the server

The vLLM CLI provides a `serve` entrypoint that starts an OpenAI-compatible HTTP
server. Example:

```bash
# serve a HF model on localhost:8000
export HUGGING_FACE_HUB_TOKEN="<YOUR_HF_TOKEN>"
vllm serve microsoft/Phi-4-multimodal-instruct --port 8000 --host 0.0.0.0
```

Notes:
- Use `--api-key` or set `VLLM_API_KEY` if you want the server to require an API key.
- Many LALMs need additional Python packages or system
	libraries. Commonly required packages: `soundfile`, `librosa`, `torchaudio`,
	and system `ffmpeg`/`libsndfile`. The exact requirements depend on the model
	and any tokenizer/preprocessor it uses. Check the model's Hugging Face page
	and the vLLM audio docs linked above.
- If you plan to use GPU acceleration, ensure a compatible PyTorch/CUDA
	combination is installed in the environment (or use vLLM Docker images with
	prebuilt CUDA support). If you run into missing symbols, check CUDA/PyTorch
	compatibility and rebuild or pick a different image.

4) Point `run_configs` to the local endpoint

Update your run config to use the local server URL (example YAML snippet):

```yaml
# example run_configs entry
# For OpenAI-compatible API calls use endpoints like /v1/completions or /v1/chat/completions
url: "http://localhost:8000/v1/completions"
```





## **B. Kubernetes — use the provided `k8.yaml`**

What the example does:

- Launches a single Pod template containing multiple vLLM containers (ports 8000..8007).
- Each container is configured with the same model and listens on a distinct port.
- A `Service` of type `NodePort` exposes the Pod ports on the cluster nodes.

Pre-apply checklist (LALMs)

1. Replace the placeholder image in `k8.yaml`:

	 - Find and replace `<YOUR_VLLM_IMAGE_WITH_AUDIO_DEPENDANCIES_INSTALLED>` with an image
		 that includes:
		 - vLLM installed
		 - Python audio libs used by your model: `soundfile`, `librosa`, `torchaudio`, etc.
		 - System binaries: `ffmpeg` and `libsndfile` (or equivalents).

2. Secrets: create a Kubernetes Secret for your Hugging Face token, e.g.:

```bash
kubectl -n <namespace> create secret generic hf-token \
	--from-literal=HUGGING_FACE_HUB_TOKEN='<YOUR_HF_TOKEN>'
```

Then update `k8.yaml` container env to use `valueFrom.secretKeyRef` instead of a plain `value`.

3. Cluster requirements

- GPU-enabled nodes and drivers (matching the image / CUDA version)
- If using Run:AI or a custom scheduler, ensure `schedulerName` matches your cluster. Remove
	or edit `schedulerName` if not applicable.

Apply the example

```bash
# make any replacements (image, secret references), then:
kubectl apply -f vllm_configs/k8.yaml

# monitor rollout
kubectl -n <namespace> rollout status deployment/infer-phi4-multimodal-instruct
kubectl -n <namespace> get pods -l app=infer-phi4-multimodal-instruct
```

Accessing the service

- The `Service` in `k8.yaml` is `NodePort`. To see which node port range your cluster assigned,
	run:

```bash
kubectl -n <namespace> get svc infer-phi4-multimodal-instruct-service -o wide
```

- You can then use `http://<node-ip>:<nodePort>` for the port you want (8000..8007 map to
	cluster node ports). For production, consider exposing via `LoadBalancer` or an Ingress.



Troubleshooting:
- Check container logs: `kubectl -n <namespace> logs <pod> -c deployment0` (replace container name).
- If model fails to load: check `HUGGING_FACE_HUB_TOKEN`, image CUDA/PyTorch compatibility, and
	that `--trust_remote_code` is set only when you trust the model repo.

