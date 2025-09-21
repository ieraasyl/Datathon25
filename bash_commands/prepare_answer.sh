python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("issai/LLama-3.1-KazLLM-1.0-8B", local_dir="models/LLama-3.1-KazLLM-1.0-8B")
snapshot_download("intfloat/multilingual-e5-base", local_dir="models/multilingual-e5-base")
snapshot_download("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", local_dir="models/paraphrase-multilingual-MiniLM-L12-v2")
PY
