python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("cointegrated/rubert-tiny-toxicity", local_dir="models/rubert-tiny-toxicity")
snapshot_download("textdetox/bert-multilingual-toxicity-classifier", local_dir="models/bert-multilingual-toxicity-classifier")
PY
