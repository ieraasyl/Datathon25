python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("issai/rembert-sentiment-analysis-polarity-classification-kazakh", local_dir="models/rembert-sentiment-analysis-polarity-classification-kazakh")   # Kazakh polarity (RemBERT)
snapshot_download("cointegrated/rubert-tiny-sentiment-balanced", local_dir="models/rubert-tiny-sentiment-balanced")                       # RU tri-class sentiment
snapshot_download("nlptown/bert-base-multilingual-uncased-sentiment", local_dir="models/bert-base-multilingual-uncased-sentiment")                     # Multilingual fallback
PY
