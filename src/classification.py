from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


@dataclass(frozen=True)
class SentimentThresholds:
    # final label thresholds on fused score S ∈ [-1,1]
    pos_thr: float = 0.20
    neg_thr: float = -0.20
    # fusion weights: S = clamp(alpha * text_score + beta * emoji_sent, -1, 1)
    alpha_text: float = 0.80
    beta_emoji: float = 0.20
    # KK polarity to tri-class margins from P(pos)
    kk_pos_hi: float = 0.60
    kk_pos_lo: float = 0.40

@dataclass(frozen=True)
class ClassifyConfig:
    # Hugging Face model IDs (or local directories if cached)
    kk_model_id: str = "models/rembert-sentiment-analysis-polarity-classification-kazakh"   # binary
    ru_model_id: str = "models/rubert-tiny-sentiment-balanced"                       # 3-class
    xlmr_model_id: str = "models/bert-base-multilingual-uncased-sentiment"  # 1–5 stars
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    max_seq_len: int = 160
    thr: SentimentThresholds = field(default_factory=SentimentThresholds)


RE_Q_RU = re.compile(r"\b(как|когда|где|сколько|почему|зачем|можно|какой|какая|кто|куда|чего|что делать)\b|\?", re.IGNORECASE | re.UNICODE)
RE_Q_KK = re.compile(r"\b(қалай|қашан|қайда|неше|неге|не үшін|қайсы|кім|қайдан|қалайша)\b|\?", re.IGNORECASE | re.UNICODE)

RE_COMPL_RU = re.compile(
    r"\b(не\s*работает|ужасн|плохо|медленн|не\s*ловит|нет\s*сети|проблем|отврат|мошенн|обман|позор|теряет|не\s*подключ|связь\s*пропада|зависает|лаг|списал[аи]?|пропал[аи]? баланс)\b",
    re.IGNORECASE | re.UNICODE,
)
RE_COMPL_KK = re.compile(
    r"\b(істемейді|ақауы|нашар|қиын|жаман|желі\s*жоқ|қамту\s*жоқ|байланыс\s*жоқ|тоқтап\s*қалды|жылдамдығы\s*төмен|төлем\s*өтпеді|баланс\s*қате)\b",
    re.IGNORECASE | re.UNICODE,
)

RE_THANKS_RU = re.compile(r"\b(спасибо|благодарю|огромное\s*спасибо|большое\s*спасибо|мерси)\b|🙏", re.IGNORECASE | re.UNICODE)
RE_THANKS_KK = re.compile(r"\b(рахмет|алғыс|үлкен\s*рахмет|көп\s*рахмет)\b|🙏", re.IGNORECASE | re.UNICODE)

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _clamp11(x: float) -> float:
    return max(-1.0, min(1.0, x))


class _LazyModel:
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tok = None
        self.labels = None  # ordered list of labels per index

    def load(self):
        if self.model is None or self.tok is None:
            cfg = AutoConfig.from_pretrained(self.model_id)
            self.labels = [cfg.id2label[i] for i in range(len(cfg.id2label))] if hasattr(cfg, "id2label") else None

            # Prefer fast; if env misses optional deps (tiktoken/sentencepiece) → fallback to slow.
            try:
                self.tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            except Exception:
                # Slow tokenizer path requires `sentencepiece` for XLM-R and friends
                self.tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)

            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
        return self.tok, self.model, self.labels


class Classifier:
    """
    Usage:
        clf = Classifier()
        df2 = clf.classify_df(df)  # df has text_norm, lang_primary, emoji_sent
    """
    def __init__(self, config: ClassifyConfig = ClassifyConfig()):
        self.cfg = config
        self.m_kk = _LazyModel(self.cfg.kk_model_id, self.cfg.device)
        self.m_ru = _LazyModel(self.cfg.ru_model_id, self.cfg.device)
        self.m_xlmr = _LazyModel(self.cfg.xlmr_model_id, self.cfg.device)

    # --------- Public API ---------

    def classify_df(self, df: pd.DataFrame,
                    text_col: str = "text_norm",
                    lang_col: str = "lang_primary",
                    emoji_col: str = "emoji_sent") -> pd.DataFrame:
        if text_col not in df.columns:
            raise KeyError(f"Input DataFrame is missing '{text_col}'")
        if lang_col not in df.columns:
            raise KeyError(f"Input DataFrame is missing '{lang_col}'")
        if emoji_col not in df.columns:
            emoji = [0.0] * len(df)
        else:
            emoji = df[emoji_col].fillna(0.0).astype(float).tolist()

        texts = df[text_col].fillna("").astype(str).tolist()
        langs = df[lang_col].fillna("other").astype(str).tolist()

        # 1) Type classification (deterministic, language-aware)
        types, type_rules = self._batch_type(texts, langs,
                                             has_kk=df.get("has_kk_chars"),
                                             has_ru=df.get("has_ru_chars"))

        # 2) Sentiment routing + scoring
        s_scores, s_labels, s_models, s_probs = self._batch_sentiment(texts, langs, emoji,
                                                                      has_kk=df.get("has_kk_chars"),
                                                                      has_ru=df.get("has_ru_chars"))

        out = df.copy()
        out["type"] = types
        out["type_rules"] = [json.dumps(r, ensure_ascii=False) for r in type_rules]
        out["sentiment"] = s_labels
        out["sentiment_score"] = [round(float(s), 4) for s in s_scores]
        out["sentiment_model"] = s_models
        out["sentiment_probs_json"] = [json.dumps(p, ensure_ascii=False) for p in s_probs]
        return out


    def _batch_type(self, texts: List[str], langs: List[str],
                    has_kk: Optional[pd.Series] = None,
                    has_ru: Optional[pd.Series] = None) -> Tuple[List[str], List[List[str]]]:
        out_labels, out_rules = [], []
        for i, (t, l) in enumerate(zip(texts, langs)):
            hk = bool(has_kk.iloc[i]) if has_kk is not None else None
            hr = bool(has_ru.iloc[i]) if has_ru is not None else None
            label, rules = self._classify_type_single(t, l, hk, hr)
            out_labels.append(label)
            out_rules.append(rules)
        return out_labels, out_rules

    def _classify_type_single(self, text: str, lang: str,
                              has_kk: Optional[bool],
                              has_ru: Optional[bool]) -> Tuple[str, List[str]]:
        t = text or ""
        rules: List[str] = []
        is_q = False
        is_compl = False
        is_thanks = False

        # Check RU/KK depending on lang; for 'mix' or 'other' check both
        def check_ru():
            nonlocal is_q, is_compl, is_thanks
            if RE_Q_RU.search(t):
                is_q = True; rules.append("ru:question")
            if RE_COMPL_RU.search(t):
                is_compl = True; rules.append("ru:complaint")
            if RE_THANKS_RU.search(t):
                is_thanks = True; rules.append("ru:thanks")

        def check_kk():
            nonlocal is_q, is_compl, is_thanks
            if RE_Q_KK.search(t):
                is_q = True; rules.append("kk:question")
            if RE_COMPL_KK.search(t):
                is_compl = True; rules.append("kk:complaint")
            if RE_THANKS_KK.search(t):
                is_thanks = True; rules.append("kk:thanks")

        if lang == "ru":
            check_ru()
        elif lang == "kk":
            check_kk()
        else:
            # mix/other/en → try both, prioritize explicit matches
            if has_ru or lang == "en":
                check_ru()
            if has_kk or lang in ("mix","other"):
                check_kk()

        # Priority: thanks > complaint > question > review > other
        if is_thanks:
            return "thanks", rules
        if is_compl:
            return "complaint", rules
        if is_q:
            return "question", rules

        # Heuristic review detection (sentiment words w/o question/complaint)
        if re.search(r"\b(керемет|жақсы|ұнады|супер|өте жақсы|отлично|нормально|понравилось|классно|ужасно|плохо)\b",
                     t, flags=re.IGNORECASE | re.UNICODE):
            rules.append("review_keywords")
            return "review", rules

        return "review", rules  # default to review (generic feedback)


    def _batch_sentiment(self, texts: List[str], langs: List[str], emojis: List[float],
                         has_kk: Optional[pd.Series] = None,
                         has_ru: Optional[pd.Series] = None) -> Tuple[List[float], List[str], List[str], List[Dict[str, float]]]:
        idx_ru = [i for i, l in enumerate(langs) if l == "ru"]
        idx_kk = [i for i, l in enumerate(langs) if l == "kk"]
        # route mix/other/en by script presence; if both or none → xlmr
        idx_xlmr = [i for i, l in enumerate(langs) if l in ("mix", "other", "en")]

        scores = [0.0]*len(texts)
        labels = ["neutral"]*len(texts)
        models = [""]*len(texts)
        probs_all: List[Dict[str,float]] = [dict(negative=0.0, neutral=1.0, positive=0.0) for _ in texts]

        # RU tri-class
        if idx_ru:
            probs = self._score_triclass(self.m_ru, texts, idx_ru)
            for j, i in enumerate(idx_ru):
                s_text = float(probs[j]["positive"]) - float(probs[j]["negative"])
                s = self._fuse_score(s_text, emojis[i])
                scores[i], labels[i] = s, self._label_from_score(s)
                models[i] = "ru_rubert_tiny_snt"
                probs_all[i] = probs[j]

        # KK binary → tri-class via margins + emoji fusion
        if idx_kk:
            ppos = self._score_binary(self.m_kk, texts, idx_kk)
            for j, i in enumerate(idx_kk):
                p = float(ppos[j])  # P(positive)
                # Build a tri-class prob proxy for logging
                pos = p
                neg = 1.0 - p
                neu = max(0.0, 1.0 - abs(p - 0.5)*2.0) * 0.5  # soft belly around 0.5
                Z = pos + neg + neu
                probs_all[i] = {"negative": neg/Z, "neutral": neu/Z, "positive": pos/Z}

                # text-only score in [-1,1]
                s_text = 2.0*p - 1.0
                s = self._fuse_score(s_text, emojis[i])
                scores[i], labels[i] = s, self._label_from_score(s)
                models[i] = "kk_rembert_polarity"

        # XLM-R multilingual fallback (mix/other)
        if idx_xlmr:
            probs = self._score_triclass(self.m_xlmr, texts, idx_xlmr)
            for j, i in enumerate(idx_xlmr):
                s_text = float(probs[j]["positive"]) - float(probs[j]["negative"])
                s = self._fuse_score(s_text, emojis[i])
                scores[i], labels[i] = s, self._label_from_score(s)
                models[i] = "xlmr_multilingual_snt"
                probs_all[i] = probs[j]

        return scores, labels, models, probs_all


    def _score_triclass(self, lazy_model: _LazyModel,
                        texts: List[str], indices: List[int]) -> List[Dict[str, float]]:
        tok, mdl, labels = lazy_model.load()
        # Map whatever labels are in the model to standard keys
        # Try to find indices by lowercased label names
        def map_idx():
            name_map = {"negative": None, "neutral": None, "positive": None}
            if labels:
                lower = [str(x).lower() for x in labels]
                for k in list(name_map.keys()):
                    if k in lower:
                        name_map[k] = lower.index(k)
            # Default common order (0:neg,1:neu,2:pos) if not found
            if name_map["negative"] is None or name_map["positive"] is None:
                name_map = {"negative": 0, "neutral": 1 if mdl.num_labels == 3 else None, "positive": mdl.num_labels-1}
            return name_map

        idx_map = map_idx()
        outs: List[Dict[str, float]] = []
        bs = self.cfg.batch_size
        for start in range(0, len(indices), bs):
            part = indices[start:start+bs]
            batch = [texts[i] for i in part]
            enc = tok(batch, return_tensors="pt", truncation=True, padding=True,
                      max_length=self.cfg.max_seq_len).to(mdl.device)
            with torch.no_grad():
                logits = mdl(**enc).logits.cpu().numpy()
                probs = _softmax(logits, axis=1)
            for p in probs:
                neg = float(p[idx_map["negative"]])
                pos = float(p[idx_map["positive"]])
                neu = float(p[idx_map["neutral"]]) if (mdl.num_labels == 3 and idx_map["neutral"] is not None) else max(0.0, 1.0 - neg - pos)
                Z = max(1e-9, neg + neu + pos)
                outs.append({"negative": neg/Z, "neutral": neu/Z, "positive": pos/Z})
        return outs

    def _score_binary(self, lazy_model: _LazyModel,
                      texts: List[str], indices: List[int]) -> List[float]:
        tok, mdl, labels = lazy_model.load()
        # Find which index is "positive"
        pos_idx = 1
        if labels:
            low = [str(x).lower() for x in labels]
            if "positive" in low:
                pos_idx = low.index("positive")
            elif "pos" in low:
                pos_idx = low.index("pos")
        outs: List[float] = []
        bs = self.cfg.batch_size
        for start in range(0, len(indices), bs):
            part = indices[start:start+bs]
            batch = [texts[i] for i in part]
            enc = tok(batch, return_tensors="pt", truncation=True, padding=True,
                      max_length=self.cfg.max_seq_len).to(mdl.device)
            with torch.no_grad():
                logits = mdl(**enc).logits.cpu().numpy()
                # Binary: softmax and take P(pos_idx)
                pr = _softmax(logits, axis=1)[:, pos_idx]
            outs.extend(pr.tolist())
        return outs


    def _fuse_score(self, s_text: float, s_emoji: float) -> float:
        thr = self.cfg.thr
        s = thr.alpha_text * float(s_text) + thr.beta_emoji * float(s_emoji)
        return _clamp11(s)

    def _label_from_score(self, s: float) -> str:
        thr = self.cfg.thr
        if s >= thr.pos_thr:
            return "positive"
        if s <= thr.neg_thr:
            return "negative"
        return "neutral"


def classify_df(df: pd.DataFrame,
                text_col: str = "text_norm",
                lang_col: str = "lang_primary",
                emoji_col: str = "emoji_sent",
                config: ClassifyConfig = ClassifyConfig()) -> pd.DataFrame:
    return Classifier(config=config).classify_df(df, text_col=text_col, lang_col=lang_col, emoji_col=emoji_col)
