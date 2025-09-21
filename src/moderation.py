from __future__ import annotations

import os
import re
import math
import json
import torch

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass(frozen=True)
class ModerationThresholds:
    # Toxicity decision thresholds (tune as needed)
    ru_toxicity_thr: float = 0.50
    mtox_toxicity_thr: float = 0.50
    # Spam heuristics
    max_emoji_ratio: float = 0.35   # emojis / total chars
    max_repeat_char_run: int = 4    # '!!!!!!', 'ааааа'
    min_link_count_spam: int = 2    # >=2 links => spammy
    max_unique_chars_short: int = 6 # short low-diversity strings are likely spam
    phone_regex_hits_spam: int = 1  # >=1 phone hits => spammy
    min_caps_ratio_for_shouting: float = 0.6
    spam_score_spam_thr: float = 0.65

@dataclass(frozen=True)
class ModerationConfig:
    # HF model IDs (can be local directories if pre-downloaded)
    ru_model_id: str = "models/rubert-tiny-toxicity"
    mtox_model_id: str = "models/bert-multilingual-toxicity-classifier"
    profanity_ru_path: Optional[str] = "data/profanity_ru.txt"
    profanity_kk_path: Optional[str] = "data/profanity_kk.txt"
    spam_keywords_ru_path: Optional[str] = "data/spam_keywords_ru.txt"
    spam_keywords_kk_path: Optional[str] = "data/spam_keywords_kk.txt"
    # Inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    max_seq_len: int = 192
    thresholds: ModerationThresholds = field(default_factory=ModerationThresholds)


RE_LINK = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
RE_MENTION = re.compile(r"(^|[^@\w])@[\w_]{3,}", re.UNICODE)
RE_HASHTAG = re.compile(r"(?:^|\s)#\w{2,}", re.UNICODE)
RE_PHONE = re.compile(r"(?:(?:\+?\d{1,3})?[-.\s()]*)?(?:\d[-.\s()]*){9,}", re.UNICODE)
RE_REPEAT = re.compile(r"(.)\1{3,}", re.UNICODE)  # 4+ same chars in a row
RE_WORD = re.compile(r"\w+", re.UNICODE)

def _load_lexicon(path: Optional[str]) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    words: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            words.append(s.lower())
    return sorted(set(words))

def _compile_terms_regex(terms: List[str]) -> Optional[re.Pattern]:
    if not terms:
        return None
    escaped = [re.escape(t) for t in terms if t]
    # word-ish boundaries, but keep flexible for non-latin scripts
    pattern = r"(?<!\w)(" + r"|".join(escaped) + r")(?!\w)"
    try:
        return re.compile(pattern, re.IGNORECASE | re.UNICODE)
    except re.error:
        # Fallback to simple contains checks if regex is too large
        return None

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class _LazyModel:
    """Lazy wrapper for HF models to avoid loading if not used."""
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tok = None

    def load(self):
        if self.model is None or self.tok is None:
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
        return self.tok, self.model


class Moderator:
    """
    Usage:
        mod = Moderator()
        df2 = mod.moderate_df(df)  # df has text_norm, lang_primary
    """
    def __init__(self, config: ModerationConfig = ModerationConfig()):
        self.cfg = config

        # Load lexicons (optional)
        self.lex_ru = _load_lexicon(self.cfg.profanity_ru_path)
        self.lex_kk = _load_lexicon(self.cfg.profanity_kk_path)
        self.spam_ru = _load_lexicon(self.cfg.spam_keywords_ru_path)
        self.spam_kk = _load_lexicon(self.cfg.spam_keywords_kk_path)

        # Pre-compile where possible
        self.re_prof_ru = _compile_terms_regex(self.lex_ru)
        self.re_prof_kk = _compile_terms_regex(self.lex_kk)
        self.re_spam_ru = _compile_terms_regex(self.spam_ru)
        self.re_spam_kk = _compile_terms_regex(self.spam_kk)

        # Lazy models
        self.ru_model = _LazyModel(self.cfg.ru_model_id, self.cfg.device)
        self.mtox_model = _LazyModel(self.cfg.mtox_model_id, self.cfg.device)

    def moderate_df(self, df: pd.DataFrame,
                    text_col: str = "text_norm",
                    lang_col: str = "lang_primary") -> pd.DataFrame:
        if text_col not in df.columns:
            raise KeyError(f"Input DataFrame is missing '{text_col}'")
        if lang_col not in df.columns:
            raise KeyError(f"Input DataFrame is missing '{lang_col}'")

        texts = df[text_col].fillna("").astype(str).tolist()
        langs = df[lang_col].fillna("other").astype(str).tolist()

        # 1) Profanity + spam (pure regex/heuristics)
        prof = [self._profanity_info(t, l) for t, l in zip(texts, langs)]
        spam = [self._spam_score(t) for t in texts]

        # 2) Toxicity models (batched by routing)
        tox_scores, tox_labels, tox_models = self._batched_toxicity(texts, langs)

        out = df.copy()
        out["has_profanity"] = [p["has_profanity"] for p in prof]
        out["profanity_hits"] = [p["hits"] for p in prof]
        out["spam_score"] = [s["score"] for s in spam]
        out["is_spam"] = [s["score"] >= self.cfg.thresholds.spam_score_spam_thr for s in spam]
        out["toxicity_model"] = tox_models
        out["toxicity_score"] = [round(s, 4) for s in tox_scores]
        out["toxicity_label"] = tox_labels
        out["moderation_reasons"] = [json.dumps(self._reasons(p, s, ts, tl), ensure_ascii=False)
                                     for p, s, ts, tl in zip(prof, spam, tox_scores, tox_labels)]
        return out

    def _reasons(self, prof, spam, tox_score, tox_label) -> List[str]:
        r = []
        if prof["has_profanity"]:
            r.append(f"profanity:{prof['hits']}")
        if spam["score"] >= self.cfg.thresholds.spam_score_spam_thr:
            r.append(f"spam:{round(spam['score'],2)}")
        if tox_label == "toxic":
            r.append(f"toxicity:{round(tox_score,2)}")
        return r

    def _profanity_info(self, text: str, lang: str) -> Dict[str, object]:
        hits = 0
        # Language-aware regex; mixed texts checked by both
        if lang == "ru" or lang == "other":
            hits += self._count_hits(text, self.re_prof_ru, self.lex_ru)
        if lang == "kk" or lang == "other":
            hits += self._count_hits(text, self.re_prof_kk, self.lex_kk)
        if lang not in ("ru", "kk", "other"):  # 'mix' or anything else -> both
            hits += self._count_hits(text, self.re_prof_ru, self.lex_ru)
            hits += self._count_hits(text, self.re_prof_kk, self.lex_kk)
        return {"has_profanity": hits > 0, "hits": int(hits)}

    @staticmethod
    def _count_hits(text: str, compiled: Optional[re.Pattern], fallback_terms: List[str]) -> int:
        if not text:
            return 0
        if compiled is not None:
            return len(compiled.findall(text))
        # Fallback: simple contains checks
        tl = text.lower()
        return sum(1 for w in fallback_terms if w and w in tl)

    def _spam_score(self, text: str) -> Dict[str, object]:
        """
        Lightweight, interpretable score in [0..1].
        """
        if not text:
            return {"score": 0.0}

        score = 0.0
        L = len(text)
        unique_ratio = len(set(text)) / max(1, L)

        # Links / mentions / hashtags / phones
        link_count = len(RE_LINK.findall(text))
        mention_count = len(RE_MENTION.findall(text))
        hashtag_count = len(RE_HASHTAG.findall(text))
        phone_count = len(RE_PHONE.findall(text))
        score += min(0.5, 0.15 * link_count)
        score += min(0.2, 0.05 * mention_count)
        score += min(0.2, 0.04 * hashtag_count)
        if phone_count >= self.cfg.thresholds.phone_regex_hits_spam:
            score += 0.25

        # Repeated chars / shouting
        if RE_REPEAT.search(text):
            score += 0.15
        letters = [c for c in text if c.isalpha()]
        if letters:
            cap_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if cap_ratio >= self.cfg.thresholds.min_caps_ratio_for_shouting:
                score += 0.1

        # Emoji-heavy / low-diversity text
        emoji_count = sum(1 for c in text if c in emoji_chars)
        if L >= 6 and (emoji_count / L) >= self.cfg.thresholds.max_emoji_ratio:
            score += 0.15
        if L <= 18 and unique_ratio <= 0.35:
            score += 0.1
        if L <= 8 and len(RE_WORD.findall(text)) <= 1 and unique_ratio <= 0.3:
            score += 0.1

        # Spam keywords (language-agnostic here; lexicons are optional)
        added_kw = False
        if self.re_spam_ru and self.re_spam_ru.search(text):
            score += 0.15
            added_kw = True
        if self.re_spam_kk and self.re_spam_kk.search(text):
            score += 0.15
            added_kw = True
        # If both matched, cap the keyword contribution
        if added_kw:
            score = min(1.0, score)

        return {"score": float(max(0.0, min(1.0, score)))}

    def _batched_toxicity(self, texts: List[str], langs: List[str]) -> Tuple[List[float], List[str], List[str]]:
        # Route indices
        idx_ru = [i for i, l in enumerate(langs) if l == "ru"]
        idx_mx = [i for i, l in enumerate(langs) if l in ("kk", "mix", "other", "en")]

        scores = [0.0] * len(texts)
        labels = ["non_toxic"] * len(texts)
        models = ["none"] * len(texts)

        if idx_ru:
            batch_texts = [texts[i] for i in idx_ru]
            scr = self._score_rubert_tiny(batch_texts)
            thr = self.cfg.thresholds.ru_toxicity_thr
            for j, i in enumerate(idx_ru):
                s = float(scr[j])
                scores[i] = s
                labels[i] = "toxic" if s >= thr else "non_toxic"
                models[i] = "ru_rubert_tiny"

        if idx_mx:
            batch_texts = [texts[i] for i in idx_mx]
            scr = self._score_textdetox_mbert(batch_texts)
            thr = self.cfg.thresholds.mtox_toxicity_thr
            for j, i in enumerate(idx_mx):
                s = float(scr[j])
                scores[i] = s
                labels[i] = "toxic" if s >= thr else "non_toxic"
                models[i] = "mbert_textdetox"

        return scores, labels, models

    def _score_rubert_tiny(self, texts: List[str]) -> np.ndarray:
        """
        Returns toxicity score in [0..1] using the official aggregation:
            tox = 1 - P(non_toxic) * (1 - P(dangerous))
        """
        tok, mdl = self.ru_model.load()
        outs = []
        bs = self.cfg.batch_size
        for start in range(0, len(texts), bs):
            batch = texts[start:start+bs]
            enc = tok(batch, return_tensors="pt", truncation=True, padding=True,
                      max_length=self.cfg.max_seq_len).to(mdl.device)
            with torch.no_grad():
                logits = mdl(**enc).logits  # multilabel
                proba = torch.sigmoid(logits).cpu().numpy()  # shape [B, 5]
            # columns: [non-toxic, insult, obscenity, threat, dangerous] (per model card)
            p_non_toxic = proba[:, 0]
            p_dangerous = proba[:, -1]
            tox = 1.0 - p_non_toxic * (1.0 - p_dangerous)
            outs.append(tox)
        return np.concatenate(outs, axis=0) if outs else np.array([], dtype=np.float32)

    def _score_textdetox_mbert(self, texts: List[str]) -> np.ndarray:
        """
        Returns P(toxic) in [0..1]. textdetox: idx0=neutral, idx1=toxic
        """
        tok, mdl = self.mtox_model.load()
        outs = []
        bs = self.cfg.batch_size
        for start in range(0, len(texts), bs):
            batch = texts[start:start+bs]
            enc = tok(batch, return_tensors="pt", truncation=True, padding=True,
                      max_length=self.cfg.max_seq_len).to(mdl.device)
            with torch.no_grad():
                logits = mdl(**enc).logits  # [B, 2]
                prob = _softmax(logits.cpu().numpy(), axis=1)[:, 1]
            outs.append(prob)
        return np.concatenate(outs, axis=0) if outs else np.array([], dtype=np.float32)

def moderate_df(df: pd.DataFrame,
                text_col: str = "text_norm",
                lang_col: str = "lang_primary",
                config: ModerationConfig = ModerationConfig()) -> pd.DataFrame:
    return Moderator(config=config).moderate_df(df, text_col=text_col, lang_col=lang_col)


# Keep it local to avoid new deps; this small core set is enough for ratio-based signals.
emoji_chars = set("😀😃😄😁😆😅😂🤣😊🙂😉😍😘😗😚😙😋😛😜🤪🤨🧐🤓😎🥳🤩😡🤬😠😤😢😭😞😔😟😕🙁☹️😣😖😫😩😱😨😰😯😮😲🤯😳🤢🤮🤧🤒🤕🥵🥶🥴🤥🤡👿💀☠️👻👽👾🤖💩🙏👍👎👏🙌👌✌️🤞🤝💪🔥💯✨🎉🎊❤️🧡💛💚💙💜🖤🤍🤎💔")
