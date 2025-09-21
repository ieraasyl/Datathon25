from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import fasttext
import emoji as emoji_lib
import regex as re_u
from ftfy import fix_text



@dataclass(frozen=True)
class Config:
    lid_model_path: str = "models/lid.176.bin"
    emoji_sentiment_csv: Optional[str] = None  # if None -> use built-in fallback
    mix_min_secondary_share: float = 0.25
    mix_min_conf: float = 0.30
    max_tokens_for_per_token_lid: int = 40

# Kazakh/Russian script hints
KAZAKH_SPEC_CHARS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
RUSSIAN_SPEC_CHARS = set("ёыэъщЁЫЭЪЩ")

# Small fallback emoji polarity (augment via CSV if provided)
FALLBACK_EMOJI_POLARITY: Dict[str, float] = {
    "😀": 0.9, "😃": 0.9, "😄": 0.9, "😁": 0.9, "😊": 0.8, "🙂": 0.5, "🤗": 0.6,
    "😍": 0.9, "😘": 0.8, "😅": 0.4, "😂": 0.7, "🤣": 0.7, "👍": 0.7, "👏": 0.7,
    "😐": 0.0, "😶": 0.0, "🤨": -0.2, "😕": -0.4, "🙁": -0.5, "☹️": -0.6,
    "😒": -0.5, "😞": -0.6, "😔": -0.5, "😡": -0.9, "🤬": -1.0, "👎": -0.7,
    "😭": -0.7, "😢": -0.6, "💔": -0.8, "🔥": 0.2, "💯": 0.6, "🤝": 0.4
}


RE_SPACES = re.compile(r"\s+")
RE_CTRL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
RE_APOSTROPHE = re.compile(r"[’`´]")
RE_PUNCT_RUNS = re.compile(r"([!?.,…])\1{1,}")
RE_WORD = re.compile(r"\w+", re.UNICODE)
RE_EMOJI = re_u.compile(r"\p{Emoji_Presentation}|\p{Emoji}\uFE0F", re_u.UNICODE)


_ft_model = None
_emoji_polarity: Dict[str, float] = {}


class Preprocessor:
    def __init__(self, config: Config = Config()):
        self.cfg = config
        self._ensure_emoji_polarity_loaded()

    def preprocess_df(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Transform a DataFrame with at least `text_col` into a feature-enriched DataFrame.

        Adds:
          text_raw, text_norm, emojis, emojis_count, emoji_sent,
          lang_primary, lang_probs (JSON), is_mixed, mix_ratio,
          has_kk_chars, has_ru_chars
        """
        if text_col not in df.columns:
            raise KeyError(f"Input DataFrame is missing required column '{text_col}'")
        enrich = df[text_col].apply(self._process_text).apply(pd.Series)
        keep_cols = [c for c in df.columns if c not in enrich.columns]
        out = pd.concat([df[keep_cols], enrich], axis=1)

        front = [
            "text_raw", "text_norm", "emojis", "emojis_count", "emoji_sent",
            "lang_primary", "lang_probs", "is_mixed", "mix_ratio",
            "has_kk_chars", "has_ru_chars"
        ]
        ordered = [c for c in front if c in out.columns] + [c for c in out.columns if c not in front]
        return out.loc[:, ordered]

    # Optional helper for nested Instagram JSON structures
    @staticmethod
    def flatten_our_json(obj: Union[Mapping, List[Mapping]]) -> pd.DataFrame:
        """
        Accepts:
          - list[{ post_url, comments:[{text, username, like_count, created_at_utc}, ...], ... }]
          - { post_url, comments:[...] }
          - already-flat list[comment]
        Returns a flat DataFrame with at least a 'text' column.
        """
        if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "comments" in obj[0]:
            rows = []
            for post in obj:
                post_url = post.get("post_url")
                batch_extracted_at = post.get("extracted_at")
                total_comments = post.get("total_comments")
                note = post.get("note")
                for c in (post.get("comments") or []):
                    rows.append({
                        "text": c.get("text", ""),
                        "username": c.get("username"),
                        "like_count": c.get("like_count"),
                        "created_at_utc": c.get("created_at_utc"),
                        "source_post_url": post_url,
                        "batch_extracted_at": batch_extracted_at,
                        "post_total_comments": total_comments,
                        "post_note": note,
                    })
            return pd.DataFrame(rows)
        if isinstance(obj, dict) and "comments" in obj:
            return Preprocessor.flatten_our_json([obj])
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        return pd.json_normalize(obj)


    def _process_text(self, text: str) -> Dict:
        tx_raw = text if isinstance(text, str) else ""
        tx = self._normalize_keep_emojis(tx_raw)

        emojis = self._extract_emojis(tx)
        emoji_sent = self._emoji_sentiment_score(emojis)

        preds = self._ft_topk_lang(tx, k=3) if tx else []
        lang_primary = self._lang_primary_from_preds(preds)
        lang_probs = {l: round(p, 4) for l, p in preds}

        is_mixed, mix_ratio = self._detect_mixed_kk_ru(tx, lang_primary)

        return {
            "text_raw": tx_raw,
            "text_norm": tx,
            "emojis": "".join(emojis),
            "emojis_count": len(emojis),
            "emoji_sent": round(emoji_sent, 4),
            "lang_primary": lang_primary,
            "lang_probs": json.dumps(lang_probs, ensure_ascii=False),
            "is_mixed": bool(is_mixed),
            "mix_ratio": round(float(mix_ratio), 4),
            "has_kk_chars": self._has_kk_chars(tx),
            "has_ru_chars": self._has_ru_chars(tx),
        }

    @staticmethod
    def _normalize_keep_emojis(text: str) -> str:
        t = fix_text(text or "")
        t = RE_APOSTROPHE.sub("'", t)
        t = RE_PUNCT_RUNS.sub(r"\1\1", t)
        t = RE_CTRL.sub(" ", t)
        t = RE_SPACES.sub(" ", t).strip()
        return t

    @staticmethod
    def _extract_emojis(text: str) -> List[str]:
        lst = [m["emoji"] for m in emoji_lib.emoji_list(text)]
        if lst:
            return lst
        return RE_EMOJI.findall(text)

    def _emoji_sentiment_score(self, emojis: List[str]) -> float:
        if not emojis:
            return 0.0
        vals = [self._emoji_polarity.get(e) for e in emojis if e in self._emoji_polarity]
        if not vals:
            return 0.0
        return max(-1.0, min(1.0, sum(vals) / len(vals)))

    @staticmethod
    def _has_kk_chars(text: str) -> bool:
        return any(ch in KAZAKH_SPEC_CHARS for ch in text)

    @staticmethod
    def _has_ru_chars(text: str) -> bool:
        return any(ch in RUSSIAN_SPEC_CHARS for ch in text)

    def _ft_model(self):
        global _ft_model
        if _ft_model is None:
            if not os.path.exists(self.cfg.lid_model_path):
                raise FileNotFoundError(
                    f"fastText LID model not found at {self.cfg.lid_model_path}. "
                    f"Download lid.176.bin and set Config.lid_model_path."
                )
            _ft_model = fasttext.load_model(self.cfg.lid_model_path)
        return _ft_model

    def _ft_topk_lang(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        labels, probs = self._ft_model().predict(text.replace("\n", " "), k=k)
        out = []
        for lab, p in zip(labels, probs):
            lang = lab.split("__label__")[-1]
            out.append((lang, float(p)))
        return out

    @staticmethod
    def _lang_primary_from_preds(preds: List[Tuple[str, float]]) -> str:
        if not preds:
            return "other"
        cand = preds[0][0]
        return cand if cand in {"kk", "ru", "en"} else "other"

    def _detect_mixed_kk_ru(self, text: str, lang_primary: str) -> Tuple[bool, float]:
        toks = self._sample_tokens(self._tokenize_simple(text), self.cfg.max_tokens_for_per_token_lid)
        shares = self._per_token_lang_shares(toks)
        if lang_primary == "kk":
            secondary = shares.get("ru", 0.0)
        elif lang_primary == "ru":
            secondary = shares.get("kk", 0.0)
        else:
            secondary = min(shares.get("kk", 0.0), shares.get("ru", 0.0))
        bonus = 0.05 if (self._has_kk_chars(text) and self._has_ru_chars(text)) else 0.0
        is_mixed = (secondary + bonus) >= self.cfg.mix_min_secondary_share
        mix_ratio = max(0.0, min(1.0, secondary + bonus))
        return is_mixed, mix_ratio

    @staticmethod
    def _tokenize_simple(text: str) -> List[str]:
        return RE_WORD.findall((text or "").lower())

    @staticmethod
    def _sample_tokens(tokens: List[str], max_n: int) -> List[str]:
        if len(tokens) <= max_n:
            return tokens
        step = math.ceil(len(tokens) / max_n)
        return tokens[::step][:max_n]

    def _per_token_lang_shares(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {"kk": 0.0, "ru": 0.0, "other": 1.0}
        counts = {"kk": 0, "ru": 0, "other": 0}
        for tok in tokens:
            if len(tok) < 2:
                continue
            labels, probs = self._ft_model().predict(tok, k=1)
            lang = labels[0].split("__label__")[-1]
            prob = float(probs[0])
            if prob < self.cfg.mix_min_conf:
                counts["other"] += 1
            elif lang in ("kk", "ru"):
                counts[lang] += 1
            else:
                counts["other"] += 1
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}


    def _ensure_emoji_polarity_loaded(self):
        global _emoji_polarity
        if _emoji_polarity:
            self._emoji_polarity = _emoji_polarity
            return
        csv_path = self.cfg.emoji_sentiment_csv
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            e_col = "emoji" if "emoji" in df.columns else df.columns[0]
            s_col = "sentiment" if "sentiment" in df.columns else df.columns[1]
            _emoji_polarity = {str(r[e_col]): float(r[s_col]) for _, r in df.iterrows()}
        else:
            _emoji_polarity = FALLBACK_EMOJI_POLARITY
        self._emoji_polarity = _emoji_polarity


def preprocess_df(df: pd.DataFrame, text_col: str = "text", config: Config = Config()) -> pd.DataFrame:
    """Functional wrapper if you don’t want to keep a class instance."""
    return Preprocessor(config=config).preprocess_df(df, text_col=text_col)

def flatten_our_json(obj: Union[Mapping, List[Mapping]]) -> pd.DataFrame:
    """Functional wrapper for the IG flattener."""
    return Preprocessor.flatten_our_json(obj)
