from __future__ import annotations

import os
import re
import glob
import json
import time
import math
import torch
import hashlib

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass(frozen=True)
class RAGConfig:
    kb_folder: str = "knowledge_base"
    index_cache_path: str = "data/kb_index.pkl"
    embed_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Generation models
    gen_model_id: str = "issai/LLama-3.1-KazLLM-1.0-8B"
    gen_fallback_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Chunking
    chunk_chars: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    # Budgets
    max_context_chars: int = 1600
    max_new_tokens: int = 220
    temperature: float = 0.2
    # Behavior
    skip_spam: bool = True
    soften_on_toxic: bool = True

class KBIndex:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.embed = None
        self.embeddings: Optional[np.ndarray] = None
        self.texts: List[str] = []
        self.meta: List[Dict] = []
        self._loaded = False

    def _maybe_load_embed(self):
        if self.embed is None:
            self.embed = SentenceTransformer(self.cfg.embed_model_id)

    def _fingerprint_files(self, files: List[str]) -> str:
        h = hashlib.sha1()
        for f in sorted(files):
            st = os.stat(f)
            h.update(f.encode("utf-8"))
            h.update(str(int(st.st_mtime)).encode("utf-8"))
            h.update(str(int(st.st_size)).encode("utf-8"))
        return h.hexdigest()

    def _chunk_text(self, txt: str, path: str) -> List[Tuple[str, Dict]]:
        chunks: List[Tuple[str, Dict]] = []
        t = (txt or "").strip()
        if not t:
            return chunks
        cc = self.cfg.chunk_chars
        ov = self.cfg.chunk_overlap
        start = 0
        idx = 0
        while start < len(t):
            end = min(len(t), start + cc)
            chunk = t[start:end]
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if chunk:
                chunks.append((chunk, {"source": path, "chunk_id": idx}))
                idx += 1
            if end == len(t):
                break
            start = max(start + cc - ov, end)
        return chunks

    def _load_kb_files(self) -> List[str]:
        files = glob.glob(os.path.join(self.cfg.kb_folder, "**", "*.txt"), recursive=True)
        return [f for f in files if os.path.getsize(f) > 0]

    def _save_cache(self, fp: str, data: Dict):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        import pickle
        with open(fp, "wb") as f:
            pickle.dump(data, f)

    def _load_cache(self, fp: str) -> Optional[Dict]:
        if not os.path.exists(fp):
            return None
        import pickle
        with open(fp, "rb") as f:
            return pickle.load(f)

    def build_or_load(self):
        files = self._load_kb_files()
        fp = self._fingerprint_files(files)
        cached = self._load_cache(self.cfg.index_cache_path)
        if cached and cached.get("fp") == fp and cached.get("embed_model_id") == self.cfg.embed_model_id:
            self.embeddings = cached["embeddings"]
            self.texts = cached["texts"]
            self.meta = cached["meta"]
            self._loaded = True
            return

        # (Re)build
        self._maybe_load_embed()
        texts, meta = [], []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read()
            except Exception:
                continue
            for chunk, m in self._chunk_text(txt, path):
                texts.append(chunk)
                meta.append(m)

        if not texts:
            # empty KB placeholder
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.texts, self.meta = [], []
            self._loaded = True
            return

        vecs = self.embed.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        self.embeddings = np.asarray(vecs, dtype=np.float32)
        self.texts = texts
        self.meta = meta
        self._loaded = True

        self._save_cache(self.cfg.index_cache_path, {
            "fp": fp,
            "embed_model_id": self.cfg.embed_model_id,
            "embeddings": self.embeddings,
            "texts": self.texts,
            "meta": self.meta,
        })

    def search(self, query: str, top_k: int = None) -> List[Tuple[float, str, Dict]]:
        if not self._loaded:
            self.build_or_load()
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        self._maybe_load_embed()
        top_k = top_k or self.cfg.top_k
        qv = self.embed.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.embeddings @ qv  # cosine because normalized
        if top_k >= len(sims):
            idxs = np.argsort(-sims)
        else:
            idxs = np.argpartition(-sims, top_k)[:top_k]
            idxs = idxs[np.argsort(-sims[idxs])]
        out = [(float(sims[i]), self.texts[i], self.meta[i]) for i in idxs]
        return out

class KeynoteExtractor:
    """
    Deterministic, kk/ru/mix aware. Returns canonical tags.
    """
    def __init__(self):
        # Russian patterns
        self.ru = [
            (r"\b(не\s*работает|не\s*ловит|нет\s*сети|связь\s*пропада|обрыв|нет\s*интернета|потеря\s*сигнала)\b", "no_coverage_or_drop"),
            (r"\b(медленн|низк(ая|ая)\s*скорост|скорость\s*низкая|тормозит|лагает)\b", "slow_speed"),
            (r"\b(тариф|дорог|цены|стоимост|абонентск(ая|ий)\s*плата|подорожал|подорожание)\b", "tariff_price"),
            (r"\b(списал[аи]?|пропал[аи]?|неверн(ый|ая)\s*баланс|платеж\s*не\s*прошел|чек|возврат)\b", "billing_balance"),
            (r"\b(активаци|подключени|сим-карт|esim|e-sim|регистраци)\b", "sim_activation"),
            (r"\b(роуминг|международн|путешеств|за границей)\b", "roaming"),
            (r"\b(приложени[ея]|личный\s*кабинет|app|телеграм-бот|бот)\b", "app_support"),
            (r"\b(служба\s*поддержк|оператор|чат|горячая\s*линия)\b", "contact_support"),
        ]
        # Kazakh patterns
        self.kk = [
            (r"\b(желі\s*жоқ|қ pokрытие|қамту\s*жоқ|байланыс\s*жоқ|үзіл(е|іп)|интернет\s*жоқ)\b", "no_coverage_or_drop"),
            (r"\b(жылдамдық\s*төмен|тез\s*емес|баяу|ақырын|лаги|қат)\b", "slow_speed"),
            (r"\b(тариф|қымбат|баға|абонентт(ік)?\s*төлем|қымбаттап|қымбаттады)\b", "tariff_price"),
            (r"\b(баланс\s*қате|толықпады|төлем\s*өтпеді|ақша\s*алып қойды|қайтарым)\b", "billing_balance"),
            (r"\b(sim|сим|esim|e-sim|тіркеу|қосылу|активация)\b", "sim_activation"),
            (r"\b(роуминг|шетелде|сапарда)\b", "roaming"),
            (r"\b(қосымша|приложение|жеке\s*кабинет|бот)\b", "app_support"),
            (r"\b(қолдау\s*қызметі|оператор|чаты|жедел\s*желі)\b", "contact_support"),
        ]
        # Fallback single-word hints
        self.generic = [
            (r"\b(tariff|price|expensive)\b", "tariff_price"),
            (r"\b(slow|lag|speed)\b", "slow_speed"),
            (r"\b(balance|billing|payment)\b", "billing_balance"),
        ]

    def extract(self, text: str, lang: str) -> List[str]:
        if not text:
            return []
        t = text.lower()
        tags: List[str] = []

        def add(patterns):
            nonlocal tags
            for pat, tag in patterns:
                if re.search(pat, t, flags=re.UNICODE):
                    tags.append(tag)

        if lang == "ru":
            add(self.ru)
        elif lang == "kk":
            add(self.kk)
        else:
            add(self.ru); add(self.kk)

        if not tags:
            add(self.generic)

        # Deduplicate but keep order
        seen = set()
        out = []
        for x in tags:
            if x not in seen:
                seen.add(x)
                out.append(x)
        # Keep only a few keynotes (top 3)
        return out[:3]

def choose_reply_lang(lang_primary: str, has_kk: Optional[bool], has_ru: Optional[bool]) -> str:
    if lang_primary == "kk":
        return "kk"
    if lang_primary == "ru":
        return "ru"
    if lang_primary in ("mix", "other", "en"):
        if has_kk:
            return "kk"
        if has_ru:
            return "ru"
        return "ru"
    return "ru"

def choose_style(row) -> str:
    t = str(row.get("type", "other"))
    sent = str(row.get("sentiment", "neutral"))
    toxic = str(row.get("toxicity_label", "non_toxic")) == "toxic"

    if toxic:
        return "apologetic"  # soften
    if t == "complaint":
        return "apologetic"
    if t == "question":
        return "informative"
    if t == "thanks":
        return "friendly"
    if t == "review":
        return "friendly" if sent != "negative" else "apologetic"
    return "professional"

RU_STYLE_PROMPTS = {
    "professional": "Ответь профессионально и официально на комментарий: '{query}'\n{context}\n\nПрофессиональный ответ:",
    "friendly":     "Ответь дружелюбно и тепло на комментарий: '{query}'\n{context}\n\nДружелюбный ответ:",
    "supportive":   "Ответь поддерживающе и с пониманием на комментарий: '{query}'\n{context}\n\nПоддерживающий ответ:",
    "informative":  "Дай информативный ответ с фактами на комментарий: '{query}'\n{context}\n\nИнформативный ответ:",
    "apologetic":   "Ответь с извинениями и готовностью помочь на комментарий: '{query}'\n{context}\n\nОтвет с извинениями:",
}
KK_STYLE_PROMPTS = {
    "professional": "Пікірге кәсіби және ресми жауап бер: '{query}'\n{context}\n\nКәсіби жауап:",
    "friendly":     "Пікірге достық әрі жылы жауап бер: '{query}'\n{context}\n\nДостық жауап:",
    "supportive":   "Пікірге қолдау көрсетіп, түсіністікпен жауап бер: '{query}'\n{context}\n\nҚолдаушы жауап:",
    "informative":  "Пікірге нақты деректермен ақпараттық жауап бер: '{query}'\n{context}\n\nАқпараттық жауап:",
    "apologetic":   "Пікірге кешірім сұрап, көмектесуге дайын екеніңді білдіріп жауап бер: '{query}'\n{context}\n\nКешірім сұрайтын жауап:",
}

RU_SYSTEM = (
    "Правила: используй только факты из Контекста. Если фактов недостаточно — задай один короткий уточняющий вопрос "
    "или предложи официальный канал поддержки. Не выдумывай цены/номера, если их нет в Контексте. "
    "Пиши кратко: 1–3 предложения."
)
KK_SYSTEM = (
    "Ережелер: жауапты тек Контекстегі деректерге сүйеніп құр. Егер дерек жеткіліксіз болса — бір қысқа нақтылау "
    "сұрағын қой немесе ресми қолдау арнасына жүгінуді ұсын. Контексте жоқ баға/нөмірді ойдан шығарма. "
    "Қысқа жаз: 1–3 сөйлем."
)

def build_context_text(chunks: List[Tuple[float, str, Dict]], lang: str, max_chars: int) -> Tuple[str, List[str]]:
    if not chunks:
        # No KB available
        if lang == "kk":
            header = "Контекст: (мәлімет табылмады)\n"
        else:
            header = "Контекст: (факты не найдены)\n"
        return header, []
    # Join top chunks up to budget
    lines = []
    sources = []
    total = 0
    for score, txt, meta in chunks:
        piece = f"- {txt.strip()}"
        if total + len(piece) + 1 > max_chars:
            break
        lines.append(piece)
        sources.append(meta.get("source", ""))
        total += len(piece) + 1
    if lang == "kk":
        header = "Контекст (фактілер):\n"
        tail = f"\n\n{KK_SYSTEM}"
    else:
        header = "Контекст (факты):\n"
        tail = f"\n\n{RU_SYSTEM}"
    return header + "\n".join(lines) + tail, sources

class KazLLM:
    def __init__(self, model_id: str, fallback_id: str, device: str):
        self.device = device
        self.model_id = model_id
        self.fallback_id = fallback_id
        self.tok = None
        self.model = None
        self.used_id = None

    def _load(self, prefer_fallback: bool = False):
        if self.model is not None and self.tok is not None:
            return
        mid = self.fallback_id if prefer_fallback else self.model_id
        try:
            self.tok = AutoTokenizer.from_pretrained(mid)
            self.model = AutoModelForCausalLM.from_pretrained(
                mid,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.used_id = mid
        except Exception:
            # fallback if primary failed
            if mid != self.fallback_id:
                self.tok = AutoTokenizer.from_pretrained(self.fallback_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.fallback_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self.used_id = self.fallback_id
            else:
                raise

    def generate(self, prompt: str, max_new_tokens: int = 220, temperature: float = 0.2) -> str:
        self._load()
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                repetition_penalty=1.1,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        # try to cut the prompt echo
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

@dataclass
class GenerateConfig:
    rag: RAGConfig = field(default_factory=RAGConfig)

class RAGGenerator:
    def __init__(self, config: GenerateConfig = GenerateConfig()):
        self.cfg = config
        self.kb = KBIndex(config.rag)
        self.llm = KazLLM(
            model_id=config.rag.gen_model_id,
            fallback_id=config.rag.gen_fallback_id,
            device=config.rag.device,
        )
        self.keynotes = KeynoteExtractor()

    def generate_df(self, df: pd.DataFrame,
                    text_col: str = "text_norm",
                    lang_col: str = "lang_primary") -> pd.DataFrame:
        if text_col not in df.columns:
            raise KeyError(f"Input DataFrame is missing '{text_col}'")
        if lang_col not in df.columns:
            raise KeyError(f"Input DataFrame is missing '{lang_col}'")

        self.kb.build_or_load()

        out = df.copy()
        reply_langs: List[str] = []
        styles: List[str] = []
        keynotes_list: List[str] = []
        kb_sources_list: List[str] = []
        kb_ctx_list: List[str] = []
        replies: List[str] = []
        gen_model_used: List[str] = []

        has_kk = out.get("has_kk_chars")
        has_ru = out.get("has_ru_chars")

        for i, row in out.iterrows():
            txt = str(row.get(text_col, "") or "")
            lang = str(row.get(lang_col, "ru"))
            _has_kk = bool(has_kk.iloc[i]) if has_kk is not None else False
            _has_ru = bool(has_ru.iloc[i]) if has_ru is not None else True

            # Skip spam (optional)
            if self.cfg.rag.skip_spam and bool(row.get("is_spam", False)):
                reply_langs.append(choose_reply_lang(lang, _has_kk, _has_ru))
                styles.append("professional")
                keynotes_list.append(json.dumps(["skip_spam"], ensure_ascii=False))
                kb_sources_list.append(json.dumps([], ensure_ascii=False))
                kb_ctx_list.append("")
                replies.append("")
                gen_model_used.append("skipped_spam")
                continue

            # Determine reply language and style
            rlang = choose_reply_lang(lang, _has_kk, _has_ru)
            style = choose_style(row)

            # Extract keynotes
            notes = self.keynotes.extract(txt, rlang)
            keynotes_list.append(json.dumps(notes, ensure_ascii=False))

            # Retrieval: build a compact query
            # Prefer keynotes + original text
            q = " ; ".join(notes) + " ; " + txt if notes else txt
            hits = self.kb.search(q, top_k=self.cfg.rag.top_k)
            ctx_text, srcs = build_context_text(hits, rlang, self.cfg.rag.max_context_chars)
            kb_sources_list.append(json.dumps(srcs, ensure_ascii=False))
            kb_ctx_list.append(ctx_text)

            # Compose final prompt (bilingual style templates)
            if rlang == "kk":
                template = KK_STYLE_PROMPTS.get(style, KK_STYLE_PROMPTS["informative"])
            else:
                template = RU_STYLE_PROMPTS.get(style, RU_STYLE_PROMPTS["informative"])

            prompt = template.format(query=txt, context=ctx_text).strip()

            # If toxic, prepend an empathy line instruction
            if self.cfg.rag.soften_on_toxic and str(row.get("toxicity_label", "non_toxic")) == "toxic":
                if rlang == "kk":
                    prompt = ("Эмпатия қос: агрессияға жауап бермей, сабырлы түрде көмек ұсын.\n" + prompt)
                else:
                    prompt = ("Добавь эмпатию: не отвечай на агрессию, спокойно предложи помощь.\n" + prompt)

            # Generate
            reply = self.llm.generate(
                prompt,
                max_new_tokens=self.cfg.rag.max_new_tokens,
                temperature=self.cfg.rag.temperature,
            )

            reply_langs.append(rlang)
            styles.append(style)
            replies.append(reply)
            gen_model_used.append(self.llm.used_id or self.cfg.rag.gen_model_id)

        out["reply_lang"] = reply_langs
        out["reply_style"] = styles
        out["keynotes"] = keynotes_list
        out["kb_sources"] = kb_sources_list
        out["kb_context"] = kb_ctx_list
        out["reply_text"] = replies
        out["gen_model"] = gen_model_used
        return out


def generate_answers_df(df: pd.DataFrame,
                        text_col: str = "text_norm",
                        lang_col: str = "lang_primary",
                        config: GenerateConfig = GenerateConfig()) -> pd.DataFrame:
    return RAGGenerator(config=config).generate_df(df, text_col=text_col, lang_col=lang_col)
