import os
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any


_WORD_RE = re.compile(r"\w+")
_STOPWORDS = {
    'the','is','are','a','an','of','and','to','in','on','for','with','as','by','at','from','or','that','this','it','be','was','were','what','who','which','did','does','do','she','he','they','her','his','their','about','tell','me','you','we'
}


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower()) if text else []


def _score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    # Simple TF-based score with length normalization
    tf = 0
    doc_freqs: Dict[str, int] = {}
    for t in doc_tokens:
        doc_freqs[t] = doc_freqs.get(t, 0) + 1
    for qt in query_tokens:
        tf += doc_freqs.get(qt, 0)
    return tf / math.sqrt(len(doc_tokens))


def _levenshtein(a: str, b: str, max_distance: int = 2) -> int:
    """Compute Levenshtein distance with early exit when distance exceeds max_distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_distance:
        return max_distance + 1
    # Ensure a is shorter
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        cur = [j] + [0] * la
        bj = b[j - 1]
        min_row = cur[0]
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == bj else 1
            cur[i] = min(
                prev[i] + 1,        # deletion
                cur[i - 1] + 1,     # insertion
                prev[i - 1] + cost  # substitution
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[i] = min(cur[i], prev[i - 2] + 1)  # transposition
            if cur[i] < min_row:
                min_row = cur[i]
        if min_row > max_distance:
            return max_distance + 1
        prev = cur
    return prev[-1]


def _has_fuzzy_match(query_token: str, doc_tokens: List[str]) -> bool:
    q = query_token
    if len(q) <= 2:
        return False
    for t in doc_tokens:
        if not t:
            continue
        if t == q:
            return True
        # Allow small typos (distance <= 2)
        if _levenshtein(q, t, max_distance=2) <= 2:
            return True
    return False
def _split_into_chunks(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    # Prefer paragraph boundaries
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + "\n\n" + p) if current else p
        else:
            if current:
                chunks.append(current)
            # If single paragraph is too big, hard-slice it
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(len(p), start + max_chars)
                    chunks.append(p[start:end])
                    start = max(start + max_chars - overlap, end)
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks


def _extractive_answer(query: str, docs: List[Dict[str, Any]], max_lines: int = 6) -> str:
    qtokens = [t for t in _tokenize(query) if t not in _STOPWORDS and len(t) > 2]
    if not qtokens:
        return ""
    matches: List[str] = []
    for doc in docs:
        snippet = doc.get("snippet", "")
        if not snippet:
            continue
        lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            ln_low = ln.lower()
            if any(qt in ln_low for qt in qtokens):
                # include a bit of local context
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                chunk = "\n".join(lines[start:end])
                if chunk not in matches:
                    matches.append(chunk)
                if len(matches) >= max_lines:
                    break
            else:
                # fuzzy token check per line
                line_tokens = _tokenize(ln_low)
                matched = False
                for qt in qtokens:
                    if _has_fuzzy_match(qt, line_tokens):
                        start = max(0, i - 1)
                        end = min(len(lines), i + 2)
                        chunk = "\n".join(lines[start:end])
                        if chunk not in matches:
                            matches.append(chunk)
                        matched = True
                        break
                if matched and len(matches) >= max_lines:
                    break
        if len(matches) >= max_lines:
            break
    return "\n\n".join(matches[:max_lines])


def _iter_scraped_json_files(data_dir: str) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        return []
    files: List[Path] = []
    # Include top-level JSON files
    files.extend(root.glob("*.json"))
    # Include nested JSON files
    for sub in root.rglob("*.json"):
        if sub not in files:
            files.append(sub)
    return files


def search_data_dir(query: str, k: int = 5) -> List[Dict[str, Any]]:
    data_dir = os.getenv("DATA_DIR", "./scraped_pages")
    query_tokens = _tokenize(query)
    query_token_set = set(query_tokens)
    candidates: List[Dict[str, Any]] = []
    for fp in _iter_scraped_json_files(data_dir):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            content = data.get("content") or data.get("markdown") or ""
            title = data.get("title", "")
            url = data.get("url", "")
            # Chunk content to improve recall on specific sections like Work Experience
            chunks = _split_into_chunks(content, max_chars=1200, overlap=150)
            if not chunks:
                chunks = [content]
            title_url_tokens = _tokenize(title) + _tokenize(url)
            title_url_set = set(title_url_tokens)
            for chunk in chunks:
                content_tokens = _tokenize(chunk)
                tokens = content_tokens + title_url_tokens
                base = _score(query_tokens, tokens)
                # Boost for title/url match (e.g., uploaded resume identifiers)
                overlap = len(query_token_set.intersection(title_url_set))
                # Small fuzzy bonus for minor typos (e.g., achivements -> achievements)
                fuzzy_bonus = 0.0
                for qt in query_tokens:
                    if _has_fuzzy_match(qt, tokens):
                        fuzzy_bonus += 0.5
                s = base + (5.0 * overlap) + fuzzy_bonus
                if s > 0:
                    snippet = chunk[:1200]
                    candidates.append({
                        "path": str(fp),
                        "url": url,
                        "title": title,
                        "score": s,
                        "snippet": snippet,
                    })
        except Exception:
            continue
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]


def lookup_kb_minimal(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Minimal KB lookup: retrieve top-k JSON documents by simple lexical score.
    If OPENAI_API_KEY is set, generate a grounded answer using the snippets; otherwise, return extractive summary.
    """
    top = search_data_dir(query, k=k)
    if not top:
        return {
            "answer": "No relevant information found in the local knowledge base.",
            "citations": [],
            "confidence": 0.0,
        }

    citations = []
    for doc in top:
        citations.append({
            "url": doc.get("url") or doc.get("path"),
            "title": doc.get("title", ""),
        })

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Extractive fallback: concatenate snippets
        joined = _extractive_answer(query, top) or "\n\n".join(doc["snippet"] for doc in top)
        return {
            "answer": joined[:1200],
            "citations": citations,
            "confidence": 0.4,
        }

    # Generate grounded answer with OpenAI
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        context_parts = []
        for idx, doc in enumerate(top, 1):
            src = doc.get("url") or doc.get("path")
            context_parts.append(f"[{idx}] Source: {src}\n{doc['snippet']}")
        context = "\n\n".join(context_parts)
        user_prompt = (
            "Answer the question using only the context below. Be concise and cite sources like [1], [2].\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n"
        )
        response = client.chat.completions.create(
            model=os.getenv("RAG_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You answer strictly from context and always cite sources [n]."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        answer = response.choices[0].message.content
        # If model fails to answer, provide extractive lines to ensure a useful reply
        if not answer or len(answer.strip()) < 20 or ("not" in answer.lower() and "context" in answer.lower()):
            extracted = _extractive_answer(query, top)
            if extracted:
                answer = extracted[:1200]
        return {
            "answer": answer,
            "citations": citations,
            "confidence": 0.7,
        }
    except Exception:
        joined = "\n\n".join(doc["snippet"] for doc in top)
        return {
            "answer": joined[:1200],
            "citations": citations,
            "confidence": 0.4,
        }


