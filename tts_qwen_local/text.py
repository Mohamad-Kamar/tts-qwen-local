from __future__ import annotations

import re

WHITESPACE_RE = re.compile(r"[ \t]+")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")
SENTENCE_RE = re.compile(r"(?<=[.!?;:。！？])\s+")
CLAUSE_RE = re.compile(r"(?<=[,，、;；])\s+")

ABBREVIATIONS: dict[str, str] = {
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Prof.": "Professor",
    "Sr.": "Senior",
    "Jr.": "Junior",
    "vs.": "versus",
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "et cetera",
    "approx.": "approximately",
}

_ABBREV_RE = re.compile("|".join(re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True)))


def normalize_text(text: str) -> str:
    lines = [WHITESPACE_RE.sub(" ", line.strip()) for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def preprocess_text(text: str) -> str:
    result = _ABBREV_RE.sub(lambda m: ABBREVIATIONS[m.group(0)], text)
    result = re.sub(r"\s*[—–]\s*", ", ", result)
    result = re.sub(r"\s*\(([^)]+)\)\s*", r", \1, ", result)
    result = re.sub(r",\s*,", ",", result)
    result = re.sub(r",\s*\.", ".", result)
    result = re.sub(r"\s+,", ",", result)
    result = re.sub(r"[^\S\n]{2,}", " ", result)
    return result.strip()


def chunk_text(text: str, max_chars: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    preprocessed = preprocess_text(normalized)
    paragraphs = [part.strip() for part in PARAGRAPH_RE.split(preprocessed) if part.strip()]
    pieces: list[tuple[str, bool]] = []

    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph_chunks = _chunk_paragraph(paragraph, max_chars)
        for chunk_index, piece in enumerate(paragraph_chunks):
            pieces.append((piece, paragraph_index > 0 and chunk_index == 0))

    return _merge_tiny_tail(_pack_pieces(pieces, max_chars=max_chars), max_chars=max_chars)


def _chunk_paragraph(paragraph: str, max_chars: int) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_RE.split(paragraph) if part.strip()]
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        for piece in _split_oversized(sentence, max_chars):
            candidate = piece if not current else f"{current} {piece}"
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate

    if current:
        chunks.append(current)

    return chunks


def _pack_pieces(pieces: list[tuple[str, bool]], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for piece, starts_new_paragraph in pieces:
        if not current:
            current = piece
            continue

        separator = "\n\n" if starts_new_paragraph else " "
        candidate = f"{current}{separator}{piece}"
        if len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    pieces = [text]
    for splitter in (CLAUSE_RE, None):
        next_pieces: list[str] = []
        for piece in pieces:
            if len(piece) <= max_chars:
                next_pieces.append(piece)
                continue
            next_pieces.extend(_split_piece(piece, max_chars=max_chars, splitter=splitter))
        pieces = next_pieces
    return pieces


def _split_piece(text: str, max_chars: int, splitter: re.Pattern[str] | None) -> list[str]:
    parts = text.split() if splitter is None else [part.strip() for part in splitter.split(text)]
    parts = [part for part in parts if part]
    if not parts:
        return [text]

    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_token(part, max_chars=max_chars))
            continue

        candidate = part if not current else f"{current} {part}"
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _split_long_token(text: str, max_chars: int) -> list[str]:
    return [text[index:index + max_chars] for index in range(0, len(text), max_chars)]


def _merge_tiny_tail(chunks: list[str], max_chars: int) -> list[str]:
    if len(chunks) < 2:
        return chunks

    merged = list(chunks)
    tail = merged[-1]
    if len(tail) >= 80:
        return merged

    candidate = f"{merged[-2]} {tail}"
    if len(candidate) <= max_chars:
        merged[-2] = candidate
        merged.pop()
    return merged
