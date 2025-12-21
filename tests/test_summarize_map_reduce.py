from __future__ import annotations

from dataclasses import dataclass

from insights.llm import ChatMessage
from insights.summarize import MAP_SYSTEM, REDUCE_SYSTEM, SUMMARY_SYSTEM, _LLM, _clean_paragraph, chunk_text, map_reduce_summary


def test_clean_paragraph_collapses_and_strips_bullets() -> None:
    raw = "- first line\ncontinued\n- second\n  wrapped too\n- third"
    out = _clean_paragraph(raw)
    assert out == "first line continued second wrapped too third"


def test_chunk_text_raw_slicing_overlap() -> None:
    # Force raw slicing path by providing a single paragraph (no double newlines).
    text = "".join(chr(ord("a") + (i % 26)) for i in range(0, 200))
    chunks = chunk_text(text, chunk_chars=50, overlap_chars=10)
    assert chunks
    # Consecutive chunks should overlap by exactly overlap_chars under raw slicing.
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a[-10:] == b[:10]
    # Ensure coverage: the last character of the original text appears somewhere.
    assert text[-1] in chunks[-1]


def test_chunk_text_paragraphs_covers_all_markers() -> None:
    # Paragraph-aware chunking should not drop content.
    paras = [f"PARA{i} " + ("x" * 50) for i in range(10)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, chunk_chars=120, overlap_chars=40)
    assert len(chunks) >= 2
    for i in range(10):
        assert any(f"PARA{i}" in c for c in chunks)


@dataclass
class _Recorder:
    calls: list[str]

    def generate(self, messages: list[ChatMessage], used_model: str, max_tokens: int) -> str:
        sys = messages[0].content
        user = messages[1].content
        if sys == MAP_SYSTEM:
            # Return a short paragraph per chunk with its index.
            prefix = "Chunk "
            i = int(user[user.index(prefix) + len(prefix) :].split("/", 1)[0])
            self.calls.append(f"map:{i}")
            return f"MAP{i} paragraph."
        if sys == REDUCE_SYSTEM:
            # Reduce reports count.
            n = user.count("MAP")
            self.calls.append(f"reduce:{n}")
            return f"REDUCED{n} paragraph."
        if sys == SUMMARY_SYSTEM:
            self.calls.append("single")
            return "SINGLE paragraph."
        raise AssertionError("unexpected system prompt")


def test_map_reduce_single_chunk_uses_single_pass() -> None:
    rec = _Recorder(calls=[])
    llm = _LLM(provider="anthropic", model="stub", generate=rec.generate)
    out = map_reduce_summary(content="short text", llm=llm, chunk_chars=10_000, overlap_chars=0, reduce_batch_size=10)
    assert out == "SINGLE paragraph."
    assert rec.calls == ["single"]


def test_map_reduce_hierarchical_reduction_batches() -> None:
    # Force multiple chunks by small chunk size and force multiple reduction rounds via small batch size.
    text = "\n\n".join([f"PARA{i} " + ("y" * 100) for i in range(8)])
    rec = _Recorder(calls=[])
    llm = _LLM(provider="anthropic", model="stub", generate=rec.generate)
    out = map_reduce_summary(content=text, llm=llm, chunk_chars=120, overlap_chars=0, reduce_batch_size=2)
    assert out.startswith("REDUCED")
    # Should have mapped at least 3 chunks and reduced more than once.
    assert len([c for c in rec.calls if c.startswith("map:")]) >= 3
    assert len([c for c in rec.calls if c.startswith("reduce:")]) >= 2


def test_map_reduce_emits_progress_messages() -> None:
    text = "\n\n".join([f"PARA{i} " + ("z" * 200) for i in range(6)])
    rec = _Recorder(calls=[])
    llm = _LLM(provider="anthropic", model="stub", generate=rec.generate)

    progress: list[str] = []

    def on_progress(msg: str) -> None:
        progress.append(msg)

    out = map_reduce_summary(
        content=text,
        llm=llm,
        chunk_chars=120,
        overlap_chars=0,
        reduce_batch_size=2,
        progress=on_progress,
        progress_every_chunks=2,
    )
    assert out
    # Should include at least one map and one reduce progress message.
    assert any("summary map:" in m for m in progress)
    assert any("summary reduce:" in m for m in progress)

