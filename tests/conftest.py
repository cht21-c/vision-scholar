import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend.config import Settings
from backend.papers import PaperLibrary
from backend.retrieval import HybridIndex
from backend.store import Store


@pytest.fixture
def pdf_bytes():
    writer = PdfWriter()
    for text in [
        "Residual learning reformulates the target mapping as F(x)+x. Identity shortcut "
        "connections add no parameters. The residual function approximates H(x)-x. "
        "The optimizer can drive weights toward zero when identity mapping is optimal.",
        "Vision transformers split an image into fixed size patches. "
        "The patch embeddings receive position embeddings and a classification token. "
        "Self attention mixes representations across patches before classification.",
    ]:
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        stream = DecodedStreamObject()
        escaped = text.replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 10 Tf 40 750 Td ({escaped}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def context(tmp_path, pdf_bytes):
    config = Settings(tmp_path, provider="mock", model="scripted-mock", run_timeout=20)
    config.prepare()
    store = Store(tmp_path / "scholar.db")
    library = PaperLibrary(store, tmp_path / "papers")
    library.ingest(pdf_bytes, paper_id="fixture", title="Engineering fixture — not a real paper")
    return config, store, library, HybridIndex(store)
