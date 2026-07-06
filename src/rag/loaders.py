from __future__ import annotations

from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {'.pdf', '.html', '.htm', '.md', '.markdown', '.txt'}


def iter_files(root: str | Path):
    root = Path(root)
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_text(path: str | Path) -> tuple[str, dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    metadata = {
        'source_path': str(path),
        'source_name': path.name,
        'doc_type': suffix.replace('.', '') or 'txt',
    }

    if suffix == '.pdf':
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ''
            if text.strip():
                pages.append(f'\n\n[page {i}]\n{text}')
        metadata['page_count'] = len(reader.pages)
        return '\n'.join(pages).strip(), metadata

    raw = path.read_text(encoding='utf-8', errors='ignore')
    if suffix in {'.html', '.htm'}:
        soup = BeautifulSoup(raw, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        return md(str(soup)).strip(), metadata

    return raw.strip(), metadata
