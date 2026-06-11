import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
PUBLIC_TEXT_FILES = [
    *ROOT.joinpath("src").rglob("*.py"),
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / ".env.example",
]


def test_public_files_do_not_contain_private_or_legacy_names():
    text = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_TEXT_FILES)
    forbidden = [
        r"sk-[A-Za-z0-9_-]{10,}",
        r"\bDirectAgent\b",
        r"\bdirect_agent\b",
        r"\bABLATION_LLM_MODEL\b",
        r"\bsentance\b",
        r"\bzzz\w*",
        r"_(?:ghp|lhy|qjz|zq)\b",
        r"\b(?:deepseek|qwen|claude|siliconflow|dashscope)\b",
        r"\b(?:matplotlib|seaborn|plotly)\b",
    ]
    matches = [pattern for pattern in forbidden if re.search(pattern, text, re.IGNORECASE)]
    assert matches == []


def test_source_file_names_do_not_use_legacy_prefixes_or_copy_suffixes():
    bad_names = [
        path.name
        for path in ROOT.joinpath("src").rglob("*.py")
        if path.stem.startswith("zzz")
        or re.search(r"_(?:ghp|lhy|qjz|zq)$", path.stem, re.IGNORECASE)
    ]
    assert bad_names == []
