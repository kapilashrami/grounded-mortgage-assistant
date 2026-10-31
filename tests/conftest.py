import pytest

from grounded_mortgage_assistant.rag.retrieval import ARTIFACT


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_corpus: needs the private frozen corpus at DATA_DIR"
    )


def pytest_collection_modifyitems(config, items):
    if (ARTIFACT / "union" / "manifest.json").is_file():
        return
    skip = pytest.mark.skip(
        reason="the private frozen corpus is not available at DATA_DIR"
    )
    for item in items:
        if "requires_corpus" in item.keywords:
            item.add_marker(skip)
