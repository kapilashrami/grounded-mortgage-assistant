import pytest

from grounded_mortgage_assistant.config import (
    ROOT,
    corpus_name,
    data_dir,
    not_found_answer,
    tokenizer_path,
    unavailable_answer,
)


def test_data_dir_defaults_inside_the_repository():
    assert data_dir({}) == ROOT / "data" / "corpus"
    assert data_dir({"DATA_DIR": "/srv/corpus"}).as_posix() == "/srv/corpus"
    assert data_dir({"DATA_DIR": "relative/corpus"}).is_absolute()


def test_tokenizer_path_is_required_and_has_no_home_default():
    with pytest.raises(RuntimeError, match="LOCAL_TOKENIZER_PATH"):
        tokenizer_path({})
    assert (
        tokenizer_path({"LOCAL_TOKENIZER_PATH": "/models/qwen"}).as_posix()
        == "/models/qwen"
    )


def test_answers_name_the_corpus_only_when_configured():
    assert corpus_name({}) == ""
    assert not_found_answer("") == (
        "I could not find that in the available information. "
        "Please ask about loans, insurance, the company or its services."
    )
    assert not_found_answer("Acme") == (
        "I could not find that in the available Acme information. "
        "Please ask about Acme's loans, insurance, company or services."
    )
    assert (
        unavailable_answer("")
        == "Please ask a question about the available information."
    )
    assert (
        unavailable_answer("Acme")
        == "Please ask a question about the available Acme information."
    )
