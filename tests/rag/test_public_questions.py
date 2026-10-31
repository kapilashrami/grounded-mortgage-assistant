import pytest

from grounded_mortgage_assistant.rag.public_questions import PublicQuestionScope

is_public = PublicQuestionScope.is_public_question


@pytest.mark.parametrize(
    "text",
    [
        "Who won the football World Cup in 2022?",
        "Is 80% LTV allowed?",
        "What changed for mortgages in 2024?",
        "What is a mortgage?",
    ],
)
def test_numbers_that_are_not_amounts_keep_a_public_question(text):
    assert is_public(text)


@pytest.mark.parametrize(
    "text",
    [
        "Can I borrow 2000000?",
        "Can I borrow 500 000?",
        "Is 2,000,000 enough for a flat?",
        "Can I get a loan of 5 000 kr?",
        "Can I borrow 3m?",
        "Is 2.5 million too much?",
        "What does SEK 400 cost?",
        "Could I borrow 300k?",
    ],
)
def test_amounts_keep_a_message_off_the_public_path(text):
    assert not is_public(text)


def test_personal_ltv_and_actions_stay_off_the_public_path():
    assert not is_public("What is my LTV?")
    assert not is_public("What is my loan to value?")
    assert not is_public("What is our loan-to-value?")
    assert not is_public("Can you calculate this?")


def test_general_ltv_questions_stay_public():
    assert is_public("What is loan to value?")
    assert is_public("What is the maximum loan-to-value?")
