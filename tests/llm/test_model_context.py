import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from grounded_mortgage_assistant.guardrails.privacy import OriginalProof, ingest
from grounded_mortgage_assistant.llm.conversation import interpretation_tool
from grounded_mortgage_assistant.llm.interpretation_candidates import (
    build_candidate_catalog,
)
from grounded_mortgage_assistant.llm.model_context import (
    ModelContextError,
    pack_model_context,
    token_count,
)
from grounded_mortgage_assistant.state.memory import (
    Memory,
    PendingQuestion,
    SafeExchange,
)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        assert kwargs["tools"]
        assert messages[0]["role"] == "system"
        assert all(message["role"] != "system" for message in messages[1:])
        return {
            "input_ids": [[0] * sum(len(message["content"]) for message in messages)]
        }


def test_prunes_whole_old_history_and_never_exposes_original_or_value():
    turn = ingest("My mortgage is SEK 1820000.", turn_ref="turn_server")
    state = Memory(exchanges=(SafeExchange("old", ("A",) * 9000),))
    packed, count = pack_model_context(
        state,
        turn,
        build_candidate_catalog(state, turn),
        tokenizer=Tokenizer(),
        tools=[{"type": "function"}],
        provider_capacity=262144,
    )
    body = "".join(message["content"] for message in packed.messages)
    assert not packed.history
    assert "1820000" not in body
    assert turn.protected_values[0].value_ref in body
    assert count <= 8192


def test_mandatory_overflow_fails_closed():
    turn = ingest("My mortgage is SEK 1820000.")
    with pytest.raises(ModelContextError, match="CONTEXT_BUDGET_EXCEEDED"):
        pack_model_context(
            Memory(),
            turn,
            build_candidate_catalog(Memory(), turn),
            tokenizer=Tokenizer(),
            tools=[{}],
            provider_capacity=2,
        )


def test_current_turn_is_once_and_tools_change_the_count():
    turn = ingest("Thanks.", turn_ref="current")
    state = Memory(exchanges=(SafeExchange("current", ("THANKS",)),))
    packed, _ = pack_model_context(
        state,
        turn,
        build_candidate_catalog(state, turn),
        tokenizer=Tokenizer(),
        tools=[{"name": "one"}],
        provider_capacity=262144,
    )
    body = "".join(message["content"] for message in packed.messages)
    assert body.count('"turn_ref":"current"') == 1

    class ToolAware:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": list(range(len(kwargs["tools"]) + 1))}

    assert token_count(ToolAware(), packed.messages, []) == 1
    assert token_count(ToolAware(), packed.messages, [{}, {}]) == 3


def test_multiple_tokenizer_batches_fail_closed():
    class Batches:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [[1], [2]]}

    with pytest.raises(ModelContextError, match="TOKENIZER_FAILURE"):
        token_count(Batches(), ({"role": "user", "content": "x"},), [])


def test_system_contract_expresses_safe_query_and_span_protocol():
    packed = pack_model_context(
        Memory(),
        ingest("Thanks."),
        build_candidate_catalog(Memory(), ingest("Thanks.")),
        tokenizer=Tokenizer(),
        tools=[{}],
        provider_capacity=262144,
    )[0]
    instructions = packed.messages[0]["content"]
    assert "Prior state is committed" in instructions
    assert "Candidates are uncommitted current-turn evidence" in instructions
    assert "scope candidate applies all of its concepts atomically" in instructions
    assert "Knowledge uses a nonempty array" not in instructions
    for action in (
        "acknowledge",
        "knowledge",
        "calculate",
        "review",
        "decline",
        "forget",
        "clarify",
    ):
        assert action in instructions
    assert "My mortgage is SEK" not in instructions


def test_safe_state_and_complete_history_share_the_initial_system_message():
    packed, _ = pack_model_context(
        Memory(exchanges=(SafeExchange("old", ("THANKS",), ("ACK",)),)),
        ingest("Thanks.", turn_ref="current"),
        build_candidate_catalog(Memory(), ingest("Thanks.", turn_ref="current")),
        tokenizer=Tokenizer(),
        tools=[{}],
        provider_capacity=262144,
    )
    assert [message["role"] for message in packed.messages] == ["system", "user"]
    assert "SAFE_STATE_AND_HISTORY=" in packed.messages[0]["content"]
    assert '"turn_ref":"old"' in packed.messages[0]["content"]


def test_matching_local_tokenizer_accepts_system_then_current_user_only():
    value = os.environ.get("LOCAL_TOKENIZER_PATH", "")
    path = Path(value).expanduser()
    if not value or not path.is_dir():
        pytest.skip("LOCAL_TOKENIZER_PATH does not point at the local Qwen tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    state = Memory(
        exchanges=(SafeExchange("old", ("THANKS",), ("ACK",)),),
        pending_question=PendingQuestion(
            "property_value",
            1,
            "pending_1_property_value",
            OriginalProof("old", 0, 1, "x"),
        ),
    )
    for active_state in (Memory(pending_question=state.pending_question), state):
        packed, count = pack_model_context(
            active_state,
            ingest("Thanks.", turn_ref="current"),
            build_candidate_catalog(
                active_state, ingest("Thanks.", turn_ref="current")
            ),
            tokenizer=tokenizer,
            tools=[
                interpretation_tool(
                    build_candidate_catalog(
                        active_state, ingest("Thanks.", turn_ref="current")
                    )
                )
            ],
            provider_capacity=262144,
        )
        assert [message["role"] for message in packed.messages] == ["system", "user"]
        assert count > 0
