"""Graph end-to-end behaviour, driven through the real ``Engine`` against a
scripted chat model (see ``tests/integration/conftest.py``'s ``make_engine``
for why ``STREAM_MODES`` is patched to a list by default).

Covers ARCHITECTURE.md SS6's turn lifecycle: the single-subtask path bypassing
``synthesize``, the multi-subtask ``Send`` fan-out merging results, a
``ToolError``/``DataError`` degrading to a retryable ``ToolMessage`` rather
than failing the turn, the ``max_tool_rounds`` cap forcing a final answer,
the guardrail rejecting off-topic questions (and failing open when its own
call raises), and cancellation mid-turn. Every scenario asserts the hard
contract: exactly one ``TurnFinished`` or ``TurnFailed``.
"""

from __future__ import annotations

import threading

from cellsense.config import AgentConfig, Config
from cellsense.events import (
    Notice,
    PlanCreated,
    ToolFailed,
    ToolFinished,
    ToolStarted,
    TurnFailed,
    TurnFinished,
)
from tests.fixtures.fake_chat_model import (
    ai_text,
    ai_tool_call,
    has_tool_message,
    is_forced_final_call,
    is_guardrail_call,
    is_planner_call,
    is_synthesize_call,
)


def _agent_script_answers_directly(final_text: str):
    """A script for guardrail(yes) -> planner(no decompose) -> agent(answers,
    no tool calls) -- the simplest possible single-subtask turn.
    """

    def script(messages):
        if is_guardrail_call(messages):
            return ai_text("yes")
        if is_planner_call(messages):
            return ai_text('[{"id": "t1", "question": "q"}]')
        return ai_text(final_text)

    return script


class TestSingleSubtaskPath:
    def test_bypasses_synthesize_and_finishes_normally(
        self, make_chat_model, make_engine, make_workspace, sales_df, assert_one_terminal_event
    ) -> None:
        ws = make_workspace(sales=sales_df)
        model = make_chat_model(_agent_script_answers_directly("All good."))
        engine = make_engine(ws, model)

        events = list(engine.stream_turn("hi", thread_id="t1", on_approval=lambda _r: "deny"))

        terminal = assert_one_terminal_event(events)
        assert isinstance(terminal, TurnFinished)
        assert terminal.answer == "All good."
        # PlanCreated is emitted (planner always runs), but nothing routes
        # through a "synthesize" node -- there is no way to observe
        # "synthesize" node_started at all for a single-subtask plan.
        plans = [e for e in events if isinstance(e, PlanCreated)]
        assert len(plans) == 1
        assert plans[0].subtasks == [("t1", "q")]
        node_names = {e.node for e in events if hasattr(e, "node")}
        assert "synthesize" not in node_names
        assert "worker" not in node_names


class TestMultiSubtaskFanout:
    def test_fanout_merges_results_and_synthesizes(
        self, make_chat_model, make_engine, make_workspace, sales_df, assert_one_terminal_event
    ) -> None:
        def script(messages):
            if is_guardrail_call(messages):
                return ai_text("yes")
            if is_planner_call(messages):
                return ai_text(
                    '[{"id": "t1", "question": "total revenue?"}, '
                    '{"id": "t2", "question": "total units?"}]'
                )
            if is_synthesize_call(messages):
                return ai_text("Combined: revenue and units both summarized.")
            # Every worker gets exactly one round: answer immediately, no tools.
            question = str(messages[-1].content)
            return ai_text(f"Answer for: {question}")

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(script)
        engine = make_engine(ws, model)

        events = list(
            engine.stream_turn(
                "total revenue and total units?", thread_id="t2", on_approval=lambda _r: "deny"
            )
        )

        terminal = assert_one_terminal_event(events)
        assert isinstance(terminal, TurnFinished)
        assert terminal.answer == "Combined: revenue and units both summarized."

        node_names = [e.node for e in events if hasattr(e, "node")]
        assert node_names.count("worker") == 2
        assert "synthesize" in node_names
        # The direct single-subtask "agent" node never runs on the fan-out path.
        assert "agent" not in node_names

    def test_worker_transcripts_never_pollute_the_shared_history(
        self, make_chat_model, make_engine, make_workspace, sales_df
    ) -> None:
        """WorkerState's own docstring: worker messages are local Python state
        and must never leak into Engine.history()'s top-level transcript.
        """

        def script(messages):
            if is_guardrail_call(messages):
                return ai_text("yes")
            if is_planner_call(messages):
                return ai_text('[{"id": "t1", "question": "q1"}, {"id": "t2", "question": "q2"}]')
            if is_synthesize_call(messages):
                return ai_text("Final combined answer.")
            return ai_text("SECRET_WORKER_CHATTER should not leak")

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(script)
        engine = make_engine(ws, model)

        list(engine.stream_turn("q1 and q2", thread_id="t3", on_approval=lambda _r: "deny"))

        history_text = " ".join(text for _role, text in engine.history("t3"))
        assert "SECRET_WORKER_CHATTER" not in history_text
        assert "Final combined answer." in history_text


class TestToolFailureIsRecoverable:
    def test_tool_error_becomes_a_retryable_tool_message_not_a_turn_failure(
        self, make_chat_model, make_engine, make_workspace, sales_df, assert_one_terminal_event
    ) -> None:
        """aggregate() on an unknown column raises ToolError; the graph must
        feed it back as a ToolMessage and let the model retry, not fail the turn.
        """
        attempts = {"n": 0}

        def script(messages):
            if is_guardrail_call(messages):
                return ai_text("yes")
            if is_planner_call(messages):
                return ai_text('[{"id": "t1", "question": "q"}]')
            if has_tool_message(messages):
                return ai_text("Recovered after the error.")
            attempts["n"] += 1
            return ai_tool_call(
                "aggregate", {"aggregations": {"not_a_real_column": "sum"}}, call_id="bad-col"
            )

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(script)
        engine = make_engine(ws, model)

        events = list(
            engine.stream_turn(
                "total of a bad column", thread_id="t4", on_approval=lambda _r: "deny"
            )
        )

        terminal = assert_one_terminal_event(events)
        assert isinstance(terminal, TurnFinished)
        assert terminal.answer == "Recovered after the error."

        failures = [e for e in events if isinstance(e, ToolFailed)]
        assert len(failures) == 1
        assert "not_a_real_column" in failures[0].error
        assert attempts["n"] == 1

    def test_data_error_from_workspace_resolve_becomes_a_retryable_tool_message(
        self, make_chat_model, make_engine, make_workspace, sales_df, assert_one_terminal_event
    ) -> None:
        """A tool call naming a file/sheet that doesn't exist raises DataError
        from Workspace.resolve() -- a previously-real bug class where only
        ToolError was handled and DataError slipped through as a turn
        failure. ``nodes._run_tool_calls`` now catches CellSenseError broadly
        (both DataError and ToolError are subclasses), so this must recover.
        """

        def script(messages):
            if is_guardrail_call(messages):
                return ai_text("yes")
            if is_planner_call(messages):
                return ai_text('[{"id": "t1", "question": "q"}]')
            if has_tool_message(messages):
                return ai_text("Recovered after the missing-file error.")
            return ai_tool_call(
                "aggregate",
                {"filename": "does_not_exist.csv", "aggregations": {"revenue": "sum"}},
                call_id="bad-file",
            )

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(script)
        engine = make_engine(ws, model)

        events = list(
            engine.stream_turn(
                "total revenue in a file that doesn't exist",
                thread_id="t5",
                on_approval=lambda _r: "deny",
            )
        )

        terminal = assert_one_terminal_event(events)
        assert isinstance(terminal, TurnFinished)
        assert terminal.answer == "Recovered after the missing-file error."

        failures = [e for e in events if isinstance(e, ToolFailed)]
        assert len(failures) == 1
        assert "No loaded file named" in failures[0].error


class TestMaxToolRoundsCap:
    def test_cap_forces_a_final_answer_emits_a_notice_and_does_not_crash(
        self, make_chat_model, make_engine, make_workspace, sales_df, assert_one_terminal_event
    ) -> None:
        """A model that only ever calls tools would loop forever without the
        cap. At max_tool_rounds, the graph must force a tool-free final
        answer instead of erroring or hanging -- this pins BUG 1 from
        nodes.py's module docstring: a live Groq run rejected a
        tools-unbound generation ("Tool choice is none, but model called a
        tool") when the transcript still ended on a tool result; here the
        forced-final-answer call is made to fail the same way, and the
        fallback to raw tool results must still let the turn finish cleanly.
        """

        def script(messages):
            if is_guardrail_call(messages):
                return ai_text("yes")
            if is_planner_call(messages):
                return ai_text('[{"id": "t1", "question": "q"}]')
            if is_forced_final_call(messages):
                raise RuntimeError("Tool choice is none, but model called a tool")
            # Always ask for another tool call -- never answers on its own.
            return ai_tool_call("aggregate", {"aggregations": {"revenue": "sum"}}, call_id="loop")

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(script)
        config = Config(agent=AgentConfig(max_tool_rounds=2))
        engine = make_engine(ws, model, config=config)

        events = list(
            engine.stream_turn("keep going forever", thread_id="t6", on_approval=lambda _r: "deny")
        )

        terminal = assert_one_terminal_event(events)
        assert isinstance(terminal, TurnFinished), f"cap handling crashed the turn: {terminal!r}"

        notices = [e.message for e in events if isinstance(e, Notice)]
        assert any("round" in msg and "limit" in msg for msg in notices), notices
        assert any("also failed" in msg for msg in notices), notices
        # Degraded fallback: the raw tool result text, not a real synthesis.
        assert "reached the tool-call limit" in terminal.answer.lower()


class TestGuardrail:
    def test_rejects_an_off_topic_question_without_calling_any_tool(
        self, make_chat_model, make_engine, make_workspace, sales_df, assert_one_terminal_event
    ) -> None:
        def script(messages):
            if is_guardrail_call(messages):
                return ai_text("no")
            raise AssertionError("planner/agent should never be called once rejected")

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(script)
        engine = make_engine(ws, model)

        events = list(
            engine.stream_turn(
                "what's the weather today?", thread_id="t7", on_approval=lambda _r: "deny"
            )
        )

        terminal = assert_one_terminal_event(events)
        assert isinstance(terminal, TurnFinished)
        assert "only answer questions about your loaded data" in terminal.answer.lower()
        assert terminal.citations == []
        assert not [e for e in events if isinstance(e, (ToolStarted, ToolFinished))]

    def test_fails_open_when_the_guardrail_call_itself_raises(
        self, make_chat_model, make_engine, make_workspace, sales_df, assert_one_terminal_event
    ) -> None:
        def script(messages):
            if is_guardrail_call(messages):
                raise RuntimeError("classifier backend is down")
            if is_planner_call(messages):
                return ai_text('[{"id": "t1", "question": "q"}]')
            return ai_text("Answered despite the guardrail outage.")

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(script)
        engine = make_engine(ws, model)

        events = list(
            engine.stream_turn("total revenue?", thread_id="t8", on_approval=lambda _r: "deny")
        )

        terminal = assert_one_terminal_event(events)
        assert isinstance(terminal, TurnFinished)
        assert terminal.answer == "Answered despite the guardrail outage."
        notices = [e.message for e in events if isinstance(e, Notice)]
        assert any("guardrail check failed" in msg.lower() for msg in notices), notices


class TestCancellation:
    def test_cancel_event_set_mid_turn_yields_turn_failed_cancelled(
        self, make_chat_model, make_engine, make_workspace, sales_df, assert_one_terminal_event
    ) -> None:
        cancel = threading.Event()

        def script(messages):
            if is_guardrail_call(messages):
                return ai_text("yes")
            if is_planner_call(messages):
                return ai_text('[{"id": "t1", "question": "q"}]')
            # The first (and only, since we cancel right after) agent call:
            # flip the cancel flag as a side effect of "the model responding".
            cancel.set()
            return ai_tool_call("aggregate", {"aggregations": {"revenue": "sum"}}, call_id="c1")

        ws = make_workspace(sales=sales_df)
        model = make_chat_model(script)
        engine = make_engine(ws, model)

        events = list(
            engine.stream_turn(
                "total revenue?", thread_id="t9", on_approval=lambda _r: "deny", cancel=cancel
            )
        )

        terminal = assert_one_terminal_event(events)
        assert isinstance(terminal, TurnFailed)
        assert terminal.cancelled is True
        # Cancellation happened before the tools node ever ran.
        assert not [e for e in events if isinstance(e, (ToolStarted, ToolFinished))]

        # And the checkpoint isn't left in some half-committed state that
        # breaks a later turn against the same database (a new thread_id,
        # since the cancelled turn's own thread never reached a clean state
        # to resume from -- LangGraph only commits a checkpoint once a
        # superstep finishes, per Engine's own module docstring).
        follow_up_model = make_chat_model(_agent_script_answers_directly("Follow-up OK."))
        engine2 = make_engine(ws, follow_up_model)
        follow_up_events = list(
            engine2.stream_turn("hi again", thread_id="t9-b", on_approval=lambda _r: "deny")
        )
        follow_terminal = assert_one_terminal_event(follow_up_events)
        assert isinstance(follow_terminal, TurnFinished)
