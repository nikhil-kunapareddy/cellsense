"""Pins cellsense.permissions.PermissionPolicy: every mode, allow-list hits,
and the "allow always" mutation.
"""

from __future__ import annotations

from cellsense.permissions import PermissionPolicy
from cellsense.tools.base import tool_spec


def _spec(name: str, *, side_effect: bool) -> object:
    return tool_spec(
        name=name, description="d", parameters={"type": "object"}, side_effect=side_effect
    )(lambda args, ws: None)


PLOT = _spec("plot", side_effect=True)
DESCRIBE = _spec("describe", side_effect=False)


class TestNonSideEffectTools:
    def test_always_allowed_in_prompt_mode(self) -> None:
        policy = PermissionPolicy(mode="prompt")
        assert policy.decide(DESCRIBE, {}) == "allow"

    def test_always_allowed_in_allow_mode(self) -> None:
        policy = PermissionPolicy(mode="allow")
        assert policy.decide(DESCRIBE, {}) == "allow"

    def test_always_allowed_even_in_deny_mode(self) -> None:
        policy = PermissionPolicy(mode="deny")
        assert policy.decide(DESCRIBE, {}) == "allow"


class TestAllowMode:
    def test_side_effect_tool_is_allowed_without_being_allow_listed(self) -> None:
        policy = PermissionPolicy(mode="allow", allow_list=frozenset())
        assert policy.decide(PLOT, {}) == "allow"


class TestDenyMode:
    def test_side_effect_tool_is_denied(self) -> None:
        policy = PermissionPolicy(mode="deny")
        assert policy.decide(PLOT, {}) == "deny"

    def test_deny_mode_overrides_the_static_allow_list(self) -> None:
        policy = PermissionPolicy(mode="deny", allow_list=frozenset({"plot"}))
        assert policy.decide(PLOT, {}) == "deny"


class TestPromptMode:
    def test_tool_not_on_any_allow_list_is_asked_about(self) -> None:
        policy = PermissionPolicy(mode="prompt")
        assert policy.decide(PLOT, {}) == "ask"

    def test_tool_on_the_static_allow_list_is_allowed(self) -> None:
        policy = PermissionPolicy(mode="prompt", allow_list=frozenset({"plot"}))
        assert policy.decide(PLOT, {}) == "allow"

    def test_tool_on_the_session_allow_list_is_allowed(self) -> None:
        policy = PermissionPolicy(mode="prompt")
        policy.session_allow.add("plot")
        assert policy.decide(PLOT, {}) == "allow"


class TestRemember:
    def test_remember_adds_to_session_allow(self) -> None:
        policy = PermissionPolicy(mode="prompt")
        policy.remember("plot")
        assert "plot" in policy.session_allow

    def test_remember_then_decide_allows_subsequent_calls(self) -> None:
        policy = PermissionPolicy(mode="prompt")
        assert policy.decide(PLOT, {}) == "ask"
        policy.remember("plot")
        assert policy.decide(PLOT, {}) == "allow"

    def test_remember_does_not_affect_the_static_allow_list(self) -> None:
        policy = PermissionPolicy(mode="prompt")
        policy.remember("plot")
        assert "plot" not in policy.allow_list

    def test_remember_is_scoped_to_one_policy_instance(self) -> None:
        a = PermissionPolicy(mode="prompt")
        b = PermissionPolicy(mode="prompt")
        a.remember("plot")
        assert b.decide(PLOT, {}) == "ask"


def test_default_mode_is_prompt() -> None:
    assert PermissionPolicy().mode == "prompt"


def test_default_allow_list_and_session_allow_are_empty() -> None:
    policy = PermissionPolicy()
    assert policy.allow_list == frozenset()
    assert policy.session_allow == set()
