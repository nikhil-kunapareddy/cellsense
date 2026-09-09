"""Permission policy: decide whether a side-effecting tool call may run.

Pure and side-effect free by design. This module never prompts, never writes
anything, and never calls ``interrupt()`` -- ``graph.nodes`` is the only place
that actually pauses the turn. Keeping the *decision* here, separate from the
*mechanism*, is what makes ``decide()`` trivially unit-testable: build a
policy, call it, assert the verdict, no LangGraph runtime required.

Three modes, mirroring ``cellsense.config.PermissionsConfig.mode``:

* ``"prompt"`` -- ask, unless the tool is on the static allow-list or was
  already granted "allow always" earlier in this session.
* ``"allow"``  -- never ask; every side-effecting tool runs immediately.
* ``"deny"``   -- never ask; every side-effecting tool is refused outright.

Tools without ``side_effect=True`` are never gated: :meth:`PermissionPolicy.decide`
returns ``"allow"`` for those in every mode, without even looking at ``mode``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from cellsense.tools.base import ToolSpec

__all__ = ["Decision", "PermissionMode", "PermissionPolicy"]

PermissionMode = Literal["prompt", "allow", "deny"]
Decision = Literal["ask", "allow", "deny"]


@dataclass
class PermissionPolicy:
    """Decides, and remembers, whether a side-effecting tool call may run.

    ``allow_list`` is the static, config-supplied set of tool names that never
    need a prompt (``Config.permissions.allow``). ``session_allow`` starts
    empty and only grows when the caller records an "allow always" decision
    via :meth:`remember` -- kept separate from ``allow_list`` so a session-scoped
    grant is never mistaken for, or written back into, the user's config file.
    """

    mode: PermissionMode = "prompt"
    allow_list: frozenset[str] = field(default_factory=frozenset)
    session_allow: set[str] = field(default_factory=set)

    def decide(self, tool_spec: ToolSpec, args: dict[str, Any]) -> Decision:
        """Return the verdict for one prospective call to ``tool_spec``.

        ``args`` is accepted (and currently unused) so a future policy can make
        argument-sensitive decisions -- e.g. always allow ``plot`` but ask
        before a ``join`` across more than two files -- without changing every
        call site's signature.
        """
        del args  # reserved for future argument-sensitive policies
        if not tool_spec.side_effect:
            return "allow"
        if self.mode == "allow":
            return "allow"
        if self.mode == "deny":
            return "deny"
        if tool_spec.name in self.allow_list or tool_spec.name in self.session_allow:
            return "allow"
        return "ask"

    def remember(self, tool_name: str) -> None:
        """Record an "allow always" decision for the rest of this session."""
        self.session_allow.add(tool_name)
