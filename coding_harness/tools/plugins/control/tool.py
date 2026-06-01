"""Built-in ``control`` plugin: ``task_done``.

Unlike the other built-ins, this one needs the loop's mutable
:class:`~coding_harness.tools.control.TaskState`, which the harness places on
``ctx.state``. ``task_done`` flips that flag so the agent loop knows the model
has declared the task complete. This loop handle is intentionally *not* part of
the capability surface agent-authored tools build against.
"""

from __future__ import annotations

from coding_harness.tools.context import ToolContext
from coding_harness.tools.control import build_control_tools


def build(ctx: ToolContext):
    return build_control_tools(ctx.state)
