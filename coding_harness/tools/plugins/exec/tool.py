"""Built-in ``exec`` plugin: shell, Python, and tests.

Thin adapter over :func:`coding_harness.tools.exec.build_exec_tools`. The
sandbox settings (timeout, byte budget, deny-list, dry-run) come from the
read-only :class:`~coding_harness.tools.context.ToolConfigView`, so these tools
run under exactly the v1 confinement.
"""

from __future__ import annotations

from coding_harness.tools.context import ToolContext
from coding_harness.tools.exec import build_exec_tools


def build(ctx: ToolContext):
    return build_exec_tools(
        ctx.jail,
        command_timeout_s=ctx.config.command_timeout_s,
        max_output_bytes=ctx.config.max_output_bytes,
        deny_patterns=list(ctx.config.deny_patterns),
        dry_run=ctx.config.dry_run,
    )
