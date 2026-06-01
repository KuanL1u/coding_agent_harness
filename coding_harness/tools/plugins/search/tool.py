"""Built-in ``search`` plugin: content grep + filename glob.

Thin adapter over :func:`coding_harness.tools.search.build_search_tools`, bound
to the sandboxed jail on the :class:`~coding_harness.tools.context.ToolContext`.
"""

from __future__ import annotations

from coding_harness.tools.context import ToolContext
from coding_harness.tools.search import build_search_tools


def build(ctx: ToolContext):
    return build_search_tools(ctx.jail, ctx.config.max_output_bytes)
