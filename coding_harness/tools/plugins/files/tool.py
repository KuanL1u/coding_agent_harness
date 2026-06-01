"""Built-in ``files`` plugin: read / write / edit / list.

A thin adapter over :func:`coding_harness.tools.files.build_file_tools` that
binds the file tools to the sandboxed jail carried on the
:class:`~coding_harness.tools.context.ToolContext`. The implementation lives in
``coding_harness/tools/files.py`` so the registration path (this plugin) stays
separate from the tool logic.
"""

from __future__ import annotations

from coding_harness.tools.context import ToolContext
from coding_harness.tools.files import build_file_tools


def build(ctx: ToolContext):
    return build_file_tools(ctx.jail, ctx.config.max_output_bytes)
