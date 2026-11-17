"""The `streamlx` command.

    streamlx serve <model dir> [options]       an OpenAI-compatible server
    streamlx generate <model dir> [options]    one prompt, with its figures
    streamlx validate <model dir> [options]    streaming against the model
                                               in RAM, token for token

The model may also be given as --model. Each command's own --help lists its
options (`streamlx serve` adds mlx_lm.server's).
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {"serve": "streamlx.serve", "generate": "streamlx.generate",
            "validate": "streamlx.validate"}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__.strip(), file=sys.stderr)
        raise SystemExit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    command = sys.argv.pop(1)
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sys.argv[1:2] = ["--model", sys.argv[1]]
    sys.argv[0] = f"streamlx {command}"
    raise SystemExit(importlib.import_module(COMMANDS[command]).main())


if __name__ == "__main__":
    main()
