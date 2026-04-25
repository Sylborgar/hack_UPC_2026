"""Package entrypoint."""

from __future__ import annotations

import sys

from .cli import parse_args
from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        outputs = run_pipeline(args)
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    print("Pipeline completed. Outputs:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
