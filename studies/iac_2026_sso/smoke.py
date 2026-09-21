"""Launch one short, real end-to-end SSO study realization."""

from __future__ import annotations

import sys
from pathlib import Path


def _main() -> int:
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from studies.iac_2026_sso.campaign import smoke_main

    return smoke_main()


if __name__ == "__main__":
    raise SystemExit(_main())
