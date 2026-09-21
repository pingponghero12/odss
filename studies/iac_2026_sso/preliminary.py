"""Launch the short SSO Monte Carlo calibration campaign."""

from __future__ import annotations

import sys
from pathlib import Path


def _main() -> int:
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from studies.iac_2026_sso.campaign import preliminary_main

    return preliminary_main()


if __name__ == "__main__":
    raise SystemExit(_main())
