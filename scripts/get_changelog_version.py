from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    changelog = Path("CHANGELOG.md")
    if not changelog.exists():
        print("CHANGELOG.md not found.", file=sys.stderr)
        return 1

    pattern = re.compile(r"^##\s+(v\d+\.\d+\.\d+)\s*$")
    for line in changelog.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            print(match.group(1))
            return 0

    print("No version header found in CHANGELOG.md (expected '## vX.Y.Z').", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
