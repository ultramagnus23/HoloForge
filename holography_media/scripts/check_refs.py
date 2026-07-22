r"""
Phase 7.3: check_refs.py -- fails the build while any "% TODO-VERIFY"
comment remains in paper/refs.bib. Those comments mark bibliography
entries whose exact title (or, in one case, year) was never independently
confirmed against a primary source -- only identified via search-result
or abstract text during the Table 1 citation pass. Resolving them is
explicitly the user's task (library/database access this environment
doesn't have); this script exists so that gap can never be silently
forgotten at submission time.

Usage: python scripts/check_refs.py [path/to/refs.bib]
Exit code 0 = clean (no TODO-VERIFY remaining), 1 = still pending.
"""
import os
import re
import sys

TODO_RE = re.compile(r"%\s*TODO-VERIFY", re.IGNORECASE)
ENTRY_KEY_RE = re.compile(r"@\w+\{([^,]+),")


def find_pending_entries(text: str) -> list[str]:
    """Returns the bib keys of entries preceded by a TODO-VERIFY comment
    block (the comment immediately precedes the @article{...} entry it
    documents, by the convention used when writing refs.bib)."""
    lines = text.splitlines()
    pending = []
    for i, line in enumerate(lines):
        if TODO_RE.search(line):
            # scan forward for the next @entry{key, line
            for j in range(i, min(i + 10, len(lines))):
                m = ENTRY_KEY_RE.search(lines[j])
                if m:
                    pending.append(m.group(1))
                    break
    return pending


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "paper", "refs.bib")
    if not os.path.exists(path):
        print(f"[check_refs] {path} does not exist yet -- nothing to check.")
        return 0

    with open(path, encoding="utf-8") as f:
        text = f.read()
    pending = find_pending_entries(text)

    if pending:
        print(f"[check_refs] FAILED: {len(pending)} bibliography entr"
              f"{'y' if len(pending) == 1 else 'ies'} still marked TODO-VERIFY in {path}:")
        for key in pending:
            print(f"  {key}")
        print("Resolve these from library/database access before the build passes.")
        return 1

    print(f"[check_refs] OK: no TODO-VERIFY entries remaining in {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
