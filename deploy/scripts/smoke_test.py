#!/usr/bin/env python3
"""Post-deploy smoke test: exercises both vLLM backends end to end.

Run from anywhere: python3 deploy/scripts/smoke_test.py
Exit code 0 = service is usable; 1 = something is down or broken.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Frontend")))

import prompts  # noqa: E402

SUMMARY_TWEETS = (
    "0-[The census helps allocate federal funding to communities.] "
    "1-[I filled out my census form online today, it took five minutes.] "
    "2-[Census data should be kept private and secure.]"
)
STANCE_TWEETS = (
    "0-[The census is great and everyone should participate.]\n"
    "1-[The census is a waste of taxpayer money.]\n"
    "2-[Nice weather in Michigan today.]\n"
    "3-[Just reminded my family to fill out the census.]\n"
    "4-[I do not trust the census with my data.]\n"
)


def main():
    status = prompts.check_backends()
    print("backend status:", status)
    down = [name for name, ok in status.items() if not ok]
    if down:
        print("FAIL: backend(s) unreachable:", ", ".join(down))
        return 1

    try:
        summary = prompts.ai_summarize(SUMMARY_TWEETS)
    except prompts.BackendError as e:
        print(f"FAIL: {e}")
        return 1

    if not summary or not summary.strip():
        print("FAIL: summarizer returned an empty response")
        return 1

    print("summary ok:", summary[:120].replace("\n", " "))

    try:
        raw = asyncio.run(
            prompts.stance_annotation(STANCE_TWEETS, "the US census", ["support", "oppose"], {})
        )
    except prompts.BackendError as e:
        print(f"FAIL: {e}")
        return 1

    stances = prompts.parse_stance_response(raw, 0, 5)
    print("stances:", stances)
    if all(v == -1 for v in stances.values()):
        print("FAIL: stance response produced no usable labels; raw response:")
        print(raw[:500])
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
