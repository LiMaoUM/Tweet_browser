import asyncio

import pandas as pd
import pytest
import torch


def _session(n):
    import tweet_browser as tb

    data = pd.DataFrame(
        {
            "Message": [f"tweet {i}" for i in range(n)],
            "CreatedTime": pd.date_range("2020-01-01", periods=n).astype(str),
            "State": ["michigan"] * n,
        }
    )
    return tb.Session(data, False, embeddings=torch.zeros((n, 4)))


def _response_for(tweets_str, stance=1):
    """Build a valid JSON response covering exactly the ids present in the batch."""
    import re, json
    ids = [int(m) for m in re.findall(r"^(\d+)-\[", tweets_str, re.M)]
    return json.dumps({f"tweet-{i}": stance for i in ids})


def test_all_batches_succeed():
    import tweet_browser as tb

    s = _session(60)  # 2 batches of 50 + 10

    async def fake_annotation(tweets, topic, stances, examples):
        return _response_for(tweets)

    tb.stance_annotation = fake_annotation
    df = asyncio.run(s.stanceAnalysis("census", ["pro", "anti"], {}, updateAllData=True))
    assert list(df["stance"]) == [1] * 60
    assert s.lastStanceFailedBatches == 0
    assert list(s.allData["stance"])[:3] == [1, 1, 1]


def test_second_batch_fails_degrades_to_minus_one():
    import tweet_browser as tb
    from prompts import BackendError

    s = _session(60)
    calls = {"n": 0}

    async def fake_annotation(tweets, topic, stances, examples):
        calls["n"] += 1
        if calls["n"] == 2:
            raise BackendError("stance", "http://localhost:8001/v1", "boom")
        return _response_for(tweets)

    tb.stance_annotation = fake_annotation
    df = asyncio.run(s.stanceAnalysis("census", ["pro"], {}))
    assert list(df["stance"])[:50] == [1] * 50
    assert list(df["stance"])[50:] == [-1] * 10
    assert s.lastStanceFailedBatches == 1


def test_first_batch_failure_raises():
    import tweet_browser as tb
    from prompts import BackendError

    s = _session(10)

    async def fake_annotation(tweets, topic, stances, examples):
        raise BackendError("stance", "http://localhost:8001/v1", "down")

    tb.stance_annotation = fake_annotation
    with pytest.raises(BackendError):
        asyncio.run(s.stanceAnalysis("census", ["pro"], {}))


def test_unparseable_batch_counts_as_failed():
    import tweet_browser as tb

    s = _session(10)

    async def fake_annotation(tweets, topic, stances, examples):
        return "no json at all"

    tb.stance_annotation = fake_annotation
    df = asyncio.run(s.stanceAnalysis("census", ["pro"], {}))
    assert list(df["stance"]) == [-1] * 10
    assert s.lastStanceFailedBatches == 1
