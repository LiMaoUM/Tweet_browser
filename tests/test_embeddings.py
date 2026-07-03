import numpy as np
import pandas as pd
import pytest
import torch


def _df(n):
    return pd.DataFrame(
        {
            "Message": [f"tweet number {i}" for i in range(n)],
            "CreatedTime": pd.date_range("2020-01-01", periods=n).astype(str),
            "State": ["michigan"] * n,
        }
    )


class FakeModel:
    """Stands in for SentenceTransformer; returns deterministic embeddings."""

    def __init__(self, dim=4):
        self.dim = dim
        self.encode_calls = 0

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        arr = np.arange(len(texts) * self.dim, dtype=np.float32).reshape(len(texts), self.dim)
        return torch.from_numpy(arr)


def test_load_when_file_matches(tmp_path, monkeypatch):
    import tweet_browser as tb

    fake = FakeModel()
    monkeypatch.setattr(tb, "get_embedding_model", lambda: fake)
    path = tmp_path / "emb.csv"
    np.savetxt(path, np.ones((3, 4), dtype=np.float32), delimiter=",")

    result = tb.load_or_compute_embeddings(_df(3), str(path))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 4)
    assert fake.encode_calls == 0  # loaded, not recomputed


def test_recompute_and_save_on_row_mismatch(tmp_path, monkeypatch):
    import tweet_browser as tb

    fake = FakeModel()
    monkeypatch.setattr(tb, "get_embedding_model", lambda: fake)
    path = tmp_path / "emb.csv"
    np.savetxt(path, np.ones((2, 4), dtype=np.float32), delimiter=",")  # wrong row count

    result = tb.load_or_compute_embeddings(_df(3), str(path))
    assert result.shape == (3, 4)
    assert fake.encode_calls == 1
    saved = np.loadtxt(path, delimiter=",")
    assert saved.shape == (3, 4)  # file was refreshed


def test_recompute_when_file_corrupt(tmp_path, monkeypatch):
    import tweet_browser as tb

    fake = FakeModel()
    monkeypatch.setattr(tb, "get_embedding_model", lambda: fake)
    path = tmp_path / "emb.csv"
    path.write_text("not,numbers,at,all\nx,y,z,w\n")

    result = tb.load_or_compute_embeddings(_df(3), str(path))
    assert result.shape == (3, 4)
    assert fake.encode_calls == 1


def test_compute_when_file_missing(tmp_path, monkeypatch):
    import tweet_browser as tb

    fake = FakeModel()
    monkeypatch.setattr(tb, "get_embedding_model", lambda: fake)
    path = tmp_path / "does_not_exist.csv"

    result = tb.load_or_compute_embeddings(_df(2), str(path))
    assert result.shape == (2, 4)
    assert path.exists()


def test_get_central_works_with_tensor_embeddings(monkeypatch):
    import tweet_browser as tb

    data = _df(4)
    embeddings = torch.eye(4)  # 4 rows, trivially normalized
    s = tb.Session(data, False, embeddings=embeddings)
    result = s.getCentral()
    assert "centrality" in result.columns
    assert len(result) == 4


def test_semantic_search_with_cpu_tensor(monkeypatch):
    import tweet_browser as tb

    data = _df(4)
    embeddings = torch.eye(4)
    s = tb.Session(data, False, embeddings=embeddings)

    class QueryModel:
        def encode(self, query, **kwargs):
            return torch.tensor([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(tb, "get_embedding_model", lambda: QueryModel())
    s.semanticSearch("anything", 0.5)
    assert s.currentSet.size == 2  # top 50% of 4
