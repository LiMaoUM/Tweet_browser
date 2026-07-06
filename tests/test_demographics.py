import os

import numpy as np
import pandas as pd
import pytest

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "Frontend")


def test_precomputed_provider_returns_columns():
    import demographics

    df = pd.DataFrame({c: ["a", "b"] for c in demographics.DEMOGRAPHIC_COLUMNS})
    df["Message"] = ["x", "y"]
    result = demographics.PrecomputedProvider().infer(df)
    assert list(result.columns) == demographics.DEMOGRAPHIC_COLUMNS
    assert len(result) == 2


def test_precomputed_provider_raises_on_missing_columns():
    import demographics

    df = pd.DataFrame({"Message": ["x"], "sex": ["f"]})
    with pytest.raises(ValueError) as exc:
        demographics.PrecomputedProvider().infer(df)
    assert "age" in str(exc.value)


def test_distributions_counts_and_unknowns():
    import demographics

    attrs = pd.DataFrame({"sex": ["f", "f", "m", np.nan]})
    dists = demographics.distributions(attrs)
    assert dists["sex"]["f"] == 2
    assert dists["sex"]["m"] == 1
    assert dists["sex"]["unknown"] == 1


def test_show_results_escapes_html(monkeypatch):
    # custom_widgets' anywidget classes resolve their JS/CSS assets relative
    # to cwd, matching how the app runs in production (cwd = Frontend/).
    monkeypatch.chdir(FRONTEND_DIR)
    import ui_modules

    m = ui_modules.InferDemographicsModule()
    m.showResults({"sex": {"<script>": 1}}, 1)
    assert "<script>" not in m.resultsHTML.value
    assert "&lt;script&gt;" in m.resultsHTML.value
