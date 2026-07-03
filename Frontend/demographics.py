"""Demographics inference providers.

v1 reads the demographics columns precomputed in the dataset CSV. A future
model-backed provider implements the same interface and swaps in without UI
changes (see docs/superpowers/specs/2026-07-03-backend-service-wiring-design.md).
"""
import pandas as pd

DEMOGRAPHIC_COLUMNS = [
    "sex",
    "age",
    "education",
    "income",
    "party_identification",
    "ideology",
    "urbanicity",
    "metro",
]


class DemographicsProvider:
    """Interface: per-tweet demographic attributes for a dataframe of posts."""

    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class PrecomputedProvider(DemographicsProvider):
    """Reads demographics columns already present in the dataset CSV."""

    def infer(self, df):
        missing = [c for c in DEMOGRAPHIC_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                "Dataset lacks precomputed demographics columns: " + ", ".join(missing)
            )
        return df[DEMOGRAPHIC_COLUMNS].copy()


def distributions(attributes: pd.DataFrame) -> dict:
    """Value counts per attribute column, for the results display."""
    return {
        c: attributes[c].fillna("unknown").astype(str).value_counts().to_dict()
        for c in attributes.columns
    }
