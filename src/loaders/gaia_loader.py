"""
src/loaders/gaia_loader.py
────────────────────────────────────────────────────────────────
Loader for GAIA benchmark (gaia-benchmark/GAIA).

GAIA has 3 annotated difficulty levels:
  Level 1 → simple, mostly single-step
  Level 2 → multi-step, tool use required
  Level 3 → very hard, multi-hop, complex tool use

This makes GAIA uniquely valuable: Level is a ground-truth
complexity label we can use to VALIDATE our estimators.

Canonical columns produced:
  id       → task_id
  query    → Question
  answer   → Final answer

Extra columns kept:
  level (1-3), annotator_steps, tools_required, file_name
"""

from __future__ import annotations

import logging

import pandas as pd
from datasets import load_dataset

from .base import BaseLoader

logger = logging.getLogger(__name__)


class GAIALoader(BaseLoader):

    DATASET_NAME = "gaia"

    def _download(self) -> pd.DataFrame:
        hf_repo = self.config.get("hf_repo", "gaia-benchmark/GAIA")
        split = self.config.get("split", "validation")
        hf_config = self.config.get("hf_config", "2023_all")

        logger.info(f"  HF repo: {hf_repo}  config: {hf_config}  split: {split}")
        ds = load_dataset(hf_repo, hf_config, split=split)

        return ds.to_pandas()

    def _process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        # ── Rename to canonical columns ───────────────────────────
        # GAIA column names vary slightly across versions
        col_map = {}
        for src, dst in [
            ("task_id", "id"),
            ("Question", "query"),
            ("Final answer", "answer"),
            ("Level", "level"),
        ]:
            if src in df.columns:
                col_map[src] = dst
        df = df.rename(columns=col_map)

        # ── Parse annotator metadata ──────────────────────────────
        if "Annotator Metadata" in df.columns:
            # Metadata is a dict with keys like "Steps", "Tools"
            def safe_get(meta, key, default=None):
                try:
                    if isinstance(meta, dict):
                        return meta.get(key, default)
                    return default
                except Exception:
                    return default

            df["annotator_steps"] = df["Annotator Metadata"].apply(
                lambda m: safe_get(m, "Steps")
            )
            df["annotator_tools"] = df["Annotator Metadata"].apply(
                lambda m: safe_get(m, "Tools")
            )
            df = df.drop(columns=["Annotator Metadata"])

        # ── Normalize level to complexity_gt (0.0 – 1.0) ─────────
        # Level 1 → 0.2, Level 2 → 0.5, Level 3 → 0.9
        # level_map = {1: 0.2, 2: 0.5, 3: 0.9}
        # if "level" in df.columns:
        #     df["complexity_gt"] = df["level"].map(level_map)
        #     logger.info("  Added 'complexity_gt' from GAIA levels (ground truth)")

        # ── Keep only needed columns ──────────────────────────────
        keep = [
            "id",
            "query",
            "answer",
            "level",
            # "complexity_gt",
            "annotator_steps",
            "annotator_tools",
            "file_name",
        ]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        # ── Drop unanswerable rows ────────────────────────────────
        before = len(df)
        df = df[df["query"].notna() & (df["query"].str.strip() != "")]
        logger.info(f"  Dropped {before - len(df)} rows with empty query")

        df = self._apply_max_samples(df)
        df = df.reset_index(drop=True)
        return df

    def _extra_verify(self, df: pd.DataFrame) -> bool:
        ok = True
        if "level" in df.columns:
            dist = df["level"].value_counts().sort_index().to_dict()
            print(f"  Levels   : {dist}  (1=easy → 3=hard)")
        if "complexity_gt" in df.columns:
            print(
                f"  GT range : {df['complexity_gt'].min():.2f} – "
                f"{df['complexity_gt'].max():.2f}"
            )
        return ok
