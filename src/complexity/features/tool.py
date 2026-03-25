"""
features/tool.py
================
L3 — Tool Dependency

Identifies which external tool categories a task likely requires and
returns T ∈ [0, 1].

    T = min(category_count / MAX_CATS, 1.0)

  1 category → T = 0.25   (light tool use)
  2 categories → T = 0.50
  3 categories → T = 0.75
  4+ categories → T = 1.0  (complex agentic task)
"""

from __future__ import annotations

from typing import Tuple

from ...config import TOOL_MAX_CATS, TOOL_PATTERNS
from src.utils.helpers import clip_to_range


class ToolDependency:
    """
    Compute tool-dependency complexity for a single query.

    Parameters
    ----------
    text : str
        Raw query string.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def raw_features(self) -> dict:
        """
        Scan for each tool-category pattern and record which ones fire.

        Returns
        -------
        dict with keys:
          matched_categories : list of category names that matched
          category_count     : int — number of matched categories
          category_flags     : dict[str, bool] — one flag per category
        """
        flags = {
            cat: bool(pat.search(self.text))
            for cat, pat in TOOL_PATTERNS.items()
        }
        matched = [cat for cat, hit in flags.items() if hit]
        return {
            "matched_categories": matched,
            "category_count":     len(matched),
            "category_flags":     flags,
        }

    def score(self) -> Tuple[float, dict]:
        """
        Compute the tool-dependency score T ∈ [0, 1].

        Returns
        -------
        (T, raw_features)
        """
        raw = self.raw_features()
        T   = clip_to_range(raw["category_count"] / TOOL_MAX_CATS)
        return T, raw