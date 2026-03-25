"""
features/task_type.py
=====================
L5 — Task Type

Classifies the structural output the query demands and maps it to a
complexity score TT ∈ [0, 1].

Priority cascade — the FIRST (highest-complexity) matching rule wins:

  generative      → 1.00   (write, design, create …)
  reasoning       → 0.80   (prove, evaluate, analyze …)
  alt_choice      → 0.50   (compare, recommend, vs …)
  single_answer   → 0.30   (what is, define, name …)
  boolean_decision → 0.15  (is it, can X be …)
  unknown         → 0.20   (fallback)
"""

from __future__ import annotations

from typing import Tuple

from ...config import TASK_TYPE_RULES, TASK_TYPE_UNKNOWN_SCORE
from src.utils.helpers import clip_to_range


class TaskType:
    """
    Classify the task type of a single query.

    Parameters
    ----------
    text : str
        Raw query string (no spaCy Doc needed — regex only).
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def raw_features(self) -> dict:
        """
        Run the priority cascade and collect all matched rules.

        Returns
        -------
        dict with keys:
          primary_type  : str   — highest-priority matched type
          primary_score : float — score of the primary type
          all_matched   : list  — all matched rules (type + score), ordered
        """
        matched = [
            {"type": name, "score": score}
            for name, score, pattern in TASK_TYPE_RULES
            if pattern.search(self.text)
        ]
        best = matched[0] if matched else {"type": "unknown", "score": TASK_TYPE_UNKNOWN_SCORE}
        return {
            "primary_type":  best["type"],
            "primary_score": best["score"],
            "all_matched":   matched,
        }

    def score(self) -> Tuple[float, dict]:
        """
        Compute the task-type score TT ∈ [0, 1].

        Returns
        -------
        (TT, raw_features)
        """
        raw = self.raw_features()
        return clip_to_range(raw["primary_score"]), raw