"""Curriculum helpers shared by the warm-started tasks (steady-cam, C5, C3x).

Kept in its own module so steady-cam and the follow-ups can both import it
without importing each other (that cycle silently skipped registering C5 and
C3x when steady-cam was imported first).
"""

from __future__ import annotations


def _collapse_curricula_to_final(cfg) -> None:
    """Warm start: pin every inherited velocity curriculum at its final stage.

    Same rule as the vendored velstand task: the loaded policy was trained
    under the final stage of each curriculum, so restarting them at stage 0
    would suddenly make its world easier (and its regularizers weaker).
    """
    for term in cfg.curriculum.values():
        for key, val in list(term.params.items()):
            if isinstance(val, list) and val and all(isinstance(v, dict) and "step" in v for v in val):
                final = {**val[-1], "step": 0}
                term.params[key] = [final]
                if key == "weight_stages" and term.params.get("reward_name") in cfg.rewards:
                    cfg.rewards[term.params["reward_name"]].weight = final["weight"]
