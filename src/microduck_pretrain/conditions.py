"""The four training conditions compared in the experiment.

Pure metadata: importing this module never pulls in mjlab, torch, or warp, so
the CLI, the report, and the tests can use it on any machine.

    uv run md-conditions          # print the table
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Condition:
    key: str                 # short id used in file names and reports
    name: str                # human-readable name
    task_id: str | None      # mjlab task id to train; None = not trained here
    summary: str             # one line: what changes
    analog: str              # the industrial deployment practice it stands for
    details: tuple[str, ...] = field(default_factory=tuple)


VENDOR_DEFAULT = Condition(
    key="c1",
    name="Vendor default",
    task_id=None,
    summary="The official shipped walking policy, untouched.",
    analog="Robot arrives out of the box and is used as delivered.",
    details=(
        "Not trained in this repo; download it with scripts/get_vendor_policy.py.",
        "Evaluated on the same held-out site as every other condition.",
    ),
)

IDEALIZED = Condition(
    key="c2",
    name="Idealized simulation",
    task_id="Pretrain-C2-Idealized-Flat-MicroDuck",
    summary="Flat floor, every randomization switched off, ideal actuators.",
    analog="Offline programming against a perfect digital model.",
    details=(
        "All ENABLE_* domain-randomization toggles in the velocity task set False.",
        "mjlab's base random pushes and +/-2.5 cm trunk CoM offset removed.",
        "BAM actuator at fixed 7.4 V, no voltage sag, no command delay.",
        "Foot friction fixed at 1.0; observation noise removed.",
    ),
)

STANDARD = Condition(
    key="c3",
    name="Standard randomization",
    task_id="Pretrain-C3-Standard-Flat-MicroDuck",
    summary="Flat floor with Pollen's full sim-to-real recipe.",
    analog="Vendor-grade simulation with generic robustness training.",
    details=(
        "Identical to Mjlab-Velocity-Flat-MicroDuck; separate id so logs stay apart.",
        "Battery voltage, sag, delay, friction, mass, CoM, IMU and encoder DR on.",
    ),
)

SITE_SPECIFIC = Condition(
    key="c4",
    name="Site-specific pre-training",
    task_id="Pretrain-C4-Site-Rough-Backlash-MicroDuck",
    summary="Standard recipe plus rough terrain, gear backlash, payload, wider floor friction.",
    analog="Simulation tuned to the destination site before delivery.",
    details=(
        "Rough-terrain generator from the vendored Velocity-Rough task.",
        "Walk-model backlash robot: +/-1 deg gear play per servo.",
        "Payload: 0-100 g point mass added to the trunk, sampled per env.",
        "Foot friction range widened from (0.7, 1.3) to (0.5, 1.5).",
    ),
)

CONDITIONS: tuple[Condition, ...] = (VENDOR_DEFAULT, IDEALIZED, STANDARD, SITE_SPECIFIC)
TRAINABLE: tuple[Condition, ...] = tuple(c for c in CONDITIONS if c.task_id)

# Settings the site-specific condition uses; tasks.py reads them so the
# metadata above and the trained environment cannot drift apart.
SITE_PAYLOAD_KG = (0.0, 0.10)
SITE_FOOT_FRICTION = (0.5, 1.5)
IDEAL_VIN = 7.4


def by_key(key: str) -> Condition:
    for c in CONDITIONS:
        if c.key == key.lower():
            return c
    raise KeyError(f"Unknown condition {key!r}; choose from {[c.key for c in CONDITIONS]}")


def main() -> None:
    for c in CONDITIONS:
        print(f"{c.key}  {c.name}")
        print(f"    task:   {c.task_id or '(not trained here)'}")
        print(f"    what:   {c.summary}")
        print(f"    analog: {c.analog}")
        for d in c.details:
            print(f"      - {d}")
        print()


if __name__ == "__main__":
    main()
