"""Microduck pre-training experiments: deployment readiness on delivery.

Modules:
    conditions  - the four training conditions (pure metadata, no heavy imports)
    tasks       - registers the trainable conditions as mjlab tasks
    train       - `md-train`: launch a condition with a fixed, comparable budget
    evaluate    - `md-eval`: headless held-out-site evaluation of an ONNX policy
    report      - `md-report`: aggregate evaluation results, compute readiness gaps
    check_env   - `md-check`: verify the local install
"""

__version__ = "0.1.0"
