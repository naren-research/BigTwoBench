# BigTwoBench

This repository contains the paper-facing materials for **Prompting as a Strong Baseline for Big Two Persona Alignment in Small Open-Weight LLMs**. BigTwoBench is a diagnostic evaluation framework for higher-order persona control over Stability and Plasticity. It compares five matched intervention conditions: Prompt, Standard LoRA, LoRA+Prompt, LoRA+Ctrl, and LoRA+Infer.

## Contents

- `prompts/`: the four Big Two target prompts used in the manuscript.
- `configs/`: the five-condition design, the eight evaluated base models, fixed training recipe, evaluation settings and randomization seeds, equal-budget ablation, and LoRA sensitivity grid.
- `code/`: subset selection, questionnaire evaluation, and behavioral evaluation scripts.
- `behavioral_probes/`: the six reported deterministic behavioral probes.
- `questionnaires/`: instrument identifiers and instructions; third-party item text is not redistributed.
- `data/JIC_SOURCE.md`: where to obtain the upstream JIC source corpus.

## Data

The upstream JIC corpus is not redistributed here. See [data/JIC_SOURCE.md](data/JIC_SOURCE.md) for the official dataset release and accompanying project. This repository also excludes raw derived subsets, raw completions, checkpoints, and logs.

## Use

Install the small Python dependency set with `pip install -r requirements.txt`. Supply a scored dialogue table to `code/construct_big_two_subsets.py` to reconstruct the four manuscript target subsets. For model evaluation, copy and complete `configs/run_matrix.template.csv`, provide questionnaire definitions, then use `code/questionnaire_eval.py` or `code/behavior_eval.py` against an OpenAI-compatible local inference endpoint. The exact study design is in `configs/experiment.json`.
