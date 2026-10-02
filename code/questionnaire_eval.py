#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIR = Path(__file__).resolve().parents[1]
QUESTIONNAIRE_DIR = ROOT_DIR / "questionnaires"
DEFAULT_RUN_MATRIX = ROOT_DIR / "configs" / "run_matrix.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "questionnaires"

DEFAULT_TIMEOUT = 60
DEFAULT_RETRY_DELAY = 5
DEFAULT_MAX_NEW_TOKENS = 1024

QUESTIONNAIRE_ALIASES = {
    "bfm": "bfm-50",
}


@dataclass(frozen=True)
class QuestionnaireSpec:
    key: str
    path: Path


QUESTIONNAIRES = {
    "tipi": QuestionnaireSpec("tipi", QUESTIONNAIRE_DIR / "tipi.json"),
    "bfm-50": QuestionnaireSpec("bfm-50", QUESTIONNAIRE_DIR / "bfm-50.json"),
    "bfi-10": QuestionnaireSpec("bfi-10", QUESTIONNAIRE_DIR / "bfi-10.json"),
    "bfi-44": QuestionnaireSpec("bfi-44", QUESTIONNAIRE_DIR / "bfi-44.json"),
}

CANONICAL_TRAIT_MAP = {
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "conscientiousness": "Conscientiousness",
    "emotional stability": "Emotional Stability",
    "neuroticism": "Neuroticism",
    "openness": "Openness",
    "openness to experience": "Openness",
    "openness to experiences": "Openness",
    "intellect/imagination": "Openness",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def canonical_questionnaire_key(raw_key: str) -> str:
    key = raw_key.strip().lower()
    key = QUESTIONNAIRE_ALIASES.get(key, key)
    if key not in QUESTIONNAIRES and key != "all":
        raise ValueError(f"Unsupported questionnaire key: {raw_key}")
    return key


def parse_temperatures(raw: str) -> list[float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one temperature must be provided.")
    return [float(part) for part in parts]


def parse_runs_per_temperature(raw: str, temperatures: list[float]) -> dict[float, int]:
    defaults = {temperature: 1 for temperature in temperatures}
    if not raw.strip():
        return defaults

    overrides = dict(defaults)
    for entry in raw.split(","):
        piece = entry.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(
                "runs-per-temperature entries must look like '0.0=1,0.6=3'."
            )
        temp_text, count_text = piece.split("=", 1)
        overrides[float(temp_text.strip())] = int(count_text.strip())
    return overrides


def load_run_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def select_rows(
    rows: list[dict[str, str]],
    run_ids: list[str],
    method: str | None,
    model_filter: str | None,
    target_filter: str | None,
) -> list[dict[str, str]]:
    if run_ids:
        wanted = set(run_ids)
        selected = [row for row in rows if row["run_id"] in wanted]
        missing = sorted(wanted - {row["run_id"] for row in selected})
        if missing:
            raise ValueError(f"Unknown run_id values: {', '.join(missing)}")
        return selected

    selected = []
    for row in rows:
        if method and row["method"] != method:
            continue
        if model_filter and row["model"] != model_filter:
            continue
        if target_filter and row["target"] != target_filter:
            continue
        selected.append(row)
    return selected


def read_questionnaire(spec: QuestionnaireSpec) -> dict[str, Any]:
    questionnaire = load_json(spec.path)
    questionnaire["questionnaire_key"] = spec.key
    scale = questionnaire.get("scale", {})
    numeric_scale = sorted(int(key) for key in scale.keys())
    if not numeric_scale:
        raise ValueError(f"Questionnaire scale is missing numeric keys: {spec.path}")
    questionnaire["scale_min"] = numeric_scale[0]
    questionnaire["scale_max"] = numeric_scale[-1]
    questionnaire["midpoint"] = int(round((numeric_scale[0] + numeric_scale[-1]) / 2))
    questionnaire["display_name"] = questionnaire.get("name", spec.key.upper())
    return questionnaire


def canonicalize_trait_name(trait_name: str) -> str:
    key = trait_name.strip().lower()
    if key not in CANONICAL_TRAIT_MAP:
        raise ValueError(f"Unsupported trait name: {trait_name}")
    return CANONICAL_TRAIT_MAP[key]


def reverse_score(score: float, scale_min: int, scale_max: int) -> float:
    return scale_min + scale_max - score


def compute_trait_scores(
    raw_scores: dict[str, Any],
    questionnaire: dict[str, Any],
) -> dict[str, Any]:
    scale_min = int(questionnaire["scale_min"])
    scale_max = int(questionnaire["scale_max"])
    per_trait: dict[str, list[float]] = {}
    neuroticism_scores: list[float] = []

    for item in questionnaire["items"]:
        item_id = str(item["id"])
        if item_id not in raw_scores:
            continue
        score = float(raw_scores[item_id])
        if item.get("scoring") == "reverse":
            score = reverse_score(score, scale_min, scale_max)

        canonical_trait = canonicalize_trait_name(item["trait"])
        if canonical_trait == "Neuroticism":
            neuroticism_scores.append(score)
        else:
            per_trait.setdefault(canonical_trait, []).append(score)

    if neuroticism_scores:
        mean_neuroticism = sum(neuroticism_scores) / len(neuroticism_scores)
        per_trait["Emotional Stability"] = [
            reverse_score(mean_neuroticism, scale_min, scale_max)
        ]

    trait_scores: dict[str, Any] = {}
    for trait_name, values in sorted(per_trait.items()):
        trait_scores[trait_name] = round(sum(values) / len(values), 4)

    openness = trait_scores.get("Openness")
    extraversion = trait_scores.get("Extraversion")
    agreeableness = trait_scores.get("Agreeableness")
    conscientiousness = trait_scores.get("Conscientiousness")
    emotional_stability = trait_scores.get("Emotional Stability")

    if openness is not None and extraversion is not None:
        trait_scores["Plasticity"] = round((openness + extraversion) / 2, 4)
    else:
        trait_scores["Plasticity"] = None

    stability_components = [
        value
        for value in [agreeableness, conscientiousness, emotional_stability]
        if value is not None
    ]
    if len(stability_components) == 3:
        trait_scores["Stability"] = round(sum(stability_components) / 3, 4)
    else:
        trait_scores["Stability"] = None

    return trait_scores


def build_scale_text(
    questionnaire: dict[str, Any],
    option_order: list[int] | None = None,
) -> str:
    scale = questionnaire.get("scale", {})
    numeric_scale = sorted(int(key) for key in scale)
    if option_order is None:
        option_order = numeric_scale
    if sorted(option_order) != numeric_scale:
        raise ValueError("Option order must contain every scale value exactly once.")
    ordered = [(key, scale[str(key)]) for key in option_order]
    return "\n".join(f"{key} = {label}" for key, label in ordered)


def build_system_prompt(
    questionnaire: dict[str, Any],
    conditioning_text: str | None,
    allow_reasoning: bool,
    option_order: list[int] | None = None,
) -> str:
    if allow_reasoning:
        prompt = (
            "You are completing a personality self-assessment. "
            "For each statement, rate yourself on the provided scale.\n\n"
            f"{build_scale_text(questionnaire, option_order)}"
            "You may use a <think> block to reason, but you MUST end your response "
            "with exactly one integer on the provided scale."
        )
    else:
        prompt = (
            "You are completing a personality self-assessment. "
            "For each statement, respond with exactly one integer on the provided scale.\n\n"
            f"{build_scale_text(questionnaire, option_order)}"
        )
    if not conditioning_text:
        return prompt
    return (
        f"{conditioning_text.strip()}\n\n"
        "Answer the questionnaire as the assistant described above. "
        "Do not describe the assistant; rate the statements as that assistant would rate itself.\n\n"
        f"{prompt}"
    )


def build_user_message(questionnaire: dict[str, Any], item_text: str) -> str:
    prompt_prefix = questionnaire.get("prompt_prefix", "").strip()
    scale_min = questionnaire["scale_min"]
    scale_max = questionnaire["scale_max"]
    parts = []
    if prompt_prefix:
        parts.append(prompt_prefix)
    parts.append(item_text.strip())
    parts.append(f"Rating ({scale_min}-{scale_max}):")
    return "\n\n".join(parts)


def parse_integer_response(content: str, scale_min: int, scale_max: int) -> int | None:
    candidates = re.findall(r"(?<!\d)(-?\d+)(?!\d)", content)
    for candidate in reversed(candidates):
        value = int(candidate)
        if scale_min <= value <= scale_max:
            return value
    return None


def ask_item_resilient(
    base_url: str,
    server_model: str,
    system_prompt: str,
    user_message: str,
    scale_min: int,
    scale_max: int,
    temperature: float,
    attempts_per_tier: int = 3,
    temp_step: float = 0.1,
    num_tiers: int = 3,
) -> tuple[int | None, bool, str | None]:
    last_raw: str | None = None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for tier in range(num_tiers):
        current_temp = round(temperature + tier * temp_step, 6)
        payload = {
            "model": server_model,
            "messages": messages,
            "temperature": current_temp,
            "max_tokens": DEFAULT_MAX_NEW_TOKENS,
        }
        for attempt in range(1, attempts_per_tier + 1):
            try:
                response = requests.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                last_raw = content
                parsed = parse_integer_response(content, scale_min, scale_max)
                if parsed is not None:
                    return parsed, True, content
            except requests.RequestException as exc:
                last_raw = f"[http_error] {exc}"
            if attempt < attempts_per_tier:
                time.sleep(DEFAULT_RETRY_DELAY)

    return None, False, last_raw


def get_server_model(base_url: str) -> str | None:
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            return None
        return data[0]["id"]
    except Exception:
        return None


def load_conditioning_text(
    row: dict[str, str],
    force_conditioning: bool = False,
) -> str | None:
    if row["method"] != "prompt" and not force_conditioning:
        return None
    prompt_file = row.get("prompt_file", "").strip()
    if not prompt_file:
        return None
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file listed in run matrix is missing: {prompt_path}")
    return prompt_path.read_text().strip()


def evaluate_questionnaire_run(
    row: dict[str, str],
    questionnaire: dict[str, Any],
    base_url: str,
    server_model: str,
    conditioning_text: str | None,
    temperature: float,
    run_number: int,
    limit_items: int | None,
    allow_reasoning: bool,
    dry_run: bool,
    option_order_manifest: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    raw_scores: dict[str, Any] = {}
    pass_fail: dict[str, Any] = {}
    raw_outputs: dict[str, Any] = {}
    option_orders: dict[str, list[int]] = {}

    items = questionnaire["items"]
    if limit_items is not None:
        items = items[:limit_items]

    for item in items:
        item_id = str(item["id"])
        option_order = option_order_for_item(
            manifest=option_order_manifest,
            run_id=row["run_id"],
            questionnaire_key=questionnaire["questionnaire_key"],
            temperature=temperature,
            run_number=run_number,
            item_id=item_id,
        )
        if option_order is not None:
            option_orders[item_id] = option_order
        system_prompt = build_system_prompt(
            questionnaire=questionnaire,
            conditioning_text=conditioning_text,
            allow_reasoning=allow_reasoning,
            option_order=option_order,
        )
        if dry_run:
            score = questionnaire["midpoint"]
            passed = True
            raw_output = f"[dry-run midpoint {questionnaire['midpoint']}]"
        else:
            score, passed, raw_output = ask_item_resilient(
                base_url=base_url,
                server_model=server_model,
                system_prompt=system_prompt,
                user_message=build_user_message(questionnaire, item["text"]),
                scale_min=questionnaire["scale_min"],
                scale_max=questionnaire["scale_max"],
                temperature=temperature,
            )
            if not passed:
                score = questionnaire["midpoint"]
        raw_scores[item_id] = score
        pass_fail[item_id] = passed
        raw_outputs[item_id] = raw_output

    return {
        "raw_scores": raw_scores,
        "pass_fail": pass_fail,
        "raw_outputs": raw_outputs,
        "option_orders": option_orders,
    }


def option_order_key(
    run_id: str,
    questionnaire_key: str,
    temperature: float,
    run_number: int,
    item_id: str,
) -> str:
    return "|".join(
        (run_id, questionnaire_key, f"{temperature:.6f}", str(run_number), item_id)
    )


def load_option_order_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported option-order manifest: {path}")
    if manifest.get("control") != "R1-M2 balanced response-option order":
        raise ValueError(f"Unexpected option-order manifest: {path}")
    return manifest


def option_order_for_item(
    manifest: dict[str, Any] | None,
    run_id: str,
    questionnaire_key: str,
    temperature: float,
    run_number: int,
    item_id: str,
) -> list[int] | None:
    if manifest is None:
        return None
    key = option_order_key(run_id, questionnaire_key, temperature, run_number, item_id)
    try:
        return [int(value) for value in manifest["entries"][questionnaire_key][key]]
    except KeyError as exc:
        raise ValueError(
            f"Option-order manifest has no entry for {key}. Regenerate it for this run matrix."
        ) from exc


def init_results_document(
    row: dict[str, str],
    run_matrix_path: Path,
    questionnaire: dict[str, Any],
    server_model: str,
    conditioning_text: str | None,
    allow_reasoning: bool,
    option_order_manifest_path: Path | None,
    option_order_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "method": row["method"],
        "planned_model": row["model"],
        "target": row["target"],
        "target_label": row["target_label"],
        "questionnaire_key": questionnaire["questionnaire_key"],
        "questionnaire": questionnaire["display_name"],
        "questionnaire_reference": questionnaire.get("reference"),
        "scale_min": questionnaire["scale_min"],
        "scale_max": questionnaire["scale_max"],
        "server_model": server_model,
        "conditioning_applied": bool(conditioning_text),
        "conditioning_prompt_file": row.get("prompt_file") or None,
        "conditioning_prompt_text": conditioning_text,
        "allow_reasoning": allow_reasoning,
        "option_order_mode": "balanced_random" if option_order_manifest else "canonical",
        "option_order_manifest": str(option_order_manifest_path) if option_order_manifest_path else None,
        "option_order_seed": option_order_manifest.get("seed") if option_order_manifest else None,
        "dataset_jsonl": row.get("dataset_jsonl") or None,
        "train_config": row.get("train_config") or None,
        "train_script": row.get("train_script") or None,
        "run_matrix_path": str(run_matrix_path),
        "notes": row.get("notes", ""),
        "runs": [],
    }


def upsert_run(results: dict[str, Any], run_record: dict[str, Any], overwrite: bool) -> bool:
    for index, existing in enumerate(results["runs"]):
        same_key = (
            existing.get("temperature") == run_record["temperature"]
            and existing.get("run") == run_record["run"]
        )
        if not same_key:
            continue
        if not overwrite:
            return False
        results["runs"][index] = run_record
        return True
    results["runs"].append(run_record)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paper3 questionnaire evaluation for prompt-conditioned or LoRA runs."
    )
    parser.add_argument(
        "--run-matrix",
        default=str(DEFAULT_RUN_MATRIX),
        help="Path to the generated paper3 run matrix CSV.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Specific run_id to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--method",
        choices=["prompt", "lora", "lora_prompted", "lora_control"],
        help="Filter run matrix rows by method when --run-id is not provided.",
    )
    parser.add_argument(
        "--model-filter",
        help="Filter run matrix rows by planned model name when --run-id is not provided.",
    )
    parser.add_argument(
        "--target-filter",
        help="Filter run matrix rows by target name when --run-id is not provided.",
    )
    parser.add_argument(
        "--questionnaire",
        required=True,
        choices=sorted(list(QUESTIONNAIRES.keys()) + ["all", "bfm"]),
        help="Questionnaire to run. Use 'all' to run every local questionnaire asset.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="Base URL for the OpenAI-compatible endpoint serving the model.",
    )
    parser.add_argument(
        "--server-model",
        help="Model identifier exposed by the endpoint. If omitted, the script queries /models.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store result JSON files.",
    )
    parser.add_argument(
        "--option-order-manifest",
        type=Path,
        help="Use this fixed R1-M2 balanced response-option-order manifest.",
    )
    parser.add_argument(
        "--temperatures",
        default="0.0",
        help="Comma-separated temperatures, for example '0.0,0.6'.",
    )
    parser.add_argument(
        "--runs-per-temperature",
        default="",
        help="Optional overrides such as '0.0=1,0.6=3'.",
    )
    parser.add_argument(
        "--limit-items",
        type=int,
        help="Optional item limit for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing temperature/run results instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the model API; use midpoint scores instead.",
    )
    parser.add_argument(
        "--allow-reasoning",
        action="store_true",
        help="Allow models to emit <think>...</think> reasoning so long as they end with one integer.",
    )
    parser.add_argument(
        "--force-conditioning",
        action="store_true",
        help="Apply the row's prompt_file as the system conditioning prompt even for non-prompt runs.",
    )
    args = parser.parse_args()

    run_matrix_path = Path(args.run_matrix)
    output_dir = Path(args.output_dir)
    base_url = normalize_base_url(args.base_url)
    temperatures = parse_temperatures(args.temperatures)
    runs_per_temperature = parse_runs_per_temperature(args.runs_per_temperature, temperatures)
    option_order_manifest = (
        load_option_order_manifest(args.option_order_manifest)
        if args.option_order_manifest
        else None
    )

    rows = load_run_matrix(run_matrix_path)
    selected_rows = select_rows(
        rows=rows,
        run_ids=args.run_id,
        method=args.method,
        model_filter=args.model_filter,
        target_filter=args.target_filter,
    )
    if not selected_rows:
        raise SystemExit("No run matrix rows matched the requested selection.")

    questionnaire_keys = (
        sorted(QUESTIONNAIRES.keys())
        if canonical_questionnaire_key(args.questionnaire) == "all"
        else [canonical_questionnaire_key(args.questionnaire)]
    )

    if args.dry_run:
        server_model = args.server_model or "dry-run-model"
    else:
        server_model = args.server_model or get_server_model(base_url)
        if not server_model:
            raise SystemExit(
                "Could not discover server model from /models. Pass --server-model explicitly."
            )

    print(
        f"Running questionnaire evaluation for {len(selected_rows)} run(s) "
        f"against server model '{server_model}'."
    )

    for row in selected_rows:
        conditioning_text = load_conditioning_text(row, force_conditioning=args.force_conditioning)
        for questionnaire_key in questionnaire_keys:
            questionnaire = read_questionnaire(QUESTIONNAIRES[questionnaire_key])
            results_path = output_dir / questionnaire_key / f"{slugify(row['run_id'])}.json"
            if results_path.exists():
                results = load_json(results_path)
            else:
                results = init_results_document(
                    row=row,
                    run_matrix_path=run_matrix_path,
                    questionnaire=questionnaire,
                    server_model=server_model,
                    conditioning_text=conditioning_text,
                    allow_reasoning=args.allow_reasoning,
                    option_order_manifest_path=args.option_order_manifest,
                    option_order_manifest=option_order_manifest,
                )

            for temperature in temperatures:
                n_runs = int(runs_per_temperature.get(temperature, 1))
                for run_idx in range(1, n_runs + 1):
                    exists = any(
                        existing.get("temperature") == temperature and existing.get("run") == run_idx
                        for existing in results["runs"]
                    )
                    if exists and not args.overwrite:
                        print(
                            f"  Skipping {row['run_id']} | {questionnaire['display_name']} "
                            f"| T={temperature} run={run_idx} because it already exists."
                        )
                        continue

                    payload = evaluate_questionnaire_run(
                        row=row,
                        questionnaire=questionnaire,
                        base_url=base_url,
                        server_model=server_model,
                        conditioning_text=conditioning_text,
                        temperature=temperature,
                        run_number=run_idx,
                        limit_items=args.limit_items,
                        allow_reasoning=args.allow_reasoning,
                        dry_run=args.dry_run,
                        option_order_manifest=option_order_manifest,
                    )
                    trait_scores = compute_trait_scores(
                        raw_scores=payload["raw_scores"],
                        questionnaire=questionnaire,
                    )
                    n_items = len(payload["pass_fail"])
                    n_failed = sum(1 for passed in payload["pass_fail"].values() if not passed)
                    run_record = {
                        "temperature": temperature,
                        "run": run_idx,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "raw_scores": payload["raw_scores"],
                        "pass_fail": payload["pass_fail"],
                        "raw_outputs": payload["raw_outputs"],
                        "option_orders": payload["option_orders"],
                        "trait_scores": trait_scores,
                        "n_items": n_items,
                        "n_failed_items": n_failed,
                        "failure_rate": round(n_failed / n_items, 6) if n_items else None,
                    }
                    upsert_run(results, run_record, overwrite=args.overwrite)
                    write_json(results_path, results)
                    print(
                        f"  Saved {row['run_id']} | {questionnaire['display_name']} "
                        f"| T={temperature} run={run_idx} "
                        f"| Stability={trait_scores.get('Stability')} "
                        f"| Plasticity={trait_scores.get('Plasticity')}"
                    )


if __name__ == "__main__":
    main()
