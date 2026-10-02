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


ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT_DIR / "behavioral_probes"
DEFAULT_RUN_MATRIX = ROOT_DIR / "configs" / "run_matrix.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "behavioral"

DEFAULT_TIMEOUT = 60
DEFAULT_RETRY_DELAY = 5


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    path: Path


BENCHMARKS = {
    "stability_instruction_following": BenchmarkSpec(
        "stability_instruction_following",
        BENCHMARK_DIR / "stability_instruction_following.json",
    ),
    "stability_rule_maintenance": BenchmarkSpec(
        "stability_rule_maintenance",
        BENCHMARK_DIR / "stability_rule_maintenance.json",
    ),
    "stability_deescalation": BenchmarkSpec(
        "stability_deescalation",
        BENCHMARK_DIR / "stability_deescalation.json",
    ),
    "stability_boundary_repair": BenchmarkSpec(
        "stability_boundary_repair",
        BENCHMARK_DIR / "stability_boundary_repair.json",
    ),
    "plasticity_idea_diversity": BenchmarkSpec(
        "plasticity_idea_diversity",
        BENCHMARK_DIR / "plasticity_idea_diversity.json",
    ),
    "plasticity_reframing_breadth": BenchmarkSpec(
        "plasticity_reframing_breadth",
        BENCHMARK_DIR / "plasticity_reframing_breadth.json",
    ),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def canonical_benchmark_key(raw_key: str) -> str:
    key = raw_key.strip().lower()
    if key not in BENCHMARKS and key != "all":
        raise ValueError(f"Unsupported benchmark key: {raw_key}")
    return key


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


def get_server_model(base_url: str) -> str | None:
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            return None
        return str(data[0]["id"])
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


def read_benchmark(spec: BenchmarkSpec) -> dict[str, Any]:
    benchmark = load_json(spec.path)
    benchmark["benchmark_key"] = spec.key
    benchmark["display_name"] = benchmark.get("name", spec.key)
    benchmark.setdefault("default_temperature", 0.0)
    benchmark.setdefault("default_max_tokens", 512)
    return benchmark


def build_system_prompt(conditioning_text: str | None) -> str:
    base = "You are responding as a conversational assistant. Follow the user's instructions exactly."
    if not conditioning_text:
        return base
    return (
        f"{conditioning_text.strip()}\n\n"
        "Answer future prompts as the assistant described above while following the user's instructions exactly."
    )


def post_chat_completion(
    base_url: str,
    server_model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": server_model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error: str | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"]).strip()
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < 3:
                time.sleep(DEFAULT_RETRY_DELAY)
    raise RuntimeError(f"Benchmark request failed after retries: {last_error}")


def cleaned_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def count_sentences(text: str) -> int:
    segments = re.split(r"[.!?]+", text)
    return sum(1 for part in segments if part.strip())


def line_word_count(line: str) -> int:
    words = re.findall(r"\b[\w'-]+\b", line)
    return len(words)


def extract_content_segment(line: str) -> str:
    segment = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line.strip())
    segment = re.sub(r"^\[[^\]]+\]\s*", "", segment)
    return segment.strip()


def extract_bracket_tags(lines: list[str]) -> list[str]:
    tags: list[str] = []
    for line in lines:
        match = re.search(r"\[([^\]]+)\]", line)
        if match:
            tags.append(match.group(1).strip().lower())
    return tags


def first_content_words(lines: list[str]) -> list[str]:
    words: list[str] = []
    for line in lines:
        segment = extract_content_segment(line)
        match = re.search(r"\b([A-Za-z][A-Za-z'-]*)\b", segment)
        if match:
            words.append(match.group(1).lower())
    return words


def score_check(output: str, check: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    lines = cleaned_lines(output)
    check_type = str(check["type"])
    passed = False
    details: dict[str, Any] = {"type": check_type}

    if check_type == "line_count_equals":
        observed = len(lines)
        expected = int(check["expected"])
        passed = observed == expected
        details.update({"expected": expected, "observed": observed})
    elif check_type == "bullet_count_equals":
        observed = sum(1 for line in lines if re.match(r"^\s*[-*]\s+", line))
        expected = int(check["expected"])
        passed = observed == expected
        details.update({"expected": expected, "observed": observed})
    elif check_type == "numbered_count_equals":
        observed = sum(1 for line in lines if re.match(r"^\s*\d+[.)]\s+", line))
        expected = int(check["expected"])
        passed = observed == expected
        details.update({"expected": expected, "observed": observed})
    elif check_type == "each_line_word_count_equals":
        expected = int(check["expected"])
        counts = [line_word_count(line) for line in lines]
        passed = bool(lines) and all(count == expected for count in counts)
        details.update({"expected": expected, "observed": counts})
    elif check_type == "max_words_per_line":
        expected = int(check["expected"])
        counts = [line_word_count(line) for line in lines]
        passed = bool(lines) and all(count <= expected for count in counts)
        details.update({"expected_max": expected, "observed": counts})
    elif check_type == "line_contains":
        line_index = int(check["line_index"])
        substring = str(check["substring"]).lower()
        observed_line = lines[line_index] if 0 <= line_index < len(lines) else ""
        passed = substring in observed_line.lower()
        details.update({"line_index": line_index, "substring": substring, "observed": observed_line})
    elif check_type == "line_endswith":
        line_index = int(check["line_index"])
        value = str(check["value"])
        observed_line = lines[line_index] if 0 <= line_index < len(lines) else ""
        passed = observed_line.endswith(value)
        details.update({"line_index": line_index, "expected_suffix": value, "observed": observed_line})
    elif check_type == "forbidden_characters":
        chars = [str(char) for char in check["characters"]]
        present = [char for char in chars if char in output]
        passed = not present
        details.update({"forbidden": chars, "present": present})
    elif check_type == "sentence_count_equals":
        observed = count_sentences(output)
        expected = int(check["expected"])
        passed = observed == expected
        details.update({"expected": expected, "observed": observed})
    elif check_type == "max_sentences_per_line":
        expected = int(check["expected"])
        counts = [count_sentences(line) for line in lines]
        passed = bool(lines) and all(count <= expected for count in counts)
        details.update({"expected_max": expected, "observed": counts})
    elif check_type == "total_word_count_between":
        observed = line_word_count(output)
        minimum = int(check["minimum"])
        maximum = int(check["maximum"])
        passed = minimum <= observed <= maximum
        details.update({"minimum": minimum, "maximum": maximum, "observed": observed})
    elif check_type == "text_startswith":
        value = str(check["value"])
        passed = output.strip().startswith(value)
        details.update({"expected_prefix": value})
    elif check_type == "text_contains":
        value = str(check["value"]).lower()
        passed = value in output.lower()
        details.update({"expected_substring": value})
    elif check_type == "text_contains_all":
        values = [str(value).lower() for value in check["values"]]
        present = [value for value in values if value in output.lower()]
        passed = len(present) == len(values)
        details.update({"expected_substrings": values, "present": present})
    elif check_type == "text_contains_none":
        values = [str(value).lower() for value in check["values"]]
        present = [value for value in values if value in output.lower()]
        passed = not present
        details.update({"forbidden_substrings": values, "present": present})
    elif check_type == "each_line_startswith":
        value = str(check["value"])
        passed = bool(lines) and all(line.startswith(value) for line in lines)
        details.update({"expected_prefix": value, "observed": lines})
    elif check_type == "each_line_endswith":
        value = str(check["value"])
        passed = bool(lines) and all(line.endswith(value) for line in lines)
        details.update({"expected_suffix": value, "observed": lines})
    elif check_type == "distinct_bracket_tags":
        minimum = int(check["minimum"])
        tags = extract_bracket_tags(lines)
        distinct = sorted(set(tags))
        passed = len(distinct) >= minimum
        details.update({"minimum": minimum, "observed": len(distinct), "tags": distinct})
    elif check_type == "min_unique_first_content_words":
        expected = int(check["expected"])
        words = first_content_words(lines)
        distinct = sorted(set(words))
        passed = len(distinct) >= expected
        details.update({"expected": expected, "observed": len(distinct), "words": distinct})
    else:
        raise ValueError(f"Unsupported check type: {check_type}")

    details["passed"] = passed
    return passed, details


def evaluate_item(
    base_url: str,
    server_model: str,
    system_prompt: str,
    item: dict[str, Any],
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> dict[str, Any]:
    messages = item.get("messages")
    if messages is None:
        messages = [{"role": "user", "content": str(item["prompt"])}]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Item {item['id']} must provide a nonempty messages list")
    if any(not isinstance(message, dict) or message.get("role") not in {"user", "assistant"} or not isinstance(message.get("content"), str) for message in messages):
        raise ValueError(f"Item {item['id']} contains an invalid chat-history message")
    if messages[-1]["role"] != "user":
        raise ValueError(f"Item {item['id']} chat history must end with a user message")
    if dry_run:
        output = "[dry-run]"
    else:
        output = post_chat_completion(
            base_url=base_url,
            server_model=server_model,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    checks = []
    passed_checks = 0
    for check in item.get("checks", []):
        passed, details = score_check(output, check)
        if passed:
            passed_checks += 1
        checks.append(details)

    n_checks = len(checks)
    score = round(passed_checks / n_checks, 6) if n_checks else None
    return {
        "item_id": item["id"],
        "prompt": item.get("prompt") or messages[-1]["content"],
        "messages": messages,
        "output": output,
        "score": score,
        "n_checks": n_checks,
        "n_passed_checks": passed_checks,
        "checks": checks,
    }


def init_results_document(
    row: dict[str, str],
    run_matrix_path: Path,
    benchmark: dict[str, Any],
    server_model: str,
    conditioning_text: str | None,
) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "method": row["method"],
        "planned_model": row["model"],
        "target": row["target"],
        "target_label": row["target_label"],
        "benchmark_key": benchmark["benchmark_key"],
        "benchmark": benchmark["display_name"],
        "benchmark_reference": benchmark.get("reference"),
        "server_model": server_model,
        "conditioning_applied": bool(conditioning_text),
        "conditioning_prompt_file": row.get("prompt_file") or None,
        "conditioning_prompt_text": conditioning_text,
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
        description="Run custom paper3 behavioral benchmark probes against a served model."
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
        choices=["prompt", "lora", "lora_prompted", "lora_control", "lora_prompt_inference"],
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
        "--benchmark",
        required=True,
        choices=sorted(list(BENCHMARKS.keys()) + ["all"]),
        help="Benchmark to run. Use 'all' to run every local benchmark asset.",
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
        "--temperature",
        type=float,
        help="Override benchmark default temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override benchmark default max_tokens.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run results instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the model API; write placeholder outputs instead.",
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

    benchmark_keys = (
        sorted(BENCHMARKS.keys())
        if canonical_benchmark_key(args.benchmark) == "all"
        else [canonical_benchmark_key(args.benchmark)]
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
        f"Running behavioral evaluation for {len(selected_rows)} run(s) "
        f"against server model '{server_model}'."
    )

    for row in selected_rows:
        conditioning_text = load_conditioning_text(row, force_conditioning=args.force_conditioning)
        system_prompt = build_system_prompt(conditioning_text)
        for benchmark_key in benchmark_keys:
            benchmark = read_benchmark(BENCHMARKS[benchmark_key])
            results_path = output_dir / benchmark_key / f"{slugify(row['run_id'])}.json"
            if results_path.exists():
                results = load_json(results_path)
            else:
                results = init_results_document(
                    row=row,
                    run_matrix_path=run_matrix_path,
                    benchmark=benchmark,
                    server_model=server_model,
                    conditioning_text=conditioning_text,
                )

            run_record = {
                "temperature": args.temperature
                if args.temperature is not None
                else float(benchmark["default_temperature"]),
                "max_tokens": args.max_tokens
                if args.max_tokens is not None
                else int(benchmark["default_max_tokens"]),
                "run": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "items": [],
            }
            for item in benchmark["items"]:
                run_record["items"].append(
                    evaluate_item(
                        base_url=base_url,
                        server_model=server_model,
                        system_prompt=system_prompt,
                        item=item,
                        temperature=float(run_record["temperature"]),
                        max_tokens=int(run_record["max_tokens"]),
                        dry_run=args.dry_run,
                    )
                )

            scores = [
                item["score"]
                for item in run_record["items"]
                if item.get("score") is not None
            ]
            run_record["mean_item_score"] = round(sum(scores) / len(scores), 6) if scores else None
            run_record["n_items"] = len(run_record["items"])

            upsert_run(results, run_record, overwrite=args.overwrite)
            write_json(results_path, results)
            print(
                f"  Saved {row['run_id']} | {benchmark['display_name']} "
                f"| mean_item_score={run_record['mean_item_score']}"
            )


if __name__ == "__main__":
    main()
