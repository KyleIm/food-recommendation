#!/usr/bin/env python3
"""Convert 'Food combination.txt' examples into chat-format JSONL for Qwen fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

EXAMPLE_SPLIT_RE = re.compile(r"\bexample\s+\d+\)\s*", re.IGNORECASE)
FIELD_RE = re.compile(
    r"Appetizer:\s*(?P<app>.+?)\n"
    r"Main Dish:\s*(?P<main>.+?)\n"
    r"Drink:\s*(?P<drink>.+?)\n"
    r"Dessert:\s*(?P<dessert>.+?)\n",
    re.DOTALL,
)

SYSTEM_PROMPT = (
    "You are a professional menu-pairing evaluator. "
    "Given one appetizer, one main dish, one drink, and one dessert, "
    "write five pairwise balance scores (0-20), a total score (0-100), "
    "and a concise overall evaluation in English."
)


@dataclass
class Example:
    appetizer: str
    main_dish: str
    drink: str
    dessert: str
    assistant_answer: str


def parse_examples(raw_text: str) -> list[Example]:
    chunks = [c.strip() for c in EXAMPLE_SPLIT_RE.split(raw_text) if c.strip()]
    parsed: list[Example] = []

    for idx, chunk in enumerate(chunks, start=1):
        match = FIELD_RE.search(chunk)
        if not match:
            raise ValueError(f"Could not parse menu fields in example #{idx}.")

        # Keep answer exactly as source text to preserve desired style.
        answer = chunk.strip()
        parsed.append(
            Example(
                appetizer=match.group("app").strip(),
                main_dish=match.group("main").strip(),
                drink=match.group("drink").strip(),
                dessert=match.group("dessert").strip(),
                assistant_answer=answer,
            )
        )

    return parsed


def build_chat_record(example: Example) -> dict:
    user_prompt = (
        "Evaluate this menu combination with the required format:\n"
        f"Appetizer: {example.appetizer}\n"
        f"Main Dish: {example.main_dish}\n"
        f"Drink: {example.drink}\n"
        f"Dessert: {example.dessert}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example.assistant_answer},
        ]
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Food combination.txt", help="Source TXT path")
    parser.add_argument("--train-output", default="train.jsonl", help="Train JSONL output path")
    parser.add_argument("--val-output", default="val.jsonl", help="Validation JSONL output path")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    raw_text = input_path.read_text(encoding="utf-8")
    examples = parse_examples(raw_text)
    if len(examples) < 2:
        raise ValueError("Need at least 2 examples for train/validation split.")

    records = [build_chat_record(ex) for ex in examples]
    random.Random(args.seed).shuffle(records)

    val_size = max(1, int(len(records) * args.val_ratio))
    val_records = records[:val_size]
    train_records = records[val_size:]
    if not train_records:
        raise ValueError("Validation split too large; train set became empty.")

    write_jsonl(Path(args.train_output), train_records)
    write_jsonl(Path(args.val_output), val_records)

    print(f"Parsed examples: {len(examples)}")
    print(f"Train records : {len(train_records)} -> {args.train_output}")
    print(f"Val records   : {len(val_records)} -> {args.val_output}")


if __name__ == "__main__":
    main()
