#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Chain: ChatPromptTemplate | ChatDeepSeek | JsonOutputParser
    The model only READS three raw numbers per receipt; all arithmetic
    is done in Python inside answer_queries().
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_deepseek import ChatDeepSeek

    # No max_tokens limit: this model reasons before answering, and a small
    # token cap can leave zero room for the JSON output (empty response).
    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
    )

    # NOTE: literal curly braces in the prompt MUST be escaped as {{ }}
    # because ChatPromptTemplate treats {name} as an input variable.
    system_prompt = (
        "You are a precise receipt-reading assistant. You are shown one "
        "supermarket receipt image. Read it carefully and return STRICT JSON "
        "with exactly these three keys and nothing else:\n"
        '  "final_payment": the amount actually paid — the final total AFTER '
        "the ROUNDING line (the number printed next to the payment method "
        "such as OCTOPUS / CASH / VISA, or the last grand total).\n"
        '  "subtotal": the SUBTOTAL line — after all discounts but BEFORE '
        "rounding. If no subtotal is printed, use the total before rounding.\n"
        '  "discounts": the sum of EVERY discount / promotion / coupon / '
        "member / app / packaging-damage / percentage-off line, added "
        "together as one POSITIVE number. Do NOT include the ROUNDING line "
        "here. Use 0 if there are none.\n"
        "All values must be plain numbers (no currency symbol, no commas). "
        'Example: {{"final_payment": 102.30, "subtotal": 102.31, '
        '"discounts": 5.39}}'
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Here is the receipt. Return the JSON now.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}"},
                    },
                ],
            ),
        ]
    )

    return prompt | model | JsonOutputParser()


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run the chain on every receipt and sum the two amounts.

    Per-receipt extraction runs in parallel via chain.batch; aggregation
    is deterministic Decimal arithmetic; a failing receipt is retried once
    and otherwise skipped with a warning, so the program always finishes
    and always writes results.csv.
    """

    def to_amount(value: Any) -> Decimal | None:
        """Convert one model field to a positive Decimal, or None."""
        if value is None or value == "" or value == "null":
            return None
        try:
            return Decimal(str(value).replace(",", "")).copy_abs()
        except InvalidOperation:
            return None

    inputs = [{"image_url": image_data_url(path)} for path in images]
    results = chain.batch(inputs, return_exceptions=True)

    total_paid = Decimal("0")
    total_no_discount = Decimal("0")

    for path, result in zip(images, results):
        # One retry for receipts whose parallel call failed or whose
        # output was not valid JSON.
        if isinstance(result, Exception) or not isinstance(result, dict):
            print(f"[retry] {path.name}: {result!r}")
            try:
                result = chain.invoke({"image_url": image_data_url(path)})
            except Exception as exc:
                print(f"[skip] {path.name}: {exc}")
                continue

        paid = to_amount(result.get("final_payment"))
        subtotal = to_amount(result.get("subtotal"))
        discounts = to_amount(result.get("discounts")) or Decimal("0")

        # Graceful fallbacks: never crash on a missing field.
        if paid is None and subtotal is None:
            print(f"[skip] {path.name}: unusable output {result!r}")
            continue
        if paid is None:
            paid = subtotal
        if subtotal is None:
            subtotal = paid

        print(
            f"[ok] {path.name}: paid={paid} "
            f"subtotal={subtotal} discounts={discounts}"
        )
        total_paid += paid
        # Query 2 = SUBTOTAL + discounts added back; ROUNDING is never added.
        total_no_discount += subtotal + discounts

    # Exactly one number per response, as the grader requires.
    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_no_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
