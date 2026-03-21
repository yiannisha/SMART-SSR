import argparse
from pathlib import Path
from typing import Dict, List

import jsonlines
from transformers import AutoTokenizer


def read_jsonl(path: Path) -> List[Dict]:
    with jsonlines.open(path, "r") as reader:
        return [row for row in reader]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(rows)


def parse_outputs(record: Dict) -> List[Dict[str, str]]:
    definition = record["inputs"].split("\n\nInput:")[0]
    parsed_rows: List[Dict[str, str]] = []

    for block in record["outputs"].split("Input:"):
        block = block.strip()
        if not block or "Output:" not in block:
            continue
        input_part, output_part = block.split("Output:", 1)
        candidate_input = input_part.strip()
        candidate_output = output_part.strip()
        if not candidate_input or not candidate_output:
            continue
        parsed_rows.append(
            {
                "inputs": f"{definition}\n\n{candidate_input}",
                "outputs": candidate_output
            }
        )

    return parsed_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--tokenizer_name_or_path", required=True)
    parser.add_argument(
        "--simcse_model_name_or_path",
        default="princeton-nlp/sup-simcse-roberta-base"
    )
    parser.add_argument("--max_instruction_input_len_used", type=int, default=500)
    parser.add_argument("--max_simcse_raw_input_len_used", type=int, default=510)
    parser.add_argument("--max_input_len_used", type=int, default=800)
    parser.add_argument("--max_target_len_used", type=int, default=128)
    args = parser.parse_args()

    raw_rows = read_jsonl(Path(args.input_path))
    gen_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    simcse_tokenizer = AutoTokenizer.from_pretrained(args.simcse_model_name_or_path)

    filtered_rows: List[Dict] = []
    seen_pairs = set()

    for row in raw_rows:
        definition = row["inputs"].split("\n\nInput:")[0]
        instruction_len = len(gen_tokenizer.encode(definition))

        for parsed in parse_outputs(row):
            input_text = parsed["inputs"].split("\n\n", 1)[1] if "\n\n" in parsed["inputs"] else parsed["inputs"]
            output_text = parsed["outputs"]

            input_len = len(gen_tokenizer.encode(input_text))
            output_len = len(gen_tokenizer.encode(output_text))
            simcse_len = len(simcse_tokenizer.encode(input_text))

            if input_len <= 1 or output_len <= 1:
                continue
            if instruction_len + input_len > args.max_instruction_input_len_used:
                continue
            if input_len > args.max_input_len_used or output_len > args.max_target_len_used:
                continue
            if simcse_len > args.max_simcse_raw_input_len_used:
                continue

            dedupe_key = (parsed["inputs"].strip(), parsed["outputs"].strip())
            if dedupe_key in seen_pairs:
                continue
            seen_pairs.add(dedupe_key)
            filtered_rows.append(parsed)

    write_jsonl(Path(args.output_path), filtered_rows)
    print(f"Saved {len(filtered_rows)} parsed rows to {args.output_path}.")


if __name__ == "__main__":
    main()
