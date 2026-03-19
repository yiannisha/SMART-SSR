import argparse
import gc
import itertools
import jsonlines
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed


LLAMA2_PROMPT = """<s> [INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST] """


def cleanup_model(model_pipeline=None):
    if model_pipeline is not None:
        del model_pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_jsonl(path: Path) -> List[Dict]:
    with jsonlines.open(path, "r") as reader:
        return [row for row in reader]


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(list(rows))


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_generation_pipeline(model_name_or_path: str, checkpoint_dir: Optional[str] = None):
    dtype = get_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    if checkpoint_dir:
        model = PeftModelForCausalLM.from_pretrained(model, checkpoint_dir)
        model = model.merge_and_unload()

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    gen_pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id
    return gen_pipeline


def build_synthesis_prompt(definition: str, demos: List[Dict]) -> str:
    sections: List[str] = [definition, ""]
    for example in demos:
        sections.append(f"Input: {example['input']}")
        sections.append(f"Output: {example['output']}")
        sections.append("")
    sections.append("Input:")
    return "\n".join(sections)


def split_examples(examples: List[Dict], group_size: int = 4) -> List[List[Dict]]:
    return [examples[i:i + group_size] for i in range(0, len(examples), group_size) if len(examples[i:i + group_size]) == group_size]


def parse_candidate(text: str) -> Optional[Dict[str, str]]:
    if "Output:" not in text:
        return None

    if "Input:" in text:
        text = text.split("Input:", 1)[1]
        if "Output:" not in text:
            return None

    input_part, output_part = text.split("Output:", 1)
    candidate_input = input_part.strip()
    candidate_output = output_part.split("Input:")[0].strip()

    if not candidate_input or not candidate_output:
        return None

    return {
        "input": candidate_input,
        "output": candidate_output
    }


def synthesize_candidates(
    model_name_or_path: str,
    source_examples: List[Dict],
    num_shots: int,
    prompts_per_group: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int
) -> List[Dict]:
    definition = source_examples[0]["full_prompt"].split("\n\n")[0]
    grouped_examples = split_examples(source_examples)
    rng = random.Random(seed)
    raw_candidates: List[Dict] = []

    gen_pipeline = load_generation_pipeline(model_name_or_path)
    tokenizer = gen_pipeline.tokenizer
    seen_source_inputs = {row["input"].strip() for row in source_examples}

    try:
        for group in grouped_examples:
            shot_indices = list(itertools.permutations(range(len(group)), num_shots))
            rng.shuffle(shot_indices)
            for indices in shot_indices[:prompts_per_group]:
                prompt = build_synthesis_prompt(definition, [group[i] for i in indices])
                generations = gen_pipeline(
                    prompt,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )
                parsed = parse_candidate(generations[0]["generated_text"][len(prompt):])
                if not parsed:
                    continue
                if parsed["input"] in seen_source_inputs:
                    continue
                raw_candidates.append({
                    "inputs": f"{definition}\n\n{parsed['input']}",
                    "outputs": parsed["output"]
                })
    finally:
        cleanup_model(gen_pipeline)

    return raw_candidates


def fallback_candidates(source_examples: List[Dict], limit: int) -> List[Dict]:
    fallback_rows: List[Dict] = []
    for example in source_examples[:limit]:
        fallback_rows.append({
            "inputs": example["full_prompt"],
            "outputs": example["output"]
        })
    return fallback_rows


def refine_candidates(
    model_name_or_path: str,
    checkpoint_dir: str,
    candidates: List[Dict],
    template: str,
    max_new_tokens: int
) -> List[Dict]:
    gen_pipeline = load_generation_pipeline(model_name_or_path, checkpoint_dir=checkpoint_dir)
    tokenizer = gen_pipeline.tokenizer
    refined_rows: List[Dict] = []

    try:
        for row in candidates:
            prompt = row["inputs"]
            if template == "llama2":
                prompt = LLAMA2_PROMPT.format(prompt=prompt)

            generations = gen_pipeline(
                prompt,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            target = generations[0]["generated_text"][len(prompt):].strip()
            if not target:
                continue
            refined_rows.append({
                "inputs": row["inputs"],
                "targets": target
            })
    finally:
        cleanup_model(gen_pipeline)

    return refined_rows


def dedupe_rows(rows: List[Dict], limit: int) -> List[Dict]:
    deduped: List[Dict] = []
    seen_inputs = set()
    for row in rows:
        key = row["inputs"].strip()
        if key in seen_inputs:
            continue
        seen_inputs.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--raw_output_path")
    parser.add_argument("--template", default="llama2")
    parser.add_argument("--num_shots", type=int, default=2)
    parser.add_argument("--prompts_per_group", type=int, default=3)
    parser.add_argument("--max_candidates", type=int, default=20)
    parser.add_argument("--synthesis_max_new_tokens", type=int, default=96)
    parser.add_argument("--refine_max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    raw_output_path = Path(args.raw_output_path) if args.raw_output_path else None

    source_examples = read_jsonl(input_path)
    if not source_examples:
        raise ValueError(f"No source examples found in {input_path}.")

    candidates = synthesize_candidates(
        model_name_or_path=args.model_name_or_path,
        source_examples=source_examples,
        num_shots=args.num_shots,
        prompts_per_group=args.prompts_per_group,
        max_new_tokens=args.synthesis_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed
    )
    candidates = dedupe_rows(candidates, args.max_candidates)
    if not candidates:
        candidates = fallback_candidates(source_examples, args.max_candidates)

    if raw_output_path:
        write_jsonl(raw_output_path, candidates)

    refined_rows = refine_candidates(
        model_name_or_path=args.model_name_or_path,
        checkpoint_dir=args.checkpoint_dir,
        candidates=candidates,
        template=args.template,
        max_new_tokens=args.refine_max_new_tokens
    )
    refined_rows = dedupe_rows(refined_rows, args.max_candidates)
    if not refined_rows:
        raise RuntimeError("Refinement produced no usable rehearsal instances.")

    write_jsonl(output_path, refined_rows)
    print(f"Saved {len(refined_rows)} rehearsal instances to {output_path}.")


if __name__ == "__main__":
    main()
