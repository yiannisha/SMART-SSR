import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch

from ssr_metrics import save_aggregate_metrics


TASKS = ["qa", "qg", "sa", "sum", "trans"]
MODEL_SPECS = {
    "tinyllama": {
        "train_template": "llama2",
        "dataset_key": lambda task: f"ni_c012_icl_gen_km20_self_cl_queue_tinyllama_{task}",
        "generated_data_root": Path(
            "data/ni-cus0.12/genearated-icl-naive-kmeans20-self/tinyllama/cl_queue"
        ),
    },
    "llama2-7b-chat": {
        "train_template": "llama2",
        "dataset_key": lambda task: f"ni_c012_icl_gen_km20_self_cl_queue_llama2_7b_chat_{task}",
        "generated_data_root": Path(
            "data/ni-cus0.12/genearated-icl-naive-kmeans20-self/llama2-7b-chat/cl_queue"
        ),
    },
}


def precision_flags() -> List[str]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return ["--bf16", "True"]
    if torch.cuda.is_available():
        return ["--fp16", "True"]
    return []


def repo_env(cuda: str) -> Dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = cuda
    env["LOCAL_RANK"] = "-1"
    env.pop("RANK", None)
    env.pop("WORLD_SIZE", None)
    return env


def run_command(command: List[str], env: Dict[str, str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, env=env)


def checkpoint_dir(output_root: Path, task_index: int, task_name: str) -> Path:
    return output_root / f"{task_index + 1:02d}_{task_name}"


def rehearsal_file(generated_data_root: Path, task_name: str) -> Path:
    return generated_data_root / f"{task_name}.train.smp001.2shot.smp3.rp1.2.json"


def run_train(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    datasets: List[str],
    checkpoint: Path | None,
    output_dir: Path,
    train_max_steps: int,
    train_max_samples: int,
    max_source_length: int,
    max_target_length: int
) -> None:
    command = [
        python_bin,
        "src/train_bash.py",
        "--stage", "sft",
        "--model_name_or_path", model_name_or_path,
        "--do_train", "True",
        "--overwrite_cache", "True",
        "--overwrite_output_dir", "True",
        "--finetuning_type", "lora",
        "--template", template,
        "--dataset_dir", "data",
        "--dataset", ",".join(datasets),
        "--max_source_length", str(max_source_length),
        "--max_target_length", str(max_target_length),
        "--learning_rate", "2e-4",
        "--max_steps", str(train_max_steps),
        "--max_samples", str(train_max_samples),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--lr_scheduler_type", "cosine",
        "--max_grad_norm", "1.0",
        "--logging_steps", "1",
        "--save_strategy", "no",
        "--warmup_steps", "0",
        "--lora_rank", "8",
        "--lora_dropout", "0.1",
        "--lora_target", "q_proj,v_proj",
        "--resume_lora_training", "True",
        "--output_dir", str(output_dir),
        "--plot_loss", "False",
        *precision_flags(),
    ]
    if checkpoint is not None and checkpoint.exists():
        command.extend(["--checkpoint_dir", str(checkpoint)])
    run_command(command, env)


def run_eval(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    checkpoint_dir_path: Path,
    eval_task: str,
    eval_max_samples: int,
    max_source_length: int,
    max_target_length: int
) -> None:
    output_dir = checkpoint_dir_path / f"ni_c012_{eval_task}_eval"
    command = [
        python_bin,
        "src/train_bash.py",
        "--stage", "sftrp",
        "--model_name_or_path", model_name_or_path,
        "--checkpoint_dir", str(checkpoint_dir_path),
        "--overwrite_cache", "True",
        "--predict_with_generate", "True",
        "--finetuning_type", "lora",
        "--template", template,
        "--dataset_dir", "data",
        "--dataset", f"ni_c012_{eval_task}_eval",
        "--max_source_length", str(max_source_length),
        "--max_target_length", str(max_target_length),
        "--max_samples", str(eval_max_samples),
        "--per_device_eval_batch_size", "1",
        "--output_dir", str(output_dir),
        "--do_predict", "True",
        "--do_sample", "False",
        *precision_flags(),
    ]
    run_command(command, env)


def build_rehearsal_data(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    task_name: str,
    task_checkpoint_dir: Path,
    generated_data_root: Path,
    prompts_per_group: int,
    max_candidates: int,
    synthesis_max_new_tokens: int,
    refine_max_new_tokens: int
) -> None:
    output_path = rehearsal_file(generated_data_root, task_name)
    raw_output_path = output_path.with_suffix(".raw.jsonl")
    command = [
        python_bin,
        "custom/icl_gen/build_ssr_dataset.py",
        "--model_name_or_path", model_name_or_path,
        "--checkpoint_dir", str(task_checkpoint_dir),
        "--input_path", f"data/ni-cus0.12/split/{task_name}.train.smp001.json",
        "--output_path", str(output_path),
        "--raw_output_path", str(raw_output_path),
        "--template", template,
        "--num_shots", "2",
        "--prompts_per_group", str(prompts_per_group),
        "--max_candidates", str(max_candidates),
        "--synthesis_max_new_tokens", str(synthesis_max_new_tokens),
        "--refine_max_new_tokens", str(refine_max_new_tokens)
    ]
    run_command(command, env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--model_family", choices=sorted(MODEL_SPECS), default="tinyllama")
    parser.add_argument("--output_root", default="saves/basic-ssr-5task")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--train_max_steps", type=int, default=5)
    parser.add_argument("--train_max_samples", type=int, default=20)
    parser.add_argument("--eval_max_samples", type=int, default=5)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--prompts_per_group", type=int, default=3)
    parser.add_argument("--max_candidates", type=int, default=20)
    parser.add_argument("--synthesis_max_new_tokens", type=int, default=96)
    parser.add_argument("--refine_max_new_tokens", type=int, default=96)
    parser.add_argument("--eval_all_tasks", action="store_true", default=False)
    parser.add_argument("--skip_completed", action="store_true", default=False)
    args = parser.parse_args()

    python_bin = sys.executable
    env = repo_env(args.cuda)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[args.model_family]
    generated_data_root = spec["generated_data_root"]
    generated_data_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, List[str]]] = {}

    for idx, task_name in enumerate(args.tasks):
        current_checkpoint = checkpoint_dir(output_root, idx, task_name)
        previous_checkpoint = checkpoint_dir(output_root, idx - 1, args.tasks[idx - 1]) if idx > 0 else None

        train_datasets = [f"ni_c012_{task_name}_train"]
        if idx > 0:
            train_datasets.extend(
                spec["dataset_key"](previous_task)
                for previous_task in args.tasks[:idx]
            )

        if not args.skip_completed or not current_checkpoint.exists():
            run_train(
                python_bin=python_bin,
                env=env,
                model_name_or_path=args.model_name_or_path,
                template=spec["train_template"],
                datasets=train_datasets,
                checkpoint=previous_checkpoint,
                output_dir=current_checkpoint,
                train_max_steps=args.train_max_steps,
                train_max_samples=args.train_max_samples,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length
            )

        build_rehearsal_data(
            python_bin=python_bin,
            env=env,
            model_name_or_path=args.model_name_or_path,
            template=spec["train_template"],
            task_name=task_name,
            task_checkpoint_dir=current_checkpoint,
            generated_data_root=generated_data_root,
            prompts_per_group=args.prompts_per_group,
            max_candidates=args.max_candidates,
            synthesis_max_new_tokens=args.synthesis_max_new_tokens,
            refine_max_new_tokens=args.refine_max_new_tokens
        )

        evaluated_tasks = args.tasks if args.eval_all_tasks else args.tasks[:idx + 1]
        for eval_task in evaluated_tasks:
            run_eval(
                python_bin=python_bin,
                env=env,
                model_name_or_path=args.model_name_or_path,
                template=spec["train_template"],
                checkpoint_dir_path=current_checkpoint,
                eval_task=eval_task,
                eval_max_samples=args.eval_max_samples,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length
            )

        summary[task_name] = {
            "checkpoint_dir": [str(current_checkpoint)],
            "rehearsal_file": [str(rehearsal_file(generated_data_root, task_name))],
            "train_datasets": train_datasets,
            "evaluated_tasks": evaluated_tasks
        }

    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    save_aggregate_metrics(output_root, args.tasks)
    print(f"Saved run summary to {summary_path}.")
    print(f"Saved aggregate metrics to {output_root / 'aggregate_metrics.json'}.")


if __name__ == "__main__":
    main()
