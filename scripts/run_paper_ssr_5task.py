import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


TASKS = ["qa", "qg", "sa", "sum", "trans"]
MODEL_SPECS = {
    "tinyllama": {
        "train_template": "llama2",
        "gen_template": "vanilla",
        "model_tag": "tinyllama",
        "dataset_key": lambda task: f"ni_c012_icl_gen_km20_self_cl_queue_tinyllama_{task}",
    },
    "llama2-7b-chat": {
        "train_template": "llama2",
        "gen_template": "vanilla",
        "model_tag": "llama2-7b-chat",
        "dataset_key": lambda task: f"ni_c012_icl_gen_km20_self_cl_queue_llama2_7b_chat_{task}",
    },
    "llama2-7b": {
        "train_template": "vanilla",
        "gen_template": "vanilla",
        "model_tag": "llama2-7b",
        "dataset_key": lambda task: f"ni_c012_icl_gen_km20_self_cl_queue_{task}",
    }
}


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


def build_paths(model_tag: str, task_name: str) -> Dict[str, Path]:
    raw_root = Path(f"data/ni-cus0.12/genearated-icl-naive/{model_tag}/ori-van")
    parsed_root = Path(f"data/ni-cus0.12/genearated-icl-naive-parsed-filtered/{model_tag}/ori-van")
    refined_root = Path(f"data/ni-cus0.12/genearated-icl-naive-kmeans20-self/{model_tag}/refined")
    final_root = Path(f"data/ni-cus0.12/genearated-icl-naive-kmeans20-self/{model_tag}/cl_queue")
    suffix = f"{task_name}.train.smp001.2shot.smp3.rp1.2.json"
    return {
        "raw": raw_root / suffix,
        "parsed": parsed_root / suffix,
        "refined": refined_root / suffix,
        "final": final_root / suffix,
    }


def run_single_task_train(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    task_name: str,
    output_dir: Path,
    max_source_length: int,
    max_target_length: int,
    num_train_epochs: float,
    max_samples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int
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
        "--dataset", f"ni_c012_{task_name}_train",
        "--max_source_length", str(max_source_length),
        "--max_target_length", str(max_target_length),
        "--learning_rate", "2e-4",
        "--num_train_epochs", str(num_train_epochs),
        "--max_samples", str(max_samples),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--lr_scheduler_type", "cosine",
        "--max_grad_norm", "1.0",
        "--logging_steps", "5",
        "--save_strategy", "no",
        "--warmup_steps", "0",
        "--lora_rank", "8",
        "--lora_dropout", "0.1",
        "--lora_target", "q_proj,v_proj",
        "--resume_lora_training", "True",
        "--output_dir", str(output_dir),
        "--plot_loss", "True",
        "--bf16", "True"
    ]
    run_command(command, env)


def run_cl_train(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    datasets: List[str],
    checkpoint: Path,
    output_dir: Path,
    max_source_length: int,
    max_target_length: int,
    num_train_epochs: float,
    max_samples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int
) -> None:
    command = [
        python_bin,
        "src/train_bash.py",
        "--stage", "sft",
        "--model_name_or_path", model_name_or_path,
        "--checkpoint_dir", str(checkpoint),
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
        "--num_train_epochs", str(num_train_epochs),
        "--max_samples", str(max_samples),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--lr_scheduler_type", "cosine",
        "--max_grad_norm", "1.0",
        "--logging_steps", "5",
        "--save_strategy", "no",
        "--warmup_steps", "0",
        "--lora_rank", "8",
        "--lora_dropout", "0.1",
        "--lora_target", "q_proj,v_proj",
        "--resume_lora_training", "True",
        "--output_dir", str(output_dir),
        "--plot_loss", "True",
        "--bf16", "True"
    ]
    run_command(command, env)


def run_eval(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    checkpoint_dir_path: Path,
    eval_task: str,
    max_source_length: int,
    max_target_length: int,
    max_samples: int
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
        "--max_samples", str(max_samples),
        "--per_device_eval_batch_size", "1",
        "--output_dir", str(output_dir),
        "--do_predict", "True",
        "--do_sample", "False",
        "--bf16", "True"
    ]
    run_command(command, env)


def run_raw_generation(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    task_name: str,
    output_path: Path
) -> None:
    command = [
        python_bin,
        "custom/icl_gen/complete_param_nic010_cate.py",
        "--model_name_or_path", model_name_or_path,
        "--input_path", f"data/ni-cus0.12/split/{task_name}.train.smp001.json",
        "--output_path", str(output_path),
        "--do_sample", "True",
        "--do_sample_retries", "3",
        "--top_p", "0.6",
        "--temperature", "0.9",
        "--repetition_penalty", "1.2",
        "--max_length", "2048",
        "--num_beams", "1",
        "--n_shots", "2",
        "--template", template,
        "--cate_task_style", "False",
        "--resume", "True"
    ]
    run_command(command, env)


def run_parse_filter(
    python_bin: str,
    env: Dict[str, str],
    tokenizer_name_or_path: str,
    input_path: Path,
    output_path: Path
) -> None:
    command = [
        python_bin,
        "custom/icl_gen/parse_filter_generated.py",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--tokenizer_name_or_path", tokenizer_name_or_path,
    ]
    run_command(command, env)


def run_kmeans_selection(
    python_bin: str,
    env: Dict[str, str],
    input_path: Path,
    output_path: Path,
    sample_memory: int
) -> None:
    command = [
        python_bin,
        "custom/icl_gen/select_kmeans_examples.py",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--sample_memory", str(sample_memory),
        "--n_cluster", "20",
    ]
    run_command(command, env)


def run_refinement(
    python_bin: str,
    env: Dict[str, str],
    model_name_or_path: str,
    template: str,
    checkpoint_dir_path: Path,
    input_path: Path,
    output_path: Path
) -> None:
    if output_path.exists():
        output_path.unlink()
    command = [
        python_bin,
        "custom/icl_gen/label_param.py",
        "--model_name_or_path", model_name_or_path,
        "--ckpt_dir", str(checkpoint_dir_path),
        "--finetuning_type", "lora",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--do_sample", "False",
        "--max_length", "2048",
        "--template", template
    ]
    run_command(command, env)


def load_rouge_l(path: Path) -> float:
    with path.open("r") as handle:
        return float(json.load(handle)["predict_rouge-l"])


def save_aggregate_metrics(output_root: Path, tasks: List[str]) -> None:
    score_matrix: Dict[str, Dict[str, float]] = {}
    for idx, task_name in enumerate(tasks):
        stage_dir = checkpoint_dir(output_root, idx, task_name)
        score_matrix[task_name] = {}
        for eval_task in tasks:
            result_path = stage_dir / f"ni_c012_{eval_task}_eval" / "all_results.json"
            if result_path.exists():
                score_matrix[task_name][eval_task] = load_rouge_l(result_path)

    final_task = tasks[-1]
    final_scores = [score_matrix[final_task][task] for task in tasks]
    average_rouge_l = sum(final_scores) / len(final_scores)

    bwt_terms = []
    for idx, task_name in enumerate(tasks[:-1]):
        bwt_terms.append(score_matrix[final_task][task_name] - score_matrix[task_name][task_name])
    bwt = sum(bwt_terms) / len(bwt_terms) if bwt_terms else 0.0

    fwt_terms = []
    for idx in range(1, len(tasks)):
        previous_stage = tasks[idx - 1]
        current_task = tasks[idx]
        fwt_terms.append(score_matrix[previous_stage][current_task])
    fwt = sum(fwt_terms) / len(fwt_terms) if fwt_terms else 0.0

    aggregate = {
        "average_rouge_l": average_rouge_l,
        "bwt": bwt,
        "fwt": fwt,
        "score_matrix_rouge_l": score_matrix
    }
    (output_root / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_family", choices=sorted(MODEL_SPECS), default="llama2-7b-chat")
    parser.add_argument("--output_root", default="saves/paper-ssr-5task")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--sample_memory", type=int, default=200)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_all_tasks", action="store_true", default=True)
    parser.add_argument("--skip_completed", action="store_true", default=False)
    args = parser.parse_args()

    python_bin = sys.executable
    env = repo_env(args.cuda)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[args.model_family]

    summary: Dict[str, Dict[str, List[str]]] = {}

    for idx, task_name in enumerate(args.tasks):
        paths = build_paths(spec["model_tag"], task_name)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        if not paths["raw"].exists():
            run_raw_generation(
                python_bin=python_bin,
                env=env,
                model_name_or_path=args.model_name_or_path,
                template=spec["gen_template"],
                task_name=task_name,
                output_path=paths["raw"]
            )

        if not paths["parsed"].exists():
            run_parse_filter(
                python_bin=python_bin,
                env=env,
                tokenizer_name_or_path=args.model_name_or_path,
                input_path=paths["raw"],
                output_path=paths["parsed"]
            )

        current_checkpoint = checkpoint_dir(output_root, idx, task_name)
        previous_checkpoint = checkpoint_dir(output_root, idx - 1, args.tasks[idx - 1]) if idx > 0 else None

        if not args.skip_completed or not current_checkpoint.exists():
            if idx == 0:
                run_single_task_train(
                    python_bin=python_bin,
                    env=env,
                    model_name_or_path=args.model_name_or_path,
                    template=spec["train_template"],
                    task_name=task_name,
                    output_dir=current_checkpoint,
                    max_source_length=args.max_source_length,
                    max_target_length=args.max_target_length,
                    num_train_epochs=args.num_train_epochs,
                    max_samples=args.max_samples,
                    per_device_train_batch_size=args.per_device_train_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps
                )
            else:
                datasets = [f"ni_c012_{task_name}_train"]
                datasets.extend(spec["dataset_key"](prev_task) for prev_task in args.tasks[:idx])
                run_cl_train(
                    python_bin=python_bin,
                    env=env,
                    model_name_or_path=args.model_name_or_path,
                    template=spec["train_template"],
                    datasets=datasets,
                    checkpoint=previous_checkpoint,
                    output_dir=current_checkpoint,
                    max_source_length=args.max_source_length,
                    max_target_length=args.max_target_length,
                    num_train_epochs=args.num_train_epochs,
                    max_samples=args.max_samples,
                    per_device_train_batch_size=args.per_device_train_batch_size,
                        gradient_accumulation_steps=args.gradient_accumulation_steps
                    )

        if not paths["refined"].exists():
            run_refinement(
                python_bin=python_bin,
                env=env,
                model_name_or_path=args.model_name_or_path,
                template=spec["train_template"],
                checkpoint_dir_path=current_checkpoint,
                input_path=paths["parsed"],
                output_path=paths["refined"]
            )

        if not paths["final"].exists():
            run_kmeans_selection(
                python_bin=python_bin,
                env=env,
                input_path=paths["refined"],
                output_path=paths["final"],
                sample_memory=args.sample_memory
            )

        eval_tasks = args.tasks if args.eval_all_tasks else args.tasks[:idx + 1]
        for eval_task in eval_tasks:
            run_eval(
                python_bin=python_bin,
                env=env,
                model_name_or_path=args.model_name_or_path,
                template=spec["train_template"],
                checkpoint_dir_path=current_checkpoint,
                eval_task=eval_task,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                max_samples=args.max_samples
            )

        summary[task_name] = {
            "checkpoint_dir": [str(current_checkpoint)],
            "raw_generation_file": [str(paths["raw"])],
            "parsed_generation_file": [str(paths["parsed"])],
            "refined_generation_file": [str(paths["refined"])],
            "kmeans_selection_file": [str(paths["final"])],
            "rehearsal_file": [str(paths["final"])],
            "evaluated_tasks": eval_tasks
        }

    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    save_aggregate_metrics(output_root, args.tasks)
    print(f"Saved run summary to {summary_path}.")
    print(f"Saved aggregate metrics to {output_root / 'aggregate_metrics.json'}.")


if __name__ == "__main__":
    main()
