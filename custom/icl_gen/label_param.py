from transformers import AutoTokenizer
import transformers
import torch
import os
from tqdm import tqdm
import jsonlines

# import importlib
import ast
import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModelForCausalLM

from llmtuner.extras.template import render_one_turn_prompt


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_pipeline(model, tokenizer):
    pipeline_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "torch_dtype": get_dtype(),
    }
    if torch.cuda.is_available():
        pipeline_kwargs["device_map"] = "auto"
    return transformers.pipeline("text-generation", **pipeline_kwargs)


def main(args):
    transformers.set_seed(42)

    # model = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.finetuning_type == "full":
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path if not args.ckpt_dir else args.ckpt_dir)
        pipeline = build_pipeline(model, tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        model = PeftModelForCausalLM.from_pretrained(model, args.ckpt_dir)
        model = model.merge_and_unload()
        pipeline = build_pipeline(model, tokenizer)

    print("old model.config.use_cache:", pipeline.model.config.use_cache)
    pipeline.model.config.use_cache = True
    print("new model.config.use_cache:", pipeline.model.config.use_cache)


    pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

    input_path = args.input_path
    output_path = args.output_path

    with jsonlines.open(input_path, "r") as f:
        data = [line for line in f]

    if os.path.exists(output_path):
        print('Error: destination output path already exists!')
        exit(-1)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    for i, line in tqdm(enumerate(data), total=len(data)):
        if 'inputs' in line:
            inputs = line['inputs']
        elif 'full_prompt' in line:
            inputs = line['full_prompt']
        else:
            inputs = line['definition_input']
        instruction = render_one_turn_prompt(tokenizer, args.template, inputs)
        print("========")
        print(instruction)
        print("--------")
        sequences = pipeline(
            instruction,
            do_sample=args.do_sample,
            # top_p=args.top_p,
            # temperature=args.temperature,
            max_length=args.max_length,
            # min_new_tokens=,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            batch_size=1,
        )
        print(sequences)
        result_text = sequences[0]["generated_text"][len(instruction):].strip()

        save_dict = {
            "inputs": inputs,
            "targets": result_text,
        }

        with jsonlines.open(output_path, "a") as file:
            file.write(save_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default=None, required=True, help=""
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default=None, help=""
    )
    parser.add_argument("--input_path", type=str, default=None, required=True, help="")
    parser.add_argument("--output_path", type=str, default=None, required=True, help="")
    parser.add_argument("--do_sample", type=ast.literal_eval, default=False, help="")
    parser.add_argument("--top_p", type=float, default=0.6, help="")
    parser.add_argument("--temperature", type=float, default=0.9, help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="")
    parser.add_argument("--repetition_penalty", type=float, default=1., help="")
    parser.add_argument("--finetuning_type", type=str, default="full")
    parser.add_argument("--template", type=str, default="vanilla", help="")
    # parser.add_argument("--preserve_ratio", type=float, default=0.9, help="")
    # parser.add_argument("--preserve_word_step", type=int, default=1, help="")
    # parser.add_argument("--template_list_enum", type=ast.literal_eval, default=False)
    # parser.add_argument("--icl", type=ast.literal_eval, default=False)
    args = parser.parse_args()
    print(args)
    main(args)
