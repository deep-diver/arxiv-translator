import os
import sys
import logging
import argparse
from tqdm import tqdm
import re

import multiprocessing as mp
import parmap

from transformers import T5TokenizerFast
from transformers import T5ForConditionalGeneration

from translate import translate_lines

def exclude_determiner(line):
    if len(line.split(" ")) <= 2:
        return True
    if line.startswith("**") and line.endswith("**") and len(line.split(" ")) < 4:
        return True
    if line.startswith("#") and len(line.split(" ")) < 4:
        return True
    if line.startswith("**#") and len(line.split(" ")) < 4:
        return True
    if line.startswith("[") and line.endswith("]") and len(line.split(" ")) < 3:
        return True
    if "\\begin{tabular}" in line or "\\end{tabular}" in line:
        return True
    if "\\" in line and "\\hline" in line:
        return True
    
    # pattern like \begin{???} or \end{???}
    if re.match(r"\\begin\{.*\}", line) or re.match(r"\\end\{.*\}", line):
        return True
    # pattern starts with "* [num] "
    if re.match(r"^\* \[\d+\] ", line):
        return True
    # pattern starts with "* ??? et al. [num] "
    if re.match(r"^\* .* et al\. \[\d+\] ", line):
        return True
    if re.match(r"^\* .* et al\.,? \[\d+\] ", line):
        return True

    return False


def translate_lines_async(idx, model, line, batch_size=32):
    try:
        ret = translate_lines(model, line, batch_size=batch_size, exclude_determine_fn=exclude_determiner)

        # restore if the line is begin with multiple spaces or tabs.
        while line.startswith(" ") or line.startswith("\t"):
            ret = line[0] + ret
            line = line[1:]
    except Exception as e:
        print(e)
        ret = line

    return ret

def instantiate_model(model_name, hf_token):
    model = T5ForConditionalGeneration.from_pretrained(model_name, token=hf_token, device_map="auto")
    tokenizer = T5TokenizerFast.from_pretrained(model_name, token=hf_token)
    return {"model": model, "tokenizer": tokenizer}
    # device = model.parameters().__next__().device
    # print(f"device = {device}")

def translate_mmd(args):
    lines = []

    with open(args.input_filename, "r") as f:
        for line in f:
            lines.append(line.replace("\n", ""))

    print(f"Number of lines: {len(lines)}")

    tasks = []
    models = [instantiate_model(args.model_name, args.hf_token) for _ in range(args.worker_num)]
    
    for idx, line in enumerate(lines):
        tasks.append((idx, models[idx % args.worker_num], line, args.batch_size))
    
    output_fn = args.input_filename.split(".")[:-1] + ["ko"] + [args.input_filename.split(".")[-1]]
    output_fn = ".".join(output_fn)

    with open(output_fn, "w") as f:
        for sub_tasks in tqdm(
            [tasks[i : i + args.worker_num] for i in range(0, len(tasks), args.worker_num)]
        ):
            translated_lines = parmap.starmap(
                translate_lines_async,
                sub_tasks,
                pm_pbar=False,
                pm_processes=args.worker_num,
                pm_chunksize=args.chunk_size,
            )

            for line in translated_lines:
                f.write(line + "\n")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    logger = logging.getLogger()
    logger.disabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filename', type=str, default="")
    parser.add_argument('--model-name', type=str, default="nlp-with-deeplearning/enko-t5-small-v0")
    parser.add_argument('--worker-num', type=int, default=5)
    parser.add_argument('--chunk-size', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hf-token', type=str, default=None)
    
    args = parser.parse_args()
    if args.hf_token is None:
        env_var_hf_token = os.environ.get('HF_TOKEN')
        if env_var_hf_token is not None:
            args.hf_token = env_var_hf_token
            
    print(args)
    
    translate_mmd(args)
