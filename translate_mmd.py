import os
import sys
import logging
import argparse
from tqdm import tqdm
import re

sys.path.append("/home/khkim/workspace/original_tech_demo")

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


def translate_lines_task(model_name, line, batch_size=32, hf_token=None):
    try:
        ret = translate_lines(model_name, line, batch_size=batch_size, exclude_determine_fn=exclude_determiner, hf_token=hf_token)

        # restore if the line is begin with multiple spaces or tabs.
        while line.startswith(" ") or line.startswith("\t"):
            ret = line[0] + ret
            line = line[1:]
    except Exception as e:
        ret = line

    return ret


def translate_mmd(args):
    lines = []

    with open(args.input_filename, "r") as f:
        for line in f:
            lines.append(line.replace("\n", ""))

    print(f"Number of lines: {len(lines)}")

    output_fn = args.input_filename.split(".")[:-1] + ["ko"] + [args.input_filename.split(".")[-1]]
    output_fn = ".".join(output_fn)
    print(f"Output file: {output_fn}")

    with open(output_fn, "w") as f:
      for line in tqdm(lines):
          translated_lines = translate_lines_task(
            args.model_name, line=line, batch_size=args.batch_size, hf_token=args.hf_token
          )

          f.write(translated_lines + "\n")

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filename', type=str, default="")
    parser.add_argument('--model-name', type=str, default="nlp-with-deeplearning/enko-t5-small-v0")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hf-token', type=str, default=None)
    
    args = parser.parse_args()
    if args.hf_token is None:
        env_var_hf_token = os.environ.get('HF_TOKEN')
        if env_var_hf_token is not None:
            args.hf_token = env_var_hf_token
            
    print(args)
    
    translate_mmd(args)
