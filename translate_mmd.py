import sys
import logging
import concurrent.futures
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


def translate_lines_task(model_name, line, batch_size=32):
    try:
        ret = translate_lines(model_name, line, batch_size=batch_size, exclude_determine_fn=exclude_determiner)

        # restore if the line is begin with multiple spaces or tabs.
        while line.startswith(" ") or line.startswith("\t"):
            ret = line[0] + ret
            line = line[1:]
    except Exception as e:
        ret = line

    return ret


def translate_mmd(input_fn, model_name, chunksize=10):
    lines = []

    with open(input_fn, "r") as f:
        for line in f:
            lines.append(line.replace("\n", ""))

    print(f"Number of lines: {len(lines)}")

    output_fn = input_fn.split(".")[:-1] + ["ko"] + [input_fn.split(".")[-1]]
    output_fn = ".".join(output_fn)
    print(f"Output file: {output_fn}")

    with open(output_fn, "w") as f:
      for line in tqdm(lines):
          translated_lines = translate_lines_task(
            model_name, line=line, batch_size=256
          )

          f.write(translated_lines + "\n")

if __name__ == "__main__":
    input_fn = sys.argv[1]

    logger = logging.getLogger()
    logger.disabled = True

    translate_mmd(
        input_fn,
        model_name="nlp-with-deeplearning/enko-t5-small-v0"
    )
