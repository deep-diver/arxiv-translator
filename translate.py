import requests
import json

import kss

from transformers import T5TokenizerFast
from transformers import T5ForConditionalGeneration
from transformers import GenerationConfig

model_name = "nlp-with-deeplearning/enko-t5-small-v0"
loaded = False
model = T5ForConditionalGeneration.from_pretrained("nlp-with-deeplearning/enko-t5-small-v0")
tokenizer = T5TokenizerFast.from_pretrained("nlp-with-deeplearning/enko-t5-small-v0")
device = model.parameters().__next__().device

def translate(model_name, sentences):
    input_ids = tokenizer.batch_encode_plus(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).input_ids.to(device)

    generation_config = GenerationConfig(
        max_new_tokens=256,
        early_stopping=True,
        do_sample=False,
        num_beams=8,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        repetition_penalty=1.2,
        length_penalty=1.0,
    )

    beam_output = model.generate(
        input_ids,
        generation_config=generation_config,
        # forced_decoder_ids=[[1, 5], [2, 9]],
    )

    outputs = []
    for i in range(len(sentences)):
        outputs.append(
            tokenizer.decode(
                beam_output[i],
                skip_special_tokens=False,
            )
        )

    return outputs

def translate_lines(
        model_name,
        lines,
        batch_size=32,
        exclude_determine_fn=None,
        use_kss=True,
        remove_bos_eos_pad=True,
    ):
    if exclude_determine_fn is None:
        exclude_determine_fn = lambda x: False

    if remove_bos_eos_pad:
        remove_bos_eos_pad_fn = lambda x: [
            line.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip() for line in x
        ]
    else:
        remove_bos_eos_pad_fn = lambda x: x

    if isinstance(lines, str):
        lines = lines.split("\n")

    assert isinstance(lines, list), \
        f"lines must be list or str, but {type(lines)}"

    exclude_index_map = {}
    line_to_buffer_index_map = {}
    buffer = []
    for idx, line in enumerate(lines):
        if line.strip() == "":
            exclude_index_map[idx] = True
            continue

        if exclude_determine_fn(line):
            exclude_index_map[idx] = True
            continue

        exclude_index_map[idx] = False

        splitted = kss.split_sentences(line) if use_kss else [line]
        line_to_buffer_index_map[idx] = []
        for s in splitted:
            buffer.append(s)
            line_to_buffer_index_map[idx] += [len(buffer) - 1]

    translated_buffer = []
    for i in range(0, len(buffer), batch_size):
        translated_buffer += remove_bos_eos_pad_fn(translate(model_name, buffer[i:i + batch_size]))

    translated_lines = []
    for idx, line in enumerate(lines):
        if exclude_index_map[idx]:
            translated_lines.append(line)
            continue

        line_buffer = []
        for buffer_idx in line_to_buffer_index_map[idx]:
            line_buffer.append(translated_buffer[buffer_idx])

        translated_lines.append(" ".join(line_buffer))

    translated_lines = "\n".join([line.strip() for line in translated_lines])

    return translated_lines
