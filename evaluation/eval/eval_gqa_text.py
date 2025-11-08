import argparse
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import shortuuid

from datasets import load_dataset
from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from evaluation.eval.eval_base import Eval_Base
from evaluation.config import ModelConfig, DataConfig, ResultsConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class Eval_GQA_Vicuna(Eval_Base):
    def __init__(
        self,
        model: ModelConfig,
        data: DataConfig,
        results: ResultsConfig,
        conv_mode: str,
        temperature: float,
        top_p: float,
        num_beams: int,
        permutation: bool,
        question_extension: str = "Answer the question using a single word or phrase.",
        num_chunks: int = 1,
        chunk_idx: int = 0,
        max_new_tokens: int = 128,
        seed: int = 42,
        **kwargs,
    ):
        self.model = model
        self.data = data
        self.results = results
        self.question_extension = question_extension
        self.conv_mode = conv_mode
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.permutation = permutation
        self.kwargs = kwargs

    def process(self, line, tokenizer):
        additional_prompt = (
            "Answer with exactly one word. Do not include any extra words, phrases, or punctuation—just the single word answer. "
            "For example, if the correct answer is 'woman', your entire output should be: woman."
        )
        # qs = line["question"] + f"\n{self.question_extension}"
        qs = line["question"] + f"\n{additional_prompt}"
        # image_id = line["imageId"]
        # input_image = images[image_id]
        # if input_image is not None:
        #     if model.config.mm_use_im_start_end:
        #         qs = (
        #             DEFAULT_IM_START_TOKEN
        #             + DEFAULT_IMAGE_TOKEN
        #             + DEFAULT_IM_END_TOKEN
        #             + "\n"
        #             + qs
        #         )
        #     else:
        #         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # if input_image is None:
        #     image = None
        #     image_size = None
        #     image_tensor = None
        # else:
        #     image = input_image.convert("RGB")
        #     image_size = [image.size]
        #     image_tensor = process_images([image], image_processor, model.config)
        #     image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
            device="cuda", non_blocking=True
        )
        return input_ids, prompt

    def eval_model(self):
        # Set random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load model
        model_path = os.path.expanduser(self.model.path)
        model_name = get_model_name_from_path(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        device = "cuda"
        model.to(device)

        # Load dataset
        # images_data = load_dataset(
        #     "lmms-lab/GQA", "testdev_balanced_images", split="testdev"
        # )
        # images = {row["id"]: row["image"] for row in images_data}
        questions = load_dataset(
            "lmms-lab/GQA", "testdev_balanced_instructions", split="testdev"
        )

        # Setup output file
        answers_file = os.path.join(
            self.results.directory,
            self.model.name,
            str(self.results.unique_id),
            "permutation" if self.permutation else "normal",
            "gqa_balanced.jsonl",
        )
        if not answers_file.endswith(".jsonl"):
            raise ValueError("Answers file must be a jsonl file")

        basename = os.path.basename(answers_file)
        basename = os.path.splitext(basename)[0]
        answers_dir = os.path.dirname(answers_file)
        chunk_fname = f"{basename}_{self.chunk_idx}.jsonl"
        chunk_file = os.path.join(answers_dir, chunk_fname)
        os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

        ans_file = open(chunk_file, "w")
        print(f"Writing answers to {chunk_file}...")

        # Process questions
        idx = -1
        valid_chunk = get_chunk(len(questions), self.num_chunks, self.chunk_idx)
        print(f"Valid chunk: {valid_chunk}")

        for line in tqdm(questions, total=len(questions)):
            idx = idx + 1
            if idx < valid_chunk[0] or idx > valid_chunk[1]:
                continue

            # input_ids, image_tensor, image_sizes, prompt = self.process(
            #     line, tokenizer, image_processor, model.config, images
            # )
            input_ids, prompt = self.process(line, tokenizer)
            gt_answer = line["answer"]
            gt_full_answer = line["fullAnswer"]
            category = line["types"]

            input_ids = input_ids.to(device="cuda", non_blocking=True)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    # permutation=self.permutation,
                )

            input_length = input_ids.shape[1]
            generated_ids = output_ids[0, input_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": prompt,
                        "answer": response,
                        "gt_answer": gt_answer,
                        "category": category,
                        "model_id": model_name,
                        "gt_full_answer": gt_full_answer,
                    }
                )
                + "\n"
            )
            ans_file.flush()
        ans_file.close()
