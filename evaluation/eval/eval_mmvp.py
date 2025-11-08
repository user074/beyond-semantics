import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import csv
from PIL import Image

from datasets import load_dataset
from huggingface_hub import hf_hub_download
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


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i, i + chunk_size - 1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def give_options(input_string):
    parts = input_string.split("(")
    result = [part.split(")")[1].strip() for part in parts[1:]]
    return result


class Eval_MMVP(Eval_Base):
    def __init__(
        self,
        model: ModelConfig,
        data: DataConfig,
        results: ResultsConfig,
        conv_mode: str,
        permutation: bool,
        question_extension: str = "Answer with the option's letter from the given choices directly.",
        num_chunks: int = 1,
        chunk_idx: int = 0,
        temperature: float = 0,
        top_p: float = None,
        num_beams: int = 1,
        max_new_tokens: int = 512,
        seed: int = 73,
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

    def process(self, line, tokenizer, image_processor, model_config, images):
        qs = line["question"] + " Options:"
        options = line["options"].split("(b)")
        parts = [part.strip() for part in options]
        parts = [part.replace("(a)", "A.").replace("(b)", "B.") for part in parts]
        if len(parts) > 1:
            parts[1] = "B. " + parts[1]
        for part in parts:
            qs += f"\n{part}"
        qs += f"\n{self.question_extension}"

        image_id = line["imageId"]
        input_image = images[image_id]
        if input_image is not None:
            if model_config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if input_image is None:
            image = None
            image_size = None
            image_tensor = None
        else:
            image = input_image
            image_size = [image.size]
            image_tensor = process_images([image], image_processor, model_config)

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        return input_ids, image_tensor, image_size, prompt

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
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, self.model.base, model_name
        )

        # Load dataset
        images = {}
        for i in range(300):
            file_path = hf_hub_download(
                repo_id="MMVP/MMVP",
                filename=f"{i+1}.jpg",
                subfolder="MMVP Images",
                repo_type="dataset",
            )
            images[i] = Image.open(file_path).convert("RGB")

        questions = []
        file_path = hf_hub_download(
            repo_id="MMVP/MMVP", filename="Questions.csv", repo_type="dataset"
        )
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] in ["lndex", "Index"]:
                    continue
                questions.append(
                    {
                        "question": str(row[1]),
                        "imageId": int(row[0]) - 1,
                        "options": str(row[2]),
                        "text_options": give_options(str(row[2])),
                        "answer": str(row[3]),
                    }
                )

        # Setup output file
        answers_file = os.path.join(
            self.results.directory,
            self.model.name,
            str(self.results.unique_id),
            "permutation" if self.permutation else "normal",
            "mmvp_result.jsonl",
        )
        basename = os.path.basename(answers_file)
        basename = os.path.splitext(basename)[0]
        chunk_file = os.path.join(
            os.path.dirname(answers_file), f"{basename}_{self.chunk_idx}.jsonl"
        )
        os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

        with open(chunk_file, "w") as ans_file:
            idx = -1
            valid_chunk = get_chunk(len(questions), self.num_chunks, self.chunk_idx)
            print(f"Processing chunk: {valid_chunk}")

            for line in tqdm(questions, total=len(questions)):
                idx = idx + 1
                if idx < valid_chunk[0] or idx > valid_chunk[1]:
                    continue

                input_ids, image_tensor, image_sizes, prompt = self.process(
                    line, tokenizer, image_processor, model.config, images
                )

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=(
                            image_tensor.to(device="cuda", non_blocking=True)
                            if image_tensor is not None
                            else None
                        ),
                        image_sizes=image_sizes,
                        do_sample=True if self.temperature > 0 else False,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        num_beams=self.num_beams,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=True,
                    )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                    0
                ].strip()

                ans_file.write(
                    json.dumps(
                        {
                            "question_id": idx,
                            "prompt": prompt,
                            "answer": outputs,
                            "gt_answer": line["answer"],
                            "model_id": model_name,
                            "text_options": line["text_options"],
                        }
                    )
                    + "\n"
                )
                ans_file.flush()
