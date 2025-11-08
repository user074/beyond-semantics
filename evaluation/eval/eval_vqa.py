import torch
import os
import json
from tqdm import tqdm
import shortuuid
import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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
from evaluation.eval.eval_base import Eval_Base
from evaluation.config import ModelConfig, DataConfig, ResultsConfig
from LLaVA.llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        conv_mode,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
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

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[
            0
        ]

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


class Eval_VQA(Eval_Base):
    def __init__(
        self,
        model: ModelConfig,
        data: DataConfig,
        results: ResultsConfig,
        conv_mode,
        temperature,
        top_p,
        num_beams,
        permutation,
        num_chunks=1,
        chunk_idx=0,
        max_new_tokens=128,
        **kwargs,
    ):
        self.model = model
        self.data = data
        self.results = results
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        self.max_new_tokens = max_new_tokens
        self.permutation = permutation
        self.kwargs = kwargs

    def create_data_loader(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        batch_size=1,
        num_workers=4,
    ):
        assert batch_size == 1, "batch_size must be 1"
        dataset = CustomDataset(
            questions,
            image_folder,
            tokenizer,
            image_processor,
            model_config,
            self.conv_mode,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
        return data_loader

    def convert_to_submission_format(self, result_file, test_split_file, output_file):
        """Convert model outputs to VQA submission format"""
        results = []
        error_line = 0

        # Read results
        for line_idx, line in enumerate(open(result_file)):
            try:
                results.append(json.loads(line))
            except:
                error_line += 1

        results = {x["question_id"]: x["text"] for x in results}
        test_split = [json.loads(line) for line in open(test_split_file)]
        split_ids = set([x["question_id"] for x in test_split])

        print(
            f"total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}"
        )

        all_answers = []
        answer_processor = EvalAIAnswerProcessor()

        for x in test_split:
            if x["question_id"] not in results:
                all_answers.append({"question_id": x["question_id"], "answer": ""})
            else:
                all_answers.append(
                    {
                        "question_id": x["question_id"],
                        "answer": answer_processor(results[x["question_id"]]),
                    }
                )

        # Save submission format
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(all_answers, f)

        print(f"Submission file saved to: {output_file}")

    def eval_model(self):
        """Two-step evaluation process:
        1. Generate model responses
        2. Convert to submission format
        """
        # Step 1: Generate model responses
        disable_torch_init()
        model_path = os.path.expanduser(self.model.path)
        model_name = get_model_name_from_path(model_path)

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, self.model.base, model_name
        )

        questions = [
            json.loads(q)
            for q in open(os.path.expanduser(self.data.question_file), "r")
        ]
        questions = get_chunk(questions, self.num_chunks, self.chunk_idx)

        answers_file = os.path.join(
            self.results.directory,
            self.model.name,
            str(self.results.unique_id),
            "permutation" if self.permutation else "normal",
            "vqa_result.jsonl",
        )
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")

        if (
            "plain" in model_name
            and "finetune" not in model_name.lower()
            and "mmtag" not in self.conv_mode
        ):
            self.conv_mode = self.conv_mode + "_mmtag"
            print(
                f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {self.conv_mode}."
            )

        data_loader = self.create_data_loader(
            questions, self.data.image_folder, tokenizer, image_processor, model.config
        )
        print("----------------------------------------------------------")
        print(f"---------------------model: {self.model.name}-------------------")
        print(
            f"--------------------Permutation: {self.permutation}---------------------"
        )
        print("----------------------------------------------------------")

        for (input_ids, image_tensor, image_sizes), line in tqdm(
            zip(data_loader, questions), total=len(questions)
        ):
            idx = line["question_id"]
            cur_prompt = line["text"]

            input_ids = input_ids.to(device="cuda", non_blocking=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(
                        dtype=torch.float16, device="cuda", non_blocking=True
                    ),
                    image_sizes=image_sizes,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    permutation=self.permutation,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {},
                    }
                )
                + "\n"
            )

        ans_file.close()
        print(f"Raw results saved to: {answers_file}")

        # Step 2: Convert to submission format
        submission_file = os.path.join(
            self.results.directory,
            self.model.name,
            str(self.results.unique_id),
            "submission.json",
        )

        self.convert_to_submission_format(
            result_file=answers_file,
            test_split_file=self.data.test_split_file,
            output_file=submission_file,
        )
