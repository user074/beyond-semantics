import torch
import os
import json
from tqdm import tqdm
import shortuuid

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
    KeywordsStoppingCriteria,
)


from PIL import Image
import math
from evaluation.config import ModelConfig, DataConfig, ResultsConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.eval.eval_base import Eval_Base


class Eval_Pope_Vicuna(Eval_Base):
    def __init__(
        self,
        model: ModelConfig,
        data: DataConfig,
        results: ResultsConfig,
        conv_mode,
        temperature,
        top_p,
        num_beams,
        strategy,
        permutation,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.results = results
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.permutation = permutation
        self.strategy = strategy
        self.kwargs = kwargs

    def eval_model(self):
        # Disable Torch initialization to save memory
        disable_torch_init()

        # Load pretrained LLaVA model, tokenizer, and image processor
        model_path = os.path.expanduser(self.model.path)
        model_name = os.path.basename(model_path)

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        device = "cuda"
        model.to(device)

        # Load the POPE dataset from the JSONL file
        self.question_file = os.path.join(
            self.data.question_file, f"coco_pope_{self.strategy}.jsonl"
        )
        dataset = self.read_jsonl(self.question_file)
        print(f"POPE Evaluation Strategy: {self.strategy}")
        print(f"The length of this dataset is {len(dataset)}")

        # Prepare output JSONL file for storing results
        answers_file_path = os.path.expanduser(self.results.directory)
        answers_file = os.path.join(
            answers_file_path,
            self.model.name,
            str(self.results.unique_id),
            "permutation" if self.permutation else "normal",
            f"coco_pope_{self.strategy}.jsonl",
        )
        if os.path.dirname(answers_file):
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        ans_file = open(answers_file, "w")

        print("----------------------------------------------------------")
        print(f"---------------------model: {self.model.name}-------------------")
        print(
            f"--------------------Permutation: {self.permutation}---------------------"
        )
        print("----------------------------------------------------------")

        # Iterate through the dataset
        for entry in tqdm(dataset):
            prompt = entry["text"]  # Use the given prompt
            addition_prompt = (
                ' \nPlease answer either "yes" or "no". For example, "yes".'
            )
            correct_answer = entry["label"]
            image_path = os.path.join(self.data.directory, "val2014", entry["image"])

            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}. Skipping this entry.")
                continue

            try:

                # Add special tokens if needed
                # if model.config.mm_use_im_start_end:
                #     qs = prompt + addition_prompt
                # else:
                qs = prompt + addition_prompt

                # Create conversation prompt
                conv = conv_templates[self.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                model_prompt = conv.get_prompt()

                # Tokenize the prompt
                input_ids = tokenizer.encode(model_prompt, return_tensors="pt").to(
                    device="cuda", non_blocking=True
                )

                # Define stopping criteria for generation
                stop_str = (
                    conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                )
                stopping_criteria = KeywordsStoppingCriteria(
                    [stop_str], tokenizer, input_ids
                )

                # Generate model output
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        num_beams=self.num_beams,
                        max_new_tokens=1024,
                        use_cache=True,
                    )

                # Decode the output tokens
                input_length = input_ids.shape[1]
                generated_ids = output_ids[0, input_length:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Generate unique answer ID
                ans_id = shortuuid.uuid()

                # Write the result to the JSONL output file
                ans_file.write(
                    json.dumps(
                        {
                            "idx": entry["question_id"],
                            "question": entry["text"],
                            "prompt": model_prompt,
                            "correct_answer": correct_answer,
                            "model_response": response,
                            "answer_id": ans_id,
                            "model_id": model_name,
                        }
                    )
                    + "\n"
                )
                ans_file.flush()

            except Exception as e:
                # Catch unexpected errors during image processing or model inference
                print(f"Error processing entry {entry['question_id']}: {e}")
                continue

        ans_file.close()
