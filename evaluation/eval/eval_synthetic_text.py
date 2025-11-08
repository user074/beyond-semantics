import torch
import os
import json
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
    KeywordsStoppingCriteria,
)

from PIL import Image
from evaluation.config import ModelConfig, DataConfig, ResultsConfig
from evaluation.eval.eval_base import Eval_Base
from transformers import AutoModelForCausalLM, AutoTokenizer


class EvalSynthetic_Vicuna(Eval_Base):
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
        **kwargs,
    ):
        self.model = model
        self.data = data
        self.results = results
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.permutation = permutation
        self.kwargs = kwargs

    def eval_model(self):
        # Disable Torch initialization to save memory
        disable_torch_init()

        # Load pretrained LLaVA model, tokenizer, and image processor
        model_path = os.path.expanduser(self.model.path)
        model_name = os.path.basename(model_path)

        # tokenizer, model, image_processor, context_len = load_pretrained_model(
        #     model_path, self.model.base, model_name
        # )
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

        # Load dataset using load_dataset
        dataset = load_dataset(
            "parquet", data_files=os.path.join(self.data.directory, self.data.file_path)
        )["train"]

        print(f"Dataset loaded with {len(dataset)} samples.")

        # Prepare output JSONL file for storing results
        answers_file = os.path.join(
            self.results.directory,
            self.model.name,
            str(self.results.unique_id),
            "permutation" if self.permutation else "normal",
            "synthetic_results.jsonl",
        )
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
            # image = entry["image_filename"]
            qa_pairs = entry["qa_pairs"]

            # Convert image to tensor
            # image_tensor = process_images([image], image_processor, model.config)[0]
            # image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

            for qa in qa_pairs:
                prompt = qa["question"]
                correct_answer = qa["answer"]

                # Constructing the conversation prompt
                # if model.config.mm_use_im_start_end:
                #     qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{prompt}"
                # else:
                #     qs = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
                qs = prompt
                conv = conv_templates[self.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                model_prompt = conv.get_prompt()

                # Tokenize the prompt
                # input_ids = (
                #     tokenizer_image_token(
                #         model_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                #     )
                #     .unsqueeze(0)
                #     .cuda()
                # )
                input_ids = tokenizer.encode(model_prompt, return_tensors="pt").to(
                    device="cuda", non_blocking=True
                )

                # Set stopping criteria
                # stop_str = (
                #     conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                # )
                # stopping_criteria = KeywordsStoppingCriteria(
                #     [stop_str], tokenizer, input_ids
                # )

                # Generate response
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

                # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                #     0
                # ].strip()
                input_length = input_ids.shape[1]
                generated_ids = output_ids[0, input_length:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)

                ans_file.write(
                    json.dumps(
                        {
                            "entry_id": entry["entry_id"],
                            "test_category": entry["test_category"],
                            "qa_category": qa["qa_category"],
                            "question": prompt,
                            "correct_answer": correct_answer,
                            "model_response": response,
                            "objects": entry["objects"],
                            "model_id": model_name,
                        }
                    )
                    + "\n"
                )
                ans_file.flush()

        ans_file.close()
        print("Evaluation completed. Results saved.")
