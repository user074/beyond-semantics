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
import math

from evaluation.config import ModelConfig, ResultsConfig, DataConfig

from evaluation.eval.eval_base import Eval_Base


class Eval_CV_Bench(Eval_Base):
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
        disable_torch_init()

        model_path = os.path.expanduser(self.model.path)
        model_name = os.path.basename(model_path)

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, self.model.base, model_name
        )

        # dataset = self.read_jsonl(
        #     os.path.join(self.data.directory, self.data.question_file)
        # )
        dataset = load_dataset("nyu-visionx/CV-Bench")["test"]
        print(f"Dataset loaded with {len(dataset)} samples.")

        answers_file = os.path.join(
            self.results.directory,
            self.model.name,
            str(self.results.unique_id),
            "permutation" if self.permutation else "normal",
            "result.jsonl",
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
            question = entry["question"]
            prompt = entry["prompt"]  # Use the given prompt
            addition_prompt = ' \nPlease answer based on the choices given above. For example, "A", "B", "C", "D", or "E".'
            correct_answer = entry["answer"]
            # image_path = os.path.join(self.data.directory, entry["filename"])

            # Check if image exists
            # if not os.path.exists(image_path):
            #     print(f"Warning: Image not found at {image_path}. Skipping this entry.")
            #     continue

            # Add special tokens if needed
            if model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + prompt
                    + addition_prompt
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt + addition_prompt

            # Create conversation prompt
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            model_prompt = conv.get_prompt()

            # Load and preprocess the image
            # image = Image.open(image_path).convert("RGB")
            image = entry['image']
            # image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            #     "pixel_values"
            # ][0]
            image_tensor = process_images([image], image_processor, model.config)[0]
            image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

            # Tokenize the prompt
            input_ids = (
                tokenizer_image_token(
                    model_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            # Define stopping criteria for generation
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria(
                [stop_str], tokenizer, input_ids
            )

            # Generate model output
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0),
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    max_new_tokens=1024,
                    use_cache=False,
                    permutation=self.permutation,
                )

            # Decode the output tokens
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            # if outputs.endswith(stop_str):
            #     outputs = outputs[: -len(stop_str)]

            # Generate unique answer ID
            ans_id = shortuuid.uuid()

            # Write the result to the JSONL output file
            ans_file.write(
                json.dumps(
                    {
                        "idx": entry["idx"],
                        "type": entry["type"],
                        "task": entry["task"],
                        "source": entry["source"],
                        "question": question,
                        "prompt": prompt,
                        "correct_answer": correct_answer,
                        "model_response": outputs,
                        "answer_id": ans_id,
                        "model_id": model_name,
                    }
                )
                + "\n"
            )
            ans_file.flush()

        ans_file.close()
