import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import AutoModelForCausalLM, AutoTokenizer
from LLaVA.llava.conversation import conv_templates
from datasets import load_dataset
from evaluation.eval.eval_base import Eval_Base
from evaluation.config import ModelConfig, ResultsConfig, DataConfig


class Eval_CV_Bench_Vicuna(Eval_Base):
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
        """
        Initialize the evaluation class for Vicuna on CV-Bench dataset.

        Args:
            model: Model configuration (path and name).
            data: Data configuration (directory and question file).
            results: Results configuration (directory and unique_id).
            temperature: Sampling temperature for generation.
            top_p: Top-p sampling parameter.
            num_beams: Number of beams for beam search.
            **kwargs: Additional arguments.
        """
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
        """
        Evaluate the Vicuna model on the CV-Bench dataset using text-only inputs.
        """
        # Expand model path and extract model name
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

        # Load dataset
        dataset = load_dataset("nyu-visionx/CV-Bench")["test"]
        print(f"Dataset loaded with {len(dataset)} samples.")

        # Prepare output file
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
        print("----------------------------------------------------------")

        # Iterate through the dataset
        for entry in tqdm(dataset):
            question = entry["question"]
            prompt = entry["prompt"]  # Use the given prompt from the dataset
            addition_prompt = ' \nPlease answer based on the choices given above. For example, "A", "B", "C", "D", or "E".'
            correct_answer = entry["answer"]

            # Format the prompt for Vicuna
            qs = prompt + addition_prompt
            # qs = f"USER: {prompt}{addition_prompt}\nASSISTANT:"

            # Create conversation prompt
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            model_prompt = conv.get_prompt()

            # Tokenize the prompt
            input_ids = tokenizer.encode(model_prompt, return_tensors="pt").to(
                device="cuda", non_blocking=True
            )
            # print("Devise:: ", input_ids.device)
            # print("Devise:: ", model.device)
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

            # Decode the generated output
            input_length = input_ids.shape[1]
            generated_ids = output_ids[0, input_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            #     0
            # ].strip()

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
                        "model_response": response,
                        "answer_id": ans_id,
                        "model_id": model_name,
                    }
                )
                + "\n"
            )
            ans_file.flush()

        ans_file.close()
