import os
import re
import json
import pandas as pd
from tqdm import tqdm
import logging
import time
from dotenv import load_dotenv
from openai import OpenAI
import openai
from langchain_ollama import ChatOllama
from evaluation.config import ModelConfig, ResultsConfig
import wandb

logging.getLogger("httpx").setLevel(logging.WARNING)
load_dotenv()


class Grader_GQA:
    def __init__(
        self,
        model: ModelConfig,
        results: ResultsConfig,
        llm_model=None,
        permutation=False,
        grade_method="simple",  # "simple" or "llm"
        **kwargs,
    ):
        self.model = model
        self.results_info = results
        self.permutation = permutation
        self.llm_model = llm_model
        # Defer LLM client initialization unless explicitly grading with LLM
        self.client = None
        if grade_method == "llm" and self.llm_model is not None:
            self.client = self.get_llm_client(self.llm_model)

        self.NUM_SECONDS_TO_SLEEP = 10
        self.results = []
        self.grade_method = grade_method
        self.kwargs = kwargs

        # Optional wandb (controlled via results.use_wandb in YAML; default False)
        self.use_wandb = getattr(self.results_info, "use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project="vlm-evaluation",
                group="gqa",
                name=f"gqa_{self.model.name}_{self.results_info.unique_id}_{self.results_info.strategy}_{'permutation' if self.permutation else 'normal'}",
                config={
                    "model": self.model.name,
                    "permutation": self.permutation,
                    "dataset": "gqa",
                    "strategy": self.results_info.strategy,
                    "llm_model": llm_model,
                },
                reinit=True,
            )

    def get_llm_client(self, llm_model):
        client = None
        if llm_model["type"] == "gpt":
            client = OpenAI(api_key=os.getenv("API"))
        elif llm_model["type"] == "ollama":
            client = ChatOllama(
                model=llm_model["model"],
                temperature=0.2,
                num_gpu=llm_model["num_gpu"],
                # num_predict=256,
                # other params ...
            )
        else:
            print("Warning: You have to provide the llm type, like 'gpt' or 'ollama'!")

        print("----------------------------------------")
        print(f"Client: {llm_model['type']}")
        print(f"Model: {llm_model['model']}")
        print("----------------------------------------")

        return client

    def get_llm_response(self, question):
        # Lazily initialize the client if needed
        if self.llm_model and self.client is None:
            self.client = self.get_llm_client(self.llm_model)
        if self.llm_model["type"] == "gpt":
            return self.get_yes_no_answer_gpt(question)
        elif self.llm_model["type"] == "ollama":
            return self.get_yes_no_answer_ollama(question)

    def get_yes_no_answer_gpt(self, question):
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model[
                        "model"
                    ],  # Fixed typo in model name from "gpt-4o"
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful and precise assistant for checking the quality of the answer. Please answer with exactly 'yes' or 'no' only.",
                        },
                        {
                            "role": "user",
                            "content": question,
                        },
                    ],
                    temperature=0.2,
                )

                answer = completion.choices[0].message.content.strip().lower()

                return answer

            except openai.APIError as e:
                logging.error(f"OpenAI API returned an API Error: {e}")
                time.sleep(self.NUM_SECONDS_TO_SLEEP)
            except openai.APIConnectionError as e:
                logging.error(f"Failed to connect to OpenAI API: {e}")
                time.sleep(self.NUM_SECONDS_TO_SLEEP)
            except openai.RateLimitError as e:
                logging.error(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(self.NUM_SECONDS_TO_SLEEP * 2)  # Longer wait for rate limits
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                time.sleep(self.NUM_SECONDS_TO_SLEEP)

            retry_count += 1

        # If all retries failed, default to 'no' and log
        logging.error("Max retries reached for API call. Defaulting to 'no'")
        return f"No response from {self.llm_model['model']}"

    def get_yes_no_answer_ollama(self, question):
        messages = [
            (
                "system",
                "You are a helpful and precise assistant for checking the quality of the answer. Please answer with exactly 'yes' or 'no' only.",
            ),
            ("human", question),
        ]
        answer = self.client.invoke(messages).content
        answer = answer.lower()
        return answer

    def normalize_answer(self, answer):
        if not isinstance(answer, str):
            return ""
        return answer.strip().lower().rstrip(".")

    def get_simple_answer(self, answer):
        # Strict validation of answer
        if answer in ["yes", "no"]:
            return answer
        else:
            # If answer isn't exactly 'yes' or 'no', try to extract it
            yes_no_regex = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
            match = yes_no_regex.search(answer)
            if match:
                return match.group(1).lower()
            else:
                # If no 'yes' or 'no' found, default to 'no' and log
                logging.warning(
                    f"Unexpected API response: {answer}. Defaulting to 'no'"
                )
                return "no"

    def simple_grade(self, correct_answer, model_response):
        correct_norm = self.normalize_answer(correct_answer)
        response_norm = self.normalize_answer(model_response)
        return "yes" if correct_norm == response_norm else "no"

    def grade_(self, answers_file):
        if not os.path.exists(answers_file):
            raise FileNotFoundError(f"Answer file not found: {answers_file}")

        print(f"Model name: {self.model.name}")
        try:
            with open(answers_file, "r") as file:
                lines = file.readlines()

            for line in tqdm(lines, desc="Grading answers"):
                try:
                    data = json.loads(line)
                    question = data.get("prompt", "")
                    # Support multiple schemas for answers
                    correct_answer = (
                        data.get("gt_answer")
                        or data.get("correct_answer")
                        or ""
                    )
                    model_response = (
                        data.get("answer")
                        or data.get("model_response")
                        or data.get("full_answer")
                        or ""
                    )

                    if self.grade_method == "simple":
                        grader_answer = None
                        simple_answer = self.simple_grade(correct_answer, model_response)
                    elif self.grade_method == "llm":
                        grading_prompt = (
                            f"Given the question: '{question}'\n"
                            f"The correct answer is: '{correct_answer}'.\n"
                            f"The model responded: '{model_response}'.\n"
                            "Does the model's response correctly answer the question? Please respond with only 'yes' or 'no'."
                        )

                        grader_answer = self.get_llm_response(grading_prompt)
                        simple_answer = self.get_simple_answer(grader_answer)
                    self.results.append(
                        {
                            "question": question,
                            "correct_answer": correct_answer,
                            "model_response": model_response,
                            "grader_answer": grader_answer,
                            "simple_answer": simple_answer,

                        }
                    )

                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON: {e}")
                    continue

        except Exception as e:
            logging.error(f"Error processing answer file: {e}")
            raise

    def save_results(self, grade_result_file):
        os.makedirs(os.path.dirname(grade_result_file), exist_ok=True)
        pd.DataFrame(self.results).to_csv(grade_result_file, index=False)
        # Verify save
        if os.path.exists(grade_result_file):
            file_size = os.path.getsize(grade_result_file)
            logging.info(f"Results saved successfully. File size: {file_size} bytes")
        else:
            logging.error("Failed to save results file!")

    def grade(self):
        answers_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            f"gqa_{self.results_info.strategy}_0.jsonl",
        )
        if self.grade_method == "simple":
            grade_file_name = f"gqa_{self.results_info.strategy}_graded_simple.csv"
        else:
            llm_model_name = re.sub(r"[^a-zA-Z0-9]", "", self.llm_model["model"])
            grade_file_name = (
                f"gqa_{self.results_info.strategy}_graded_{llm_model_name}.csv"
            )
        grade_result_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            grade_file_name,
        )

        try:
            if os.path.isfile(grade_result_file):
                self.results = pd.read_csv(grade_result_file)
                logging.info(f"Results file found: {grade_result_file}")
            else:
                logging.info(f"Processing new results from {answers_file}")
                print(f"\n Testing model: {self.model.name}")
                print(f"Permutation: {'Yes' if self.permutation else 'No'}")
                print(f"Strategy: {self.results_info.strategy}")
                if self.grade_method == "llm":
                    print(f"Grader client: {self.llm_model['type']}")
                    print(f"Grader model: {self.llm_model['model']}")
                self.grade_(answers_file)
                self.save_results(grade_result_file)

            self.show_metrics()
        except Exception as e:
            logging.error(f"Error in grading process: {e}")
            raise

    def show_metrics(self):
        df = pd.DataFrame(self.results)
        print(f"total number of samples: {len(df)}")
        # Map the grader's decision to binary: yes -> 1, no -> 0
        df["pred"] = df["simple_answer"].map({"yes": 1, "no": 0})

        accuracy = df["pred"].mean()
        print("\n------------------------------------------")
        print(f"Strategy: {self.results_info.strategy}")
        print(f"Grade Method: {self.grade_method}")
        print("------------------------------------------")
        print(f"Accuracy: {accuracy:.2f}")
        print("------------------------------------------")

        # Add wandb logging
        if self.use_wandb:
            metrics = {
                "accuracy": accuracy,
            }
            wandb.log(metrics)

        # Save metrics file with proper naming
        if self.grade_method == "simple" or self.llm_model is None:
            metrics_file_name = f"gqa_{self.results_info.strategy}_metrics_simple.txt"
        else:
            llm_model_name = re.sub(r"[^a-zA-Z0-9]", "", self.llm_model["model"])
            metrics_file_name = (
                f"gqa_{self.results_info.strategy}_metrics_{llm_model_name}.txt"
            )
        metrics_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            metrics_file_name,
        )

        # Save metrics to a text file
        with open(metrics_file, "w") as f:
            f.write("\n------------------------------------------\n")
            f.write(f"Strategy: {self.results_info.strategy}\n")
            f.write(f"Grade Method: {self.grade_method}\n")
            f.write("------------------------------------------\n")
            f.write(f"Accuracy: {accuracy:.2f}\n")
            f.write("------------------------------------------\n")
        print(f"Metrics saved to {metrics_file}")

        # Create and log metrics artifact
        if self.use_wandb:
            if self.grade_method == "simple" or self.llm_model is None:
                artifact_name = f"metrics_{self.model.name}_simple"
            else:
                llm_model_name = re.sub(r"[^a-zA-Z0-9]", "", self.llm_model["model"])  
                artifact_name = f"metrics_{self.model.name}_{llm_model_name}"
            metrics_artifact = wandb.Artifact(
                name=artifact_name,
                type="metrics",
            )
            metrics_artifact.add_file(metrics_file)
            wandb.log_artifact(metrics_artifact)
            wandb.finish()
