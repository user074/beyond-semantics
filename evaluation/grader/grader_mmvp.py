import json
import re
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
import openai
import pandas as pd
from tqdm import tqdm
from langchain_ollama import ChatOllama
import wandb
from evaluation.config import ModelConfig, ResultsConfig
import logging
 

logging.getLogger("httpx").setLevel(logging.WARNING)
load_dotenv()


class Grader_MMVP:
    def __init__(
        self,
        model: ModelConfig,
        results: ResultsConfig,
        llm_model=None,
        permutation=False,
        grade_method: str = "simple",  # "simple" or "llm"
        **kwargs,
    ):
        self.model = model
        self.results_info = results
        self.permutation = permutation
        self.llm_model = llm_model
        self.client = None
        self.grade_method = grade_method
        if grade_method == "llm" and self.llm_model is not None:
            self.client = self.get_llm_client(self.llm_model)

        self.NUM_SECONDS_TO_SLEEP = 10
        self.results = []
        self.kwargs = kwargs

        # Optional wandb (controlled via results.use_wandb in YAML; default False)
        self.use_wandb = getattr(self.results_info, "use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project="vlm-evaluation",
                group="mmvp",
                name=f"mmvp_{self.model.name}_{self.results_info.unique_id}_{'permutation' if self.permutation else 'normal'}",
                config={
                    "model": self.model.name,
                    "permutation": self.permutation,
                    "dataset": "mmvp",
                    "llm_model": llm_model,
                },
                reinit=True,
            )


    @staticmethod
    def extract_mcq_answer(text):
        text = text.lower().strip()
        answer_keywords = ["answer is", "answer is:", "answer:"]
        for answer_keyword in answer_keywords:
            if answer_keyword in text:
                text = text.split(answer_keyword)[-1]
        text = text.strip().rstrip('.:,').lstrip('(').rstrip(')')
        return text



    def get_llm_client(self, llm_model):
        client = None
        if llm_model["type"] == "gpt":
            client = OpenAI(api_key=os.getenv("API"))
        elif llm_model["type"] == "ollama":
            client = ChatOllama(
                model=llm_model["model"],
                temperature=0.2,
                num_gpu=llm_model["num_gpu"],
            )
        else:
            print("Warning: You have to provide the llm type, like 'gpt' or 'ollama'!")

        print("----------------------------------------")
        print(f"Client: {llm_model['type']}")
        print(f"Model: {llm_model['model']}")
        print("----------------------------------------")

        return client

    def get_simple_answer(self, answer):
        yes_no_regex = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
        match = yes_no_regex.search(answer)
        if match:
            # return match.group(1).lower()
            return answer.lower()
        else:
            logging.warning(f"Unexpected API response: {answer}. Defaulting to 'no'")
            return "Could not determine yes or no."

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

    def get_yes_no_answer_gpt(self, question):
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model["model"],
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no.",
                        },
                        {
                            "role": "user",
                            "content": question,
                        },
                    ],
                    temperature=0.2,
                )

                # answer = completion.choices[0].message.content.strip().lower()
                answer = completion.choices[0].message.content
                return answer

            except openai.APIError as e:
                logging.error(f"OpenAI API returned an API Error: {e}")
                time.sleep(self.NUM_SECONDS_TO_SLEEP)
            except openai.APIConnectionError as e:
                logging.error(f"Failed to connect to OpenAI API: {e}")
                time.sleep(self.NUM_SECONDS_TO_SLEEP)
            except openai.RateLimitError as e:
                logging.error(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(self.NUM_SECONDS_TO_SLEEP * 2)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                time.sleep(self.NUM_SECONDS_TO_SLEEP)

            retry_count += 1

        logging.error("Max retries reached for API call. Defaulting to 'no'")
        return f"No response from {self.llm_model['model']}"

    def normalize_answer(self, answer):
        if not isinstance(answer, str):
            return ""
        return answer.strip().lower().rstrip('.')

    def extract_option_letter(self, text):
        if not isinstance(text, str):
            return None
        match = re.search(r"\b([A-Da-d])\b", text)
        if match:
            return match.group(1).upper()
        return None

    def simple_grade(self, correct_answer, model_response):
        ca = self.normalize_answer(correct_answer)
        mr = self.normalize_answer(model_response)
        # If the correct answer is a single option letter, compare letters
        if len(ca) == 1 and ca in ["a", "b", "c", "d"]:
            mr_letter = self.extract_option_letter(model_response)
            return "yes" if (mr_letter == ca.upper()) else "no"
        # Fallback to direct equality or substring containment
        if mr == ca:
            return "yes"
        if ca and ca in mr:
            return "yes"
        return "no"

    def grade_(self, answer_file):
        if not os.path.exists(answer_file):
            raise FileNotFoundError(f"Answer file not found: {answer_file}")

        try:
            with open(answer_file, "r") as file:
                lines = file.readlines()

            for line in tqdm(lines, desc="Processing answers"):
                try:
                    data = json.loads(line)
                    prompt = data.get("prompt", "")
                    model_response = data.get("answer") or data.get("model_response", "")
                    text_options = [x.lower() for x in data.get("text_options", [])]
                    gt_raw = str(data.get("gt_answer", "")).lower()
                    # Extract ground-truth letter from formats like "(a)" or "a"
                    gt_letter = None
                    for ch in gt_raw:
                        if ch in ["a", "b", "c", "d"]:
                            gt_letter = ch
                            break
                    text_answer = (
                        text_options[ord(gt_letter) - ord("a")] if gt_letter and len(text_options) >= (ord(gt_letter) - ord("a") + 1) else ""
                    )

                    if self.grade_method == "simple":
                        extracted = self.extract_mcq_answer(model_response)
                        is_correct = (extracted == gt_letter) or (extracted == text_answer)
                        grader_answer = None
                        simple_answer = "yes" if is_correct else "no"
                    else:
                        # LLM-based grading prompt includes both letter and text answer when available
                        correct_desc = gt_letter.upper() if gt_letter else ""
                        if text_answer:
                            correct_desc = f"{correct_desc} ({text_answer})" if correct_desc else text_answer
                        question4gpt = (
                            f"Given the following question: '{prompt}', "
                            f"the correct answer is: {correct_desc}.\n"
                            f"Does the following answer correctly answer the question? Answer: {model_response}"
                        )
                        if self.llm_model["type"] == "gpt":
                            grader_answer = self.get_yes_no_answer_gpt(question4gpt)
                        elif self.llm_model["type"] == "ollama":
                            grader_answer = self.get_yes_no_answer_ollama(question4gpt)
                        else:
                            grader_answer = ""
                        simple_answer = self.get_simple_answer(grader_answer)

                    self.results.append(
                        {
                            "question_id": data.get("question_id"),
                            "prompt": prompt,
                            "correct_answer": gt_letter or gt_raw,
                            "model_response": model_response,
                            "grader_answer": grader_answer,
                            "simple_answer": simple_answer,
                        }
                    )

                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON line: {e}")
                    continue

        except Exception as e:
            logging.error(f"Error processing answer file: {e}")
            raise

    def grade(self):
        answer_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            "mmvp_result_0.jsonl",
        )
        if self.grade_method == "simple":
            grade_file_name = "mmvp_grade_result_simple.csv"
        else:
            llm_model_name = re.sub(r"[^a-zA-Z0-9]", "", self.llm_model["model"])  
            grade_file_name = f"mmvp_grade_result_{llm_model_name}.csv"
        grade_result_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            grade_file_name,
        )

        try:
            print(f"\n Testing model: {self.model.name}")
            print(f"Permutation: {'Yes' if self.permutation else 'No'}")
            if self.grade_method == "llm" and self.llm_model is not None:
                print(f"Grader client: {self.llm_model['type']}")
                print(f"Grader model: {self.llm_model['model']}")

            # Unified flow: (re)compute results if CSV missing; rely on grade_ for both modes
            if os.path.isfile(grade_result_file):
                self.results = pd.read_csv(grade_result_file)
                logging.info(
                    f"Loading existing graded results from {grade_result_file}"
                )
            else:
                logging.info(f"Processing new results from {answer_file}")
                self.grade_(answer_file)
                self.grade_result_df = pd.DataFrame(self.results)
                os.makedirs(os.path.dirname(grade_result_file), exist_ok=True)
                self.grade_result_df.to_csv(grade_result_file, index=False)

            self.show_metrics()

        except Exception as e:
            logging.error(f"Error in grading process: {e}")
            raise

    def show_metrics(self):
        if self.grade_method == "simple" and hasattr(self, "simple_total_lines"):
            num_total = self.simple_total_lines // 2
            num_correct = self.simple_correct_pairs
            accuracy = num_correct / num_total if num_total > 0 else 0
        else:
            df = pd.DataFrame(self.results)
            print(f"total number of samples: {len(df)}")

            # Initialize counters for pair-wise evaluation
            num_correct = 0
            num_total = 0

            # Process results in pairs
            for i in range(0, len(df), 2):
                if i + 1 >= len(df):  # Skip incomplete pair at the end
                    break

                # Get pair of answers
                first_correct = df.iloc[i]["simple_answer"] == "yes"
                second_correct = df.iloc[i + 1]["simple_answer"] == "yes"

                # Increment counters
                if first_correct and second_correct:
                    num_correct += 1
                num_total += 1

            # Calculate accuracy based on pairs
            accuracy = num_correct / num_total if num_total > 0 else 0

        print("\n=== MMVP Dataset Evaluation Results ===")
        print("\n------------------------------------------")
        print(f"Test Model: {self.model.name}")
        if self.grade_method == "llm" and self.llm_model is not None:
            print(f"Grader client: {self.llm_model['type']}")
            print(f"Grader model: {self.llm_model['model']}")
        print(f"Permutation: {'Yes' if self.permutation else 'No'}")
        print("------------------------------------------")
        print(f"Pairs evaluated: {num_total}")
        print(f"Correct pairs: {num_correct}")
        print(f"Pair-wise Accuracy: {accuracy:.2f}")
        print("------------------------------------------")

        # Add wandb logging
        if self.use_wandb:
            metrics = {
                "accuracy": accuracy,
                "num_correct_pairs": num_correct,
                "total_pairs": num_total,
            }
            wandb.log(metrics)

        # Save metrics file with LLM model name
        if self.grade_method == "simple":
            metrics_file_name = "mmvp_metrics_simple.txt"
        else:
            llm_model_name = re.sub(r"[^a-zA-Z0-9]", "", self.llm_model["model"])
            metrics_file_name = f"mmvp_metrics_{llm_model_name}.txt"
        metrics_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            metrics_file_name,
        )

        # Save metrics to a text file
        with open(metrics_file, "w") as f:
            f.write("\n=== MMVP Dataset Evaluation Results ===")
            f.write("\n------------------------------------------\n")
            f.write(f"\nTest Model: {self.model.name}\n")
            if self.grade_method == "llm" and self.llm_model is not None:
                f.write(f"Grader client: {self.llm_model['type']}\n")
                f.write(f"Grader model: {self.llm_model['model']}\n")
            else:
                f.write(f"Grader client: simple\n")
            f.write(f"Permutation: {'Yes' if self.permutation else 'No'}\n")
            f.write("------------------------------------------\n")
            f.write(f"Pairs evaluated: {num_total}\n")
            f.write(f"Correct pairs: {num_correct}\n")
            f.write(f"Pair-wise Accuracy: {accuracy:.2f}\n")
            f.write("------------------------------------------\n")
        print(f"Metrics saved to {metrics_file}")

        if self.use_wandb:
            if self.grade_method == "simple":
                artifact_name = f"metrics_{self.model.name}_simple"
            else:
                artifact_name = f"metrics_{self.model.name}_{llm_model_name}"
            metrics_artifact = wandb.Artifact(
                name=artifact_name,
                type="metrics",
            )
            metrics_artifact.add_file(metrics_file)
            wandb.log_artifact(metrics_artifact)
            wandb.finish()

