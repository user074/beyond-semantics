import json
import os
import re
import pandas as pd
from tqdm import tqdm
import logging
from dotenv import load_dotenv
from openai import OpenAI
import time
from langchain_ollama import ChatOllama
import wandb
from evaluation.config import ModelConfig, ResultsConfig

logging.getLogger("httpx").setLevel(logging.WARNING)
load_dotenv()


class GraderSynthetic:
    def __init__(
        self,
        model: ModelConfig,
        results: ResultsConfig,
        permutation,
        llm_model,
        dataset_name="synthetic",
        **kwargs,
    ):
        self.model = model
        self.results_info = results
        self.permutation = permutation
        self.llm_model = llm_model
        self.client = self.get_llm_client(self.llm_model)
        self.dataset_name = dataset_name

        self.NUM_SECONDS_TO_SLEEP = 10
        self.results = []
        self.kwargs = kwargs

        # Optional wandb (controlled via results.use_wandb in YAML; default False)
        self.use_wandb = getattr(self.results_info, "use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project="vlm-evaluation",
                group=dataset_name,
                name=f"{dataset_name}_{self.model.name}_{self.results_info.unique_id}_{'permutation' if self.permutation else 'normal'}",
                config={
                    "model": self.model.name,
                    "permutation": self.permutation,
                    "dataset": dataset_name,
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
            )
        else:
            print("Warning: You have to provide the llm type, like 'gpt' or 'ollama'!")

        print("----------------------------------------")
        print(f"Client: {llm_model['type']}")
        print(f"Model: {llm_model['model']}")
        print(f"Test model: {self.model.name}")
        print("----------------------------------------")

        return client

    def get_simple_answer(self, answer):
        if answer in ["yes", "no"]:
            return answer
        else:
            yes_no_regex = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
            match = yes_no_regex.search(answer)
            if match:
                return match.group(1).lower()
            else:
                logging.warning(
                    f"Unexpected API response: {answer}. Defaulting to 'no'"
                )
                return "no"

    def get_yes_no_answer_ollama(self, question):
        messages = [
            (
                "system",
                "You are a helpful and precise assistant for checking the quality of the answer. Please answer with exactly 'yes' or 'no' only.",
            ),
            ("human", question),
        ]
        answer = self.client.invoke(messages).content
        return answer.lower()

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
                            "content": "You are a helpful and precise assistant for checking the quality of the answer. Please answer with exactly 'yes' or 'no' only.",
                        },
                        {
                            "role": "user",
                            "content": question,
                        },
                    ],
                    temperature=0.2,
                )
                return completion.choices[0].message.content.strip().lower()

            except Exception as e:
                logging.error(f"Error in GPT API call: {e}")
                time.sleep(self.NUM_SECONDS_TO_SLEEP)
                retry_count += 1

        logging.error("Max retries reached for API call. Defaulting to 'no'")
        return f"No response from {self.llm_model['model']}"

    def grade_(self, answer_file):
        if not os.path.exists(answer_file):
            raise FileNotFoundError(f"Answer file not found: {answer_file}")

        try:
            with open(answer_file, "r") as file:
                lines = file.readlines()

            for line in tqdm(lines, desc="Processing answers"):
                try:
                    data = json.loads(line)
                    question = data["question"]
                    correct_answer = data["correct_answer"]
                    model_response = data["model_response"]

                    question4gpt = (
                        f"Given the question:\n{question}\n"
                        f"The correct answer is: {correct_answer}\n"
                        f"Does the model's response '{model_response}' correctly answer the question? "
                        f"Please respond with only 'yes' or 'no'."
                    )

                    if self.llm_model["type"] == "gpt":
                        answer = self.get_yes_no_answer_gpt(question4gpt)
                    elif self.llm_model["type"] == "ollama":
                        answer = self.get_yes_no_answer_ollama(question4gpt)

                    simple_answer = self.get_simple_answer(answer)

                    self.results.append(
                        {
                            "entry_id": data["entry_id"],
                            "test_category": data["test_category"],
                            "qa_category": data["qa_category"],
                            "question": question,
                            "correct_answer": correct_answer,
                            "model_response": model_response,
                            "llm_answer": answer,
                            "is_correct": simple_answer == "yes",
                        }
                    )

                except (json.JSONDecodeError, KeyError) as e:
                    logging.error(f"Error processing line: {e}")
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
            "synthetic_results.jsonl",
        )
        # add the llm model name to the file name, but remove special characters
        llm_model_name = re.sub(r"[^a-zA-Z0-9]", "", self.llm_model["model"])
        grade_result_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            f"synthetic_grade_result_{llm_model_name}.csv",
        )

        try:
            if os.path.isfile(grade_result_file):
                self.grade_result_df = pd.read_csv(grade_result_file)
                logging.info(
                    f"Loading existing graded results from {grade_result_file}"
                )
            else:
                logging.info(f"Processing new results from {answer_file}")
                print(f"\n Testing model: {self.model.name}")
                print(f"Permutation: {'Yes' if self.permutation else 'No'}")
                print(f"Grader client: {self.llm_model['type']}")
                print(f"Grader model: {self.llm_model['model']}")
                self.grade_(answer_file)
                self.grade_result_df = pd.DataFrame(self.results)

                os.makedirs(os.path.dirname(grade_result_file), exist_ok=True)
                self.grade_result_df.to_csv(grade_result_file, index=False)

            self.show_metrics()

        except Exception as e:
            logging.error(f"Error in grading process: {e}")
            raise

    def show_metrics(self):
        """Display and save evaluation metrics"""
        # Calculate metrics
        overall_accuracy = self.grade_result_df["is_correct"].mean()

        # Prepare metrics text
        metrics_text = []

        metrics_text.append(f"\n\nTest Model: {self.model.name}")
        metrics_text.append(f"Grader client: {self.llm_model['type']}")
        metrics_text.append(f"Grader model: {self.llm_model['model']}")
        metrics_text.append(f"Permutation: {'Yes' if self.permutation else 'No'}")

        metrics_text.append("\n=== Synthetic Dataset Evaluation Results ===")
        metrics_text.append("-------------------------------------------")
        metrics_text.append(f"\nOverall Accuracy: {overall_accuracy:.4f}")

        # Accuracy by QA category
        metrics_text.append("\nAccuracy by Question Category:")
        metrics_text.append("-------------------------------------------")
        qa_categories = sorted(self.grade_result_df["qa_category"].unique())
        for category in qa_categories:
            category_df = self.grade_result_df[
                self.grade_result_df["qa_category"] == category
            ]
            accuracy = category_df["is_correct"].mean()
            metrics_text.append(f"{category.capitalize()} Accuracy: {accuracy:.4f}")

        # Accuracy by test category
        metrics_text.append("\nAccuracy by Test Category:")
        metrics_text.append("-------------------------------------------")
        test_categories = sorted(self.grade_result_df["test_category"].unique())
        for category in test_categories:
            category_df = self.grade_result_df[
                self.grade_result_df["test_category"] == category
            ]
            accuracy = category_df["is_correct"].mean()
            metrics_text.append(f"{category.capitalize()} Accuracy: {accuracy:.4f}")

        metrics_text.append("-------------------------------------------\n")

        # Print metrics to console
        print("\n".join(metrics_text))

        llm_model_name = re.sub(r"[^a-zA-Z0-9]", "", self.llm_model["model"])
        # Save metrics to file
        metrics_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            f"synthetic_metrics_{llm_model_name}.txt",
        )

        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            f.write("\n".join(metrics_text))

        print(f"\nMetrics saved to {metrics_file}")

        # Add wandb logging
        if self.use_wandb:
            metrics = {
                "overall_accuracy": overall_accuracy,
            }
            # Add category-wise metrics
            for category in qa_categories:
                category_df = self.grade_result_df[
                    self.grade_result_df["qa_category"] == category
                ]
                accuracy = category_df["is_correct"].mean()
                metrics[f"qa_category_{category}_accuracy"] = accuracy
            for category in test_categories:
                category_df = self.grade_result_df[
                    self.grade_result_df["test_category"] == category
                ]
                accuracy = category_df["is_correct"].mean()
                metrics[f"test_category_{category}_accuracy"] = accuracy
            wandb.log(metrics)

        if self.use_wandb:
            wandb.save(metrics_file)
            metrics_artifact = wandb.Artifact(
                name=f"metrics_{self.model.name}_{llm_model_name}",
                type="metrics",
            )
            metrics_artifact.add_file(metrics_file)
            wandb.log_artifact(metrics_artifact)
            print(f"Metrics saved to {metrics_file} and logged to W&B")
            wandb.finish()
