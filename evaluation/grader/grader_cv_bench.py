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


class Grader_CV_Bench:
    def __init__(
        self,
        model: ModelConfig,
        results: ResultsConfig,
        llm_model,
        permutation=False,
        **kwargs,
    ):
        self.model = model
        self.results_info = results
        self.permutation = permutation
        self.llm_model = llm_model
        self.client = self.get_llm_client(self.llm_model)

        self.NUM_SECONDS_TO_SLEEP = 10
        self.results = []
        self.num_total = 0
        self.num_correct = 0
        self.kwargs = kwargs

        # Optional wandb (controlled via results.use_wandb in YAML; default False)
        self.use_wandb = getattr(self.results_info, "use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project="vlm-evaluation",
                group="cv_bench",
                name=f"cv_bench_{self.model.name}_{self.results_info.unique_id}_{'permutation' if self.permutation else 'normal'}",
                config={
                    "model": self.model.name,
                    "permutation": self.permutation,
                    "dataset": "cv_bench",
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

    def grade_(self, answer_file):
        if not os.path.exists(answer_file):
            raise FileNotFoundError(f"Answer file not found: {answer_file}")

        try:
            with open(answer_file, "r") as file:
                lines = file.readlines()

            for line in tqdm(lines, desc="Processing answers"):
                try:
                    data = json.loads(line)
                    prompt = data["prompt"]
                    correct_answer = data["correct_answer"]
                    model_response = data["model_response"]

                    question4gpt = (
                        f"Given the question prompt:\n{prompt}\n"
                        f"The correct answer is: {correct_answer}.\n"
                        f"Does the model's response '{model_response}' correctly answer the question? "
                        f"Please respond with only 'yes' or 'no'."
                    )

                    if self.llm_model["type"] == "gpt":
                        answer = self.get_yes_no_answer_gpt(question4gpt)
                    elif self.llm_model["type"] == "ollama":
                        answer = self.get_yes_no_answer_ollama(question4gpt)
                    else:
                        print(
                            "Warning: You have to provide the llm type, like 'gpt' or 'ollama'!"
                        )

                    simple_answer = self.get_simple_answer(answer)

                    # Update counters
                    self.num_total += 1
                    if simple_answer == "yes":
                        self.num_correct += 1

                    self.results.append(
                        {
                            "type": data.get("type", "unknown"),
                            "task": data.get("task", "unknown"),
                            "source": data.get("source", "unknown"),
                            "prompt": prompt,
                            "correct_answer": correct_answer,
                            "model_response": model_response,
                            "answer": answer,
                            "simple_answer": simple_answer,
                        }
                    )

                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON line: {e}")
                    continue
                except KeyError as e:
                    logging.error(f"Missing required field in data: {e}")
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
            "result.jsonl",
        )
        # add the llm model name to the file name, but remove special characters
        llm_model_name = re.sub(r"[^a-zA-Z0-9]", "", self.llm_model["model"])
        grade_result_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            f"grade_result_{llm_model_name}.csv",
        )

        try:
            # First check if grade_result_file exists
            if os.path.isfile(grade_result_file):
                logging.info(
                    f"Loading existing graded results from {grade_result_file}"
                )
                self.grade_result_df = pd.read_csv(grade_result_file)
            else:
                # Check if answer file exists before grading
                if not os.path.isfile(answer_file):
                    raise FileNotFoundError(f"Answer file not found: {answer_file}")

                logging.info(f"Processing new results from {answer_file}")

                print(f"\n Testing model: {self.model.name}")
                print(f"Permutation: {'Yes' if self.permutation else 'No'}")
                print(f"Grader client: {self.llm_model['type']}")
                print(f"Grader model: {self.llm_model['model']}")

                self.grade_(answer_file)
                self.grade_result_df = pd.DataFrame(self.results)

                # Save results
                logging.info(f"Saving results to {grade_result_file}")
                os.makedirs(os.path.dirname(grade_result_file), exist_ok=True)
                self.grade_result_df.to_csv(grade_result_file, index=False)

                # Verify save
                if os.path.exists(grade_result_file):
                    file_size = os.path.getsize(grade_result_file)
                    logging.info(
                        f"Results saved successfully. File size: {file_size} bytes"
                    )
                else:
                    logging.error("Failed to save results file!")

            # Split results into 2D and 3D for metrics calculation
            self.grade_result_df_2d = self.grade_result_df[
                self.grade_result_df["type"] == "2D"
            ]
            self.grade_result_df_3d = self.grade_result_df[
                self.grade_result_df["type"] == "3D"
            ]

            self.show_metrics()

        except Exception as e:
            logging.error(f"Error in grading process: {e}")
            raise

    def show_metrics(self):
        # Prepare metrics text
        metrics_text = []

        # Show the test model name
        metrics_text.append(f"\n\nTest Model: {self.model.name}")
        metrics_text.append(f"Permutation: {'Yes' if self.permutation else 'No'}")

        # Regular Accuracy section
        metrics_text.append("\n\nRegular Accuracy: ")
        metrics_text.append("----------------------------------------")
        metrics_text.append(
            f"Overall accuracy: {self.accuracy(self.grade_result_df):.4f}"
        )
        metrics_text.append(
            f"2D accuracy: {self.accuracy(self.grade_result_df_2d):.4f}"
        )
        metrics_text.append(
            f"3D accuracy: {self.accuracy(self.grade_result_df_3d):.4f}"
        )
        metrics_text.append("----------------------------------------")

        # Task Accuracies section
        metrics_text.append("\n\nTask Accuracies:")
        metrics_text.append("----------------------------------------")
        metrics_text.append(
            f"Count accuracy: {self.calculate_accuracy_from_task(self.grade_result_df, 'Count'):.4f}"
        )
        metrics_text.append(
            f"Relation accuracy: {self.calculate_accuracy_from_task(self.grade_result_df, 'Relation'):.4f}"
        )
        metrics_text.append(
            f"Depth accuracy: {self.calculate_accuracy_from_task(self.grade_result_df, 'Depth'):.4f}"
        )
        metrics_text.append(
            f"Distance accuracy: {self.calculate_accuracy_from_task(self.grade_result_df, 'Distance'):.4f}"
        )
        metrics_text.append("----------------------------------------")

        # Source Accuracies section
        metrics_text.append("\n\nSource Accuracies:")
        metrics_text.append("----------------------------------------")
        accuracy_2d_ade = self.calculate_accuracy_from_source(
            self.grade_result_df, "ADE20K"
        )
        accuracy_2d_coco = self.calculate_accuracy_from_source(
            self.grade_result_df, "COCO"
        )
        accuracy_3d_omni = self.calculate_accuracy_from_source(
            self.grade_result_df, "Omni3D"
        )

        metrics_text.append(f"ADE20K Accuracy: {accuracy_2d_ade:.4f}")
        metrics_text.append(f"COCO Accuracy: {accuracy_2d_coco:.4f}")
        metrics_text.append(f"Omni3D Accuracy: {accuracy_3d_omni:.4f}")
        metrics_text.append("----------------------------------------")

        # Print metrics to console
        print("\n".join(metrics_text))

        # Save metrics to file
        llm_model_name = re.sub(r"[^a-zA-Z0-9]", "", self.llm_model["model"])
        metrics_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            f"cv_bench_metrics_{llm_model_name}.txt",
        )

        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            f.write("\n".join(metrics_text))

        print(f"\nMetrics saved to {metrics_file}")

        # Add wandb logging
        if self.use_wandb:
            metrics = {
                "overall_accuracy": self.accuracy(self.grade_result_df),
                "accuracy_2d": self.accuracy(self.grade_result_df_2d),
                "accuracy_3d": self.accuracy(self.grade_result_df_3d),
                "accuracy_count": self.calculate_accuracy_from_task(
                    self.grade_result_df, "Count"
                ),
                "accuracy_relation": self.calculate_accuracy_from_task(
                    self.grade_result_df, "Relation"
                ),
                "accuracy_depth": self.calculate_accuracy_from_task(
                    self.grade_result_df, "Depth"
                ),
                "accuracy_distance": self.calculate_accuracy_from_task(
                    self.grade_result_df, "Distance"
                ),
                "accuracy_ade20k": accuracy_2d_ade,
                "accuracy_coco": accuracy_2d_coco,
                "accuracy_omni3d": accuracy_3d_omni,
            }
            wandb.log(metrics)

        if self.use_wandb:
            metrics_artifact = wandb.Artifact(
                name=f"metrics_{self.model.name}_{llm_model_name}",
                type="metrics",
            )
            metrics_artifact.add_file(metrics_file)
            wandb.log_artifact(metrics_artifact)
            wandb.finish()

    def accuracy(self, df):
        total_number = df.shape[0]
        correct_number = df[df["simple_answer"] == "yes"].shape[0]

        return correct_number / total_number

    # Define a function to calculate accuracy for a given source
    def calculate_accuracy_from_source(self, df, source):
        source_df = df[df["source"] == source].copy()
        source_df["result"] = source_df["simple_answer"].map({"yes": 1, "no": 0})
        accuracy = source_df[
            "result"
        ].mean()  # Assuming 'result' is 1 for correct and 0 for incorrect
        return accuracy

    # Define a function to calculate accuracy for a given task
    def calculate_accuracy_from_task(self, df, task):
        task_df = df[df["task"] == task].copy()
        task_df["result"] = task_df["simple_answer"].map({"yes": 1, "no": 0})
        accuracy = task_df[
            "result"
        ].mean()  # Assuming 'result' is 1 for correct and 0 for incorrect
        return accuracy
