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

import logging
from evaluation.config import ModelConfig, ResultsConfig

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()


class Grader_Pope:
    def __init__(
        self,
        model: ModelConfig,
        results: ResultsConfig,
        strategy,
        permutation,
        dataset_name="pope",  # Add dataset_name parameter
        **kwargs,
    ):
        self.model = model
        self.results_info = results
        self.strategy = strategy
        self.permutation = permutation
        self.dataset_name = dataset_name  # Store dataset name
        self.results = []
        self.kwargs = kwargs

        # Optional wandb (controlled via results.use_wandb in YAML; default False)
        self.use_wandb = getattr(self.results_info, "use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project="vlm-evaluation",
                group=dataset_name,
                name=f"{dataset_name}_{self.model.name}_{self.strategy}_{self.results_info.unique_id}_{'permutation' if self.permutation else 'normal'}",
                config={
                    "model": self.model.name,
                    "strategy": self.strategy,
                    "permutation": self.permutation,
                    "dataset": dataset_name,
                },
                reinit=True,
            )

    def get_simple_answer(self, text):
        # Only keep the first sentence
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "")
        words = text.split(" ")
        if "No" in words or "not" in words or "no" in words:
            return "no"
        else:
            return "yes"

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

                    simple_answer = self.get_simple_answer(model_response)

                    self.results.append(
                        {
                            "idx": data["idx"],
                            "prompt": prompt,
                            "correct_answer": correct_answer,
                            "model_response": model_response,
                            "simple_model_response": simple_answer,
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
            f"coco_pope_{self.strategy}.jsonl",
        )
        grade_result_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            f"grade_{self.strategy}_result.csv",
        )

        try:
            if os.path.isfile(grade_result_file):
                self.grade_result_df = pd.read_csv(grade_result_file)
                logging.info(f"Results file found: {grade_result_file}")
            else:
                logging.info(f"Processing new results from {answer_file}")
                print(f"\n Testing model: {self.model.name}")
                print(f"Permutation: {'Yes' if self.permutation else 'No'}")
                self.grade_(answer_file)
                self.grade_result_df = pd.DataFrame(self.results)

                # Save results
                logging.info(f"Saving results to {grade_result_file}")
                self.grade_result_df.to_csv(grade_result_file, index=False)

                # Verify save
                if os.path.exists(grade_result_file):
                    file_size = os.path.getsize(grade_result_file)
                    logging.info(
                        f"Results saved successfully. File size: {file_size} bytes"
                    )
                else:
                    logging.error("Failed to save results file!")
            self.show_metrics()

        except Exception as e:
            logging.error(f"Error in grading process: {e}")
            raise

    def show_metrics(self):
        pos = 1
        neg = 0
        self.grade_result_df["correct_answer_num"] = self.grade_result_df[
            "correct_answer"
        ].map({"yes": 1, "no": 0})
        self.grade_result_df["simple_model_response_num"] = self.grade_result_df[
            "simple_model_response"
        ].map({"yes": 1, "no": 0})
        pred_list = self.grade_result_df["simple_model_response_num"].tolist()
        label_list = self.grade_result_df["correct_answer_num"].tolist()

        yes_ratio = pred_list.count(1) / len(pred_list)

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        print("\n------------------------------------------")
        print(f"Test Model: {self.model.name}")
        print(f"Permutation: {'Yes' if self.permutation else 'No'}")
        print("\n------------------------------------------")
        print(f"Strategy: {self.strategy}")
        print("------------------------------------------")
        print("\n\nTP\tFP\tTN\tFN\t")
        print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 score: {f1:.3f}")
        print(f"Yes ratio: {yes_ratio:.3f}")

        # Log metrics to wandb if enabled
        if self.use_wandb:
            metrics = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "yes_ratio": yes_ratio,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
            }
            wandb.log(metrics)

        # Create the metrics text content
        metrics_text = (
            "\n------------------------------------------\n"
            f"\nTest Model: {self.model.name}"
            f"\nPermutation: {'Yes' if self.permutation else 'No'}"
            "\n------------------------------------------\n"
            f"\nStrategy: {self.strategy}"
            "\n------------------------------------------\n"
            f"\nAccuracy: {acc:.3f}"
            f"\nPrecision: {precision:.3f}"
            f"\nRecall: {recall:.3f}"
            f"\nF1 score: {f1:.3f}"
            f"\nYes ratio: {yes_ratio:.3f}"
            "\n------------------------------------------\n"
        )

        # Save metrics to a text file
        metrics_file = os.path.join(
            self.results_info.directory,
            self.model.name,
            str(self.results_info.unique_id),
            "permutation" if self.permutation else "normal",
            f"pope_{self.strategy}_metrics.txt",
        )
        with open(metrics_file, "w") as f:
            f.write(metrics_text)

        # Optionally log the metrics file to wandb
        if self.use_wandb:
            wandb.save(metrics_file)
            metrics_artifact = wandb.Artifact(
                name=f"metrics_{self.model.name}_{self.strategy}",
                type="metrics",
            )
            metrics_artifact.add_file(metrics_file)
            wandb.log_artifact(metrics_artifact)
            print(f"Metrics saved to {metrics_file} and logged to W&B")
        else:
            print(f"Metrics saved to {metrics_file}")
