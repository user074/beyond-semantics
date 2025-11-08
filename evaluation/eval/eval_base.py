import math
import json


class Eval_Base:
    def __init__(self):
        pass

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def read_jsonl(self, file_path):
        """Reads a JSONL file and returns a list of dictionaries."""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def read_json(self, file_path):
        """Reads a JSON file and returns a dictionary."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
