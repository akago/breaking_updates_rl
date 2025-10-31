from datasets import load_dataset
from collections import defaultdict
import unittest


class TestDataLoading(unittest.TestCase):
    @unittest.skip("Skipping")
    def test_data_loding(self):
        DATASET_PATH = "/home/xchen6/breaking_updates_rl/data/prompts/dataset.json"
        dataset = load_dataset("json", data_files=DATASET_PATH)
        self.assertIn("train", dataset)
        self.assertGreater(len(dataset["train"]), 383)
        first_sample = dataset["train"][0]["data"]
        self.assertIn("project", first_sample)
        self.assertIn("absolute_path_to_file_in_container", first_sample)
        
        self.assertIn("test", dataset)
        self.assertGreater(len(dataset["test"]), 0)
        
    def test_data_1(self):
        DATASET_PATH = "/home/xchen6/breaking_updates_rl/data/prompts/dataset.json"
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
        proj_to_indices = defaultdict(list)
        for i, ex in enumerate(dataset):
            proj = ex.get("project")
            if proj is None:
                raise ValueError("样本缺少 'project' 字段。")
            data = ex.get("data")
            if not isinstance(data, dict) or "errors" not in data or not isinstance(data["errors"], list):
                raise ValueError("样本缺少 'data.errors'（应为 list）。")
            proj_to_indices[proj].append(i)
   
   
        
if __name__ == "__main__":
    unittest.main()