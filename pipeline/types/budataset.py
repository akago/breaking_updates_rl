from __future__ import annotations
import json
from random import random
from datasets import load_dataset
from pathlib import Path
import pandas as pd
from enum import Enum
import random
import logging

class BCType(Enum):
    """Enum for breaking change types."""
    METHOD_REMOVED = "METHOD_REMOVED"
    TYPE_REMOVED = "TYPE_REMOVED"
    SUPERTYPE_REMOVED = "SUPERTYPE_REMOVED"
    METHOD_RETURN_TYPE_CHANGED = "METHOD_RETURN_TYPE_CHANGED"
    FIELD_REMOVED = "FIELD_REMOVED"
    FIELD_TYPE_CHANGED = "FIELD_TYPE_CHANGED"
    METHOD_ADDED_TO_INTERFACE = "METHOD_ADDED_TO_INTERFACE"
    METHOD_NO_LONGER_THROWS_CHECKED_EXCEPTION = "METHOD_NO_LONGER_THROWS_CHECKED_EXCEPTION"
    METHOD_ABSTRACT_ADDED_TO_CLASS = "METHOD_ABSTRACT_ADDED_TO_CLASS"
    METHOD_NOW_FINAL = "METHOD_NOW_FINAL"
    METHOD_NOW_STATIC = "METHOD_NOW_STATIC"
    CLASS_NOW_ABSTRACT = "CLASS_NOW_ABSTRACT"
    CLASS_NOW_FINAL = "CLASS_NOW_FINAL"
    

class BreakingUpdateSample:
    """A single breaking update sample. Including multiple buggy files."""
    def __init__(self, breaking_commit, buggy_files: list[dict]):
        self._buggy_files = buggy_files
        self.breaking_commit = breaking_commit
        
    def flatten(self) -> list[dict]:
        """Flattens the sample into a list of buggy files"""
        return self._buggy_files
    
    def __len__(self):
        return len(self._buggy_files)
    
    def get_bc_types_project_level(self) -> list[BCType]:
        kinds_set = set()
        for buggy_file in self._buggy_files:
            kinds_list = set(self.get_bc_types_file_level(buggy_file))
            kinds_set.update(kinds_list)
        kinds_unique = sorted(list(kinds_set))
        return kinds_unique
    
    @staticmethod
    def get_bc_types_file_level(buggy_file: dict) -> list[BCType]:
        kinds_set = set()
        errors = buggy_file.get("errors", [])
        if isinstance(errors, list):
            for error in errors:
                if isinstance(error, dict):
                    bcs = error.get("BCs", [])
                    if isinstance(bcs, list):
                        for bc in bcs:
                            if isinstance(bc, dict):
                                kind = bc.get("kind", "")
                                if kind:
                                    try:
                                       kinds_set.add(kind)
                                    except ValueError:
                                        logging.warning(f"Unknown BCType: {kind}")
        kinds_unique = sorted(list(kinds_set))
        return [BCType(kind) for kind in kinds_unique]
        
class BUDataset:
    """Base class for datasets used in the pipeline."""

    def __init__(self, name: str, data: list[BreakingUpdateSample]):
        self.name = name
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]    
    
    def split_by_bc_type(self, bc_type: BCType) -> BUDataset:
        """Splits the dataset by breaking change type."""
        return BUDataset(name=f"{self.name}_{bc_type.value}", data=[item for item in self.data if bc_type in item.get_bc_types_project_level()])
    
    def split_by_size_project_level(self, n_split:int)-> list[BUDataset]:
        """Splits the dataset into n_split parts."""
        split_size = len(self.data) // n_split
        splits = []
        for i in range(n_split):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_split - 1 else len(self.data)
            splits.append(BUDataset(name=f"{self.name}_part{i+1}", data=self.data[start_idx:end_idx]))
        return splits
    
    def flatten(self)->list[dict]:
        """Flattens the dataset into a list of buggy files."""
        flattened = []
        for sample in self.data:
            flattened.extend(sample.flatten())
        return flattened

class BUTrainingset(BUDataset):
    """Dataset class for training datasets."""
    def __init__(self, name: str, data: list[BreakingUpdateSample]):
        super().__init__(name, data)
        
    def split_by_size_file_level(self, n_split:int, shuffle:bool=False, seed:int=42)-> list[list[dict]]:
        """Splits the dataset into n_split parts at file level."""
        flattened_data = self.flatten()
        if shuffle:
            random.seed(seed)
            random.shuffle(flattened_data)
        split_size = len(flattened_data) // n_split
        splits = []
        for i in range(n_split):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_split - 1 else len(flattened_data)
            split_data = flattened_data[start_idx:end_idx]
            splits.append(split_data)
        return splits
    
    def split_by_bc_type_file_level(self, bc_type:BCType, shuffle:bool=False, seed:int=42) -> list[dict]:
        """Splits the dataset into parts by breaking change type at file level."""
        flattened_data = self.flatten()
        # TODO: filter by bc_type
        filtered_data = [item for item in flattened_data if bc_type in BreakingUpdateSample.get_bc_types_file_level(item)]
        if shuffle:
            random.seed(seed)
            random.shuffle(filtered_data)
        return filtered_data
    
  
def load_budataset_from_jsonl(file_path: Path, train: bool) -> BUDataset:
    """Loads a BUDataset from a JSONL file."""
    data = []
    # prompt-patch pairs as a list, file level
    all_buggy_files = load_dataset("json", data_files=str(file_path), split="train")
    
    # aggragate by breaking commit
    bu_dict = {}
    for item in all_buggy_files:
        bu_dict.setdefault(item["breakingCommit"], []).append(item)
    for breaking_commit, buggy_files in bu_dict.items():
        data.append(BreakingUpdateSample(breaking_commit, buggy_files))
    
    if train:
        return BUTrainingset(name=file_path.stem, data=data)
    return BUDataset(name=file_path.stem, data=data)