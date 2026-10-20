from pipeline.types.budataset import BUDataset, BreakingUpdateSample, BCType, load_budataset_from_jsonl
import json 
from pathlib import Path
import argparse


def main(args):
    butrainset_path = Path("/home/xchen6/breaking_updates_rl/data/sft/sft_data_train_build_success.jsonl")
    output_dir = Path("/home/xchen6/breaking_updates_rl/experiment/")
    # Load BUTrainset
    butrainset = load_budataset_from_jsonl(butrainset_path, train=True)
    # split by bc type
    if args.by_type:
        if args.bc is None:
            print("Please specify a breaking change type with -b or --bc")
            return
        bc_type = BCType(args.bc)
        
        split = butrainset.split_by_bc_type_file_level(bc_type, shuffle=True, seed=42)
        # save each split to a jsonl file
        output_file = output_dir / f"train_{bc_type.value}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join([json.dumps(item) for item in split]), encoding="utf-8")   
        print(f"Saved split by breaking change type {bc_type.value} with {len(split)} samples to {output_file}")         
    # split by size
    else:
        if args.n_split is None:
            print("Please specify the number of splits with -n or --n_split")
            return
        splits = butrainset.split_by_size_file_level(args.n_split, shuffle=True, seed=42)
        # save each split to a jsonl file
        for i, split_data in enumerate(splits):
            output_file = output_dir / f"train_{len(splits)}_splits_{i+1}.jsonl"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text("\n".join([json.dumps(item) for item in split_data]), encoding="utf-8")
            print(f"Saved split {i+1} with {len(split_data)} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--by_type", action="store_true", help="Whether to split by breaking change type")
    parser.add_argument("-b", "--bc", type=str, default=None, help="Breaking change type to split by")
    parser.add_argument("-n", "--n_split", type=int, default=None, help="Number of splits")
    args = parser.parse_args()
    
    main(args)