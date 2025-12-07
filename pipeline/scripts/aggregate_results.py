from pathlib import Path
import json


llama8b_offtheshelf = Path("/home/xchen6/breaking_updates_rl/results/unsloth/llama3.1-8b-diff")
gemma12b_offtheshelf = Path("/home/xchen6/breaking_updates_rl/results/unsloth/gemma3-12b-diff")
gemma4b_offtheshelf = Path("/home/xchen6/breaking_updates_rl/results/unsloth/gemma3-4b-diff")

# rl dense
# llama8b_rl_dense = Path("/home/xchen6/breaking_updates_rl/results/rl/llama8b_dense/checkpoint-300_20251114-111717")
llama8b_rl_dense = Path("/home/xchen6/breaking_updates_rl/results/rl/llama8b_dense_new/checkpoint-100_20251118-015250")
gemma12b_rl_dense = Path("/home/xchen6/breaking_updates_rl/results/rl/gemma12b_dense/checkpoint-500_20251116-145135")
gemma4b_rl_dense = Path("/home/xchen6/breaking_updates_rl/results/rl/gemma4b_merged_dense/checkpoint-500_20251114-111301")
# rl sparse
llama8b_rl_sparse = Path("/home/xchen6/breaking_updates_rl/results/rl/llama8b_sparse/checkpoint-100_20251118-015250")
gemma12b_rl_sparse = Path("/home/xchen6/breaking_updates_rl/results/rl/gemma12b_sparse/checkpoint-500_20251114-111647")
gemma4b_rl_sparse = Path("/home/xchen6/breaking_updates_rl/results/rl/gemma4b_merged_sparse/checkpoint-500_20251114-111301")
# sft
llama8b_sft = Path("/home/xchen6/breaking_updates_rl/results/sft/llama8b/3_epoch_20251107-175328")
gemma12b_sft = Path("/home/xchen6/breaking_updates_rl/results/sft/gemma12b/3_epoch_20251106-021351")
gemma4b_sft = Path("/home/xchen6/breaking_updates_rl/results/sft/sft_gemma4b/3_epoch_20251106-015319")

# sft + rl dense
llama8b_sft_rl_dense = Path("/home/xchen6/breaking_updates_rl/results/grpo/llama8b/checkpoint-500_20251109-121706")
gemma12b_sft_rl_dense = Path("/home/xchen6/breaking_updates_rl/results/grpo/gemma12b_merged/checkpoint-500_20251109-205259")
gemma4b_sft_rl_dense = Path("/home/xchen6/breaking_updates_rl/results/grpo/gemma4b_merged/checkpoint-400_20251117-012654")

# sft + rl sparse
llama8b_sft_rl_sparse = Path("/home/xchen6/breaking_updates_rl/results/grpo/llama8b_sparse/checkpoint-500_20251108-204109")
gemma12b_sft_rl_sparse = Path("/home/xchen6/breaking_updates_rl/results/grpo/gemma12b_merged_sparse/checkpoint-500_20251109-121855")
gemma4b_sft_rl_sparse = Path("/home/xchen6/breaking_updates_rl/results/grpo/gemma4b_merged_sparse/checkpoint-400_20251117-012808")

total_paths = {
    "gemma4b": gemma4b_offtheshelf,
    "llama8b": llama8b_offtheshelf,
    "gemma12b": gemma12b_offtheshelf,
    
    # "llama8b_rl": llama8b_rl,
    # "sft": sft,
    "gemma4b_sft": gemma4b_sft,
    "llama8b_sft": llama8b_sft,
    "gemma12b_sft": gemma12b_sft,
   
    # "rl_dense"
    "gemma4b_rl_dense": gemma4b_rl_dense,
    "llama8b_rl_dense": llama8b_rl_dense,
    "gemma12b_rl_dense": gemma12b_rl_dense,
    
    # "rl_sparse"
    "gemma4b_rl_sparse": gemma4b_rl_sparse,
    "llama8b_rl_sparse": llama8b_rl_sparse,
    "gemma12b_rl_sparse": gemma12b_rl_sparse,
    
    # "sft + rl dense"
    "gemma4b_sft_rl_dense": gemma4b_sft_rl_dense,
    "llama8b_sft_rl_dense": llama8b_sft_rl_dense,
    "gemma12b_sft_rl_dense": gemma12b_sft_rl_dense,
    
    # "sft + rl sparse"
    "gemma4b_sft_rl_sparse": gemma4b_sft_rl_sparse,
    "llama8b_sft_rl_sparse": llama8b_sft_rl_sparse,
    "gemma12b_sft_rl_sparse": gemma12b_sft_rl_sparse,
}


if __name__ == "__main__":
    result_file =  Path("/home/xchen6/breaking_updates_rl/results/aggregated_results.json")
    
    final_results = []
    for model_name, path in total_paths.items():
        for commit_path in path.iterdir():
            metrics = commit_path / "metrics_new.json"
            if not metrics.exists():
                continue
            metrics = json.loads(metrics.read_text())
                
            final_results.append({
                "model_name": model_name.split("_")[0],
                "bu_id": commit_path.name,
                "config": "off_the_shelf" if "_" not in model_name else model_name.split("_", 1)[1],
                "build_success": int(metrics["build_success"]),
                "fixed_error_count":  metrics["fixed_error_count"],      # int
                "fixed_file_count":   metrics["fixed_file_count"],
                "new_errors_count": metrics["new_errors_count"],
                "fixed_errors" : metrics["fixed_errors"],   # list
                # align the buggy counts correctly
                "original_file_count" : metrics["original_error_count"] if metrics["build_success"] else metrics["original_file_count"],
                "original_error_count" : metrics["original_file_count"] if metrics["build_success"] else metrics["original_error_count"],
            })
    result_file.write_text(json.dumps(final_results, indent=4))
            
           
        