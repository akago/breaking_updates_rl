import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from matplotlib.ticker import MaxNLocator

root_dir = Path(__file__).parent.parent.parent

index_file = str(root_dir / "dataset_metadata.json")


with open(index_file, "r", encoding="utf-8") as f:
    index_data = json.load(f)

sample_files = index_data.get("success", [])


projects = []
file_counts_per_sample = []
error_counts_per_sample = []
# for breaking_commit in sample_files:
data_path = root_dir / "data" / "dataset"
for folder in data_path.iterdir():
    fpath = folder / "context.json"
    if not fpath.exists():
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        meta = json.load(f)
        project = meta.get("project")
        if project:
            projects.append(project)

            # Count errors in this project
            buggy_files = meta.get("buggyFiles", {}) or {}
            n_errors = sum(len(err_list) for err_list in buggy_files.values())
            n_files = len(buggy_files.keys())
            file_counts_per_sample.append(int(n_files))
            error_counts_per_sample.append(int(n_errors))
            

df = pd.DataFrame(projects, columns=["project"])
print(df["project"].value_counts())
project_counts = df.value_counts().reset_index(name="num_commits")
project_counts.columns = ["project", "num_commits"]

save_dir = root_dir / "statistics"

# ---- Bar chart: commits per project 
plt.figure(figsize=(12,6))
plt.bar(project_counts["project"], project_counts["num_commits"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of Breaking Updates")
plt.title("Distribution of Breaking Updates per Project")
plt.text(
    0.99, 0.95,
    f"Total number of breaking updates: {len(projects)} across {project_counts.shape[0]} projects",
    ha="right", va="top", transform=plt.gca().transAxes,
    fontsize=10, color="dimgray"
)
plt.tight_layout()
plt.savefig(save_dir / "project_distribution_bar.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()


#  how many projects have N commits
max_cnt = int(project_counts["num_commits"].max())
bins = range(1, max_cnt + 2)
plt.figure(figsize=(8, 5))
plt.hist(project_counts["num_commits"], bins=bins, align="left", rwidth=0.9)
plt.xlabel("Breaking commits per project")
plt.ylabel("Number of projects")
plt.title("Distribution of Project Frequencies")
plt.xticks(range(1, max_cnt + 1))
plt.tight_layout()
hist_path = save_dir / "project_distribution_hist.png"
plt.savefig(hist_path, dpi=200, bbox_inches="tight")
plt.close()

# how many projects have N buggy files
plt.figure(figsize=(8, 5))
plt.hist(file_counts_per_sample, bins=range(max(file_counts_per_sample)+2), align="left", rwidth=0.9)
plt.xlabel("Number of Buggy Files per Project (N)")
plt.ylabel("Number of Projects")
plt.title("Distribution of Buggy Files per Project")
plt.text(
    0.99, 0.95,
    f"Total buggy files number: {sum(file_counts_per_sample)} across {len(projects)} projects",
    ha="right", va="top", transform=plt.gca().transAxes,
    fontsize=10, color="dimgray"
)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
file_hist_path = save_dir / "file_distribution_hist.png"
plt.savefig(file_hist_path, dpi=200, bbox_inches="tight")
plt.close()

# how many projects have N errors
 
plt.figure(figsize=(8, 5))
plt.hist(error_counts_per_sample, bins=range(max(error_counts_per_sample)+2), align="left", rwidth=0.9)
plt.xlabel("Number of Errors per Project (N)")
plt.ylabel("Number of Projects")
plt.title("Distribution of Errors per Project")
plt.text(
    0.99, 0.95,
    f"Total errors: {sum(error_counts_per_sample)} across {len(projects)} projects",
    ha="right", va="top", transform=plt.gca().transAxes,
    fontsize=10, color="dimgray"
)
plt.tight_layout()

error_hist_path = save_dir / "error_distribution_hist.png"
plt.savefig(error_hist_path, dpi=200, bbox_inches="tight")
plt.close()

# table view
# import caas_jupyter_tools
# caas_jupyter_tools.display_dataframe_to_user("Project distribution", project_counts)