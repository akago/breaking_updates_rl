import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== 路径设置 ====
train_jsonlpath = "/home/xchen6/breaking_updates_rl/data/prompts_diff/train.jsonl"
test_jsonlpath  = "/home/xchen6/breaking_updates_rl/data/prompts_diff/test.jsonl"

out_dir = Path(__file__).parent.parent.parent / "data"
out_dir.mkdir(parents=True, exist_ok=True)

# ==== 读入数据 ====
train_df = pd.read_json(train_jsonlpath, lines=True)
test_df  = pd.read_json(test_jsonlpath,  lines=True)

# ==== 通用：从 “每个 commit 的值” -> “值的分布（直方）” ====
def commit_value_dist(value_series: pd.Series) -> pd.Series:
    """
    value_series: index=breakingCommit, value=某个统计量（file 数 或 error 总数）
    返回: Series，index 为整数统计量，value 为拥有该统计量的 commit 数
    """
    dist = value_series.value_counts()
    dist.index = dist.index.astype(int)
    return dist.sort_index()

# ==== 1. buggy file 数 ====

# 每个 commit 对应的 buggy file 数
files_per_commit_train = train_df.groupby("breakingCommit").size()
files_per_commit_test  = test_df.groupby("breakingCommit").size()

dist_files_train = commit_value_dist(files_per_commit_train)
dist_files_test  = commit_value_dist(files_per_commit_test)

# 统一横轴范围：train + test 的 min/max
files_min = int(min(dist_files_train.index.min(), dist_files_test.index.min()))
files_max = int(max(dist_files_train.index.max(), dist_files_test.index.max()))
files_index = range(files_min, files_max + 1)

dist_files_train_full = dist_files_train.reindex(files_index, fill_value=0)
dist_files_test_full  = dist_files_test.reindex(files_index,  fill_value=0)

total_commits_train = files_per_commit_train.shape[0]
total_commits_test  = files_per_commit_test.shape[0]
total_files_train   = len(train_df)
total_files_test    = len(test_df)

# ==== 2. error 总数 ====

# 每个 buggy file 的错误数
train_df["error_count"] = train_df["errors"].apply(len)
test_df["error_count"]  = test_df["errors"].apply(len)

# 每个 breaking commit 的错误总数
errors_per_commit_train = train_df.groupby("breakingCommit")["error_count"].sum()
errors_per_commit_test  = test_df.groupby("breakingCommit")["error_count"].sum()

dist_errors_train = commit_value_dist(errors_per_commit_train)
dist_errors_test  = commit_value_dist(errors_per_commit_test)

errors_min = int(min(dist_errors_train.index.min(), dist_errors_test.index.min()))
errors_max = int(max(dist_errors_train.index.max(), dist_errors_test.index.max()))
errors_index = range(errors_min, errors_max + 1)

dist_errors_train_full = dist_errors_train.reindex(errors_index, fill_value=0)
dist_errors_test_full  = dist_errors_test.reindex(errors_index,  fill_value=0)

total_errors_train = int(train_df["error_count"].sum())
total_errors_test  = int(test_df["error_count"].sum())

# ==== 3. 画图：一行两列 ====

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 4),        # 每个子图宽度≈7，不比原来窄
    constrained_layout=True # 自动压缩空白
)

max_ticks = 20
width = 0.4  # train/test 并排柱子的宽度

# ---- 左图：buggy file 数分布 ----
ax = axes[0]
x_files = np.array(list(files_index), dtype=float)

ax.bar(
    x_files - width/2,
    dist_files_train_full.values,
    width=width,
    label=f"Train (breaking updates={total_commits_train}, files={total_files_train})",
)
ax.bar(
    x_files + width/2,
    dist_files_test_full.values,
    width=width,
    label=f"Test (breaking updates={total_commits_test}, files={total_files_test})",
)

ax.set_xlabel("Number of buggy files per breaking update")
ax.set_ylabel("Breaking Update count")
ax.set_title("Distribution of buggy file counts")
ax.tick_params(axis="x", labelrotation=0)
ax.legend(fontsize=8)

span_files = files_max - files_min + 1
if span_files <= max_ticks:
    ax.set_xticks(x_files)
else:
    step = span_files // max_ticks + 1
    xticks = np.arange(files_min, files_max + 1, step)
    ax.set_xticks(xticks)

# ---- 右图：error 总数分布 ----
ax = axes[1]
x_err = np.array(list(errors_index), dtype=float)

ax.bar(
    x_err - width/2,
    dist_errors_train_full.values,
    width=width,
    label=f"Train (breaking updates={total_commits_train}, errors={total_errors_train})",
)
ax.bar(
    x_err + width/2,
    dist_errors_test_full.values,
    width=width,
    label=f"Test (breaking updates={total_commits_test}, errors={total_errors_test})",
)

ax.set_xlabel("Total errors per breaking update")
ax.set_ylabel("Breaking Update count")
ax.set_title("Distribution of total errors")
ax.tick_params(axis="x", labelrotation=0)
ax.legend(fontsize=8)

span_err = errors_max - errors_min + 1
if span_err <= max_ticks:
    ax.set_xticks(x_err)
else:
    step = span_err // max_ticks + 1
    xticks = np.arange(errors_min, errors_max + 1, step)
    ax.set_xticks(xticks)

# 保存为一张图
out_path = out_dir / "breakingCommit_train_test_distributions_row2.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved to", out_path)