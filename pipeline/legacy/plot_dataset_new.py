import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

jsonlpath = "/home/xchen6/breaking_updates_rl/data/prompts_diff/train.jsonl"

# 读入数据
data = pd.read_json(jsonlpath, lines=True)

out_dir = Path(__file__).parent.parent.parent / "data"


def plot_int_distribution(count_series, xlabel, title, legend_label, save_path):
    """
    count_series: Index 为整数（比如 file 数），value 为该整数对应的 commit 个数
    """
    # 1) 补齐 [min, max] 范围内所有整数，没出现的记为 0
    v_min = int(count_series.index.min())
    v_max = int(count_series.index.max())
    full_index = range(v_min, v_max + 1)
    full_counts = count_series.reindex(full_index, fill_value=0)

    # 2) 画图，用数值型 x 轴
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.bar(list(full_index), full_counts.values, label=legend_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("breaking commit count")
    ax.set_title(title)

    # 3) 控制 x 轴刻度数量，避免太挤（比如最多 20 个）
    max_ticks = 20
    span = v_max - v_min + 1
    if span <= max_ticks:
        # 范围不大，所有整数都标出来
        ax.set_xticks(list(full_index))
    else:
        # 范围很大，只标每 step 个整数一个刻度
        step = span // max_ticks + 1
        xticks = list(range(v_min, v_max + 1, step))
        ax.set_xticks(xticks)

    # 让刻度文字横着，便于阅读
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
files_per_commit = data.groupby("breakingCommit").size()  # index=commit, value=file_count

# 分布：file_count -> 有多少个 commit
file_count_dist = files_per_commit.value_counts().sort_index()

total_commits = files_per_commit.shape[0]
total_files = len(data)

plot_int_distribution(
    count_series=file_count_dist,
    xlabel="number of buggy files (per breaking commit)",
    title="Test set distribution of buggy file numbers",
    legend_label=f"breaking updates={total_commits}, buggy files={total_files}",
    save_path=out_dir / "breakingCommit_buggy_files_distribution.png",
)

# ─────────────────────────────
# 2. 错误数分布：
#    x 轴：每个 breaking commit 下的 error 总数
#    y 轴：拥有该 error 总数的 breaking commit 个数
# ─────────────────────────────

data["error_count"] = data["errors"].apply(len)

# 每个 breaking commit 的错误总数
errors_per_commit = data.groupby("breakingCommit")["error_count"].sum()

# 分布：error_total -> 有多少个 commit
error_count_dist = errors_per_commit.value_counts().sort_index()

total_commits_err = errors_per_commit.shape[0]
total_errors = int(data["error_count"].sum())

plot_int_distribution(
    count_series=error_count_dist,
    xlabel="total errors (per breaking commit)",
    title="Test set distribution of total errors",
    legend_label=f"breaking updates={total_commits_err}, errors={total_errors}",
    save_path=out_dir / "breakingCommit_errors_distribution.png",
)