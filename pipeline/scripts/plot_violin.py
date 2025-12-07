from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import to_rgb
import numpy as np


def darken_color(color, factor=0.6):
    """
    把颜色调暗一点：factor < 1 越小越暗
    color 可以是 "tab:blue" 或 (r,g,b)
    """
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)

def plot_violin_metric(df, metric_col, metric_label, config_order, save_path=None, y_clip=None):
    df_plot = df.copy()
    # 比例 0~1 -> 百分比 0~100
    df_plot["metric_pct"] = df_plot[metric_col] * 100.0
    df_plot["config"] = pd.Categorical(
        df_plot["config"], categories=config_order, ordered=True
    )

    if y_clip is not None:
        low_clip, high_clip = y_clip
        df_plot["metric_pct"] = df_plot["metric_pct"].clip(lower=low_clip, upper=high_clip)
    # 1. 定义每个模型的浅色和深色
    all_models = sorted(df_plot["model_name"].unique())
    base_colors = sns.color_palette("Set2", len(all_models))  # 柔和的浅色
    model_to_light = {m: c for m, c in zip(all_models, base_colors)}
    model_to_dark  = {m: darken_color(c, 0.6) for m, c in model_to_light.items()}

    plt.figure(figsize=(10, 6))

    # 2. 用浅色画小提琴
    ax = sns.violinplot(
        data=df_plot,
        x="config",
        y="metric_pct",
        hue="model_name",
        order=config_order,
        palette=model_to_light,   # 每个模型自己的浅色
        inner=None,               # 自己画 mean/median
        cut=0,
        scale="width",
        linewidth=1,
    )

    # 让所有小提琴半透明一点
    for pc in ax.collections:
        pc.set_alpha(0.5)

    # 3. seaborn 的 legend 告诉我们 hue 的顺序
    handles_raw, labels_raw = ax.get_legend_handles_labels()
    models_order = list(labels_raw)   # 实际使用的模型顺序
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # 4. 计算每个 (config, model) 的 mean / median，用深色画横线
    stats = (
        df_plot.groupby(["config", "model_name"])["metric_pct"]
        .agg(["mean", "median"])
        .reset_index()
    )

    width = 0.8
    n_models = len(models_order)

    for _, row in stats.iterrows():
        cfg = row["config"]
        mdl = row["model_name"]
        mean = row["mean"]
        med  = row["median"]

        cfg_idx = config_order.index(cfg)
        mdl_idx = models_order.index(mdl)

        x_center = cfg_idx - width/2 + width * (mdl_idx + 0.5) / n_models
        color_dark = model_to_dark[mdl]

        # mean：短一点的深色细线
        ax.hlines(
            y=mean,
            xmin=x_center - 0.05,
            xmax=x_center + 0.05,
            colors=color_dark,
            linewidth=1.5,
        )
        # median：更长的深色粗线
        ax.hlines(
            y=med,
            xmin=x_center - 0.10,
            xmax=x_center + 0.10,
            colors=color_dark,
            linewidth=3,
        )

    # 5. 轴标签 & 百分比刻度
    ax.set_xlabel("Training configuration")
    ax.set_ylabel(metric_label)

    y_min = df_plot["metric_pct"].min()
    y_max = df_plot["metric_pct"].max()

    if y_min >= 0:
        # 没有负值：和原来差不多，0 ~ 正数，顶多到 100%
        if y_max <= 0:
            y_max = 1.0
        upper = min(100, y_max * 1.1)
        lower = 0.0
    else:
        # 有负值：上下都留一点 padding
        if y_max == y_min:
            # 极端情况：全是同一个值
            span = abs(y_max) if y_max != 0 else 1.0
        else:
            span = y_max - y_min
        pad = 0.05 * span
        lower = y_min - pad
        upper = y_max + pad
        # 正方向最多不超过 100%，以防某些比例被算得太大
        upper = min(100.0, upper)

    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    # 6. legend：每个模型一组：浅色方块 + 深色 mean/median 线
    legend_handles = []
    legend_labels  = []

    for mdl in models_order:
        light = model_to_light[mdl]
        dark  = model_to_dark[mdl]

        # 模型色块（浅色填充，深色边框）
        patch = Line2D(
            [0], [0],
            marker="s", linestyle="None",
            markerfacecolor=light, markeredgecolor=dark,
            markersize=10,
        )
        legend_handles.append(patch)
        legend_labels.append(mdl)

        # 深色 mean 线
        mean_line = Line2D([0], [0], color=dark, lw=1.5)
        legend_handles.append(mean_line)
        legend_labels.append("  mean")

        # 深色 median 线
        med_line = Line2D([0], [0], color=dark, lw=3.0)
        legend_handles.append(med_line)
        legend_labels.append("  median")

    ax.legend(
        legend_handles,
        legend_labels,
        title="Model / summary",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
        handlelength=2.5,
        handletextpad=0.8,
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

result_file =  Path("/home/xchen6/breaking_updates_rl/results/aggregated_results.json")


if __name__ == "__main__":
    df = pd.read_json(result_file)

    df = df.copy()

    # 避免除以 0：先算比例，然后把原始计数为 0 的设成 NaN 或 0
    df["error_fix_pct"] = df["fixed_error_count"] / df["original_error_count"]
    df["file_fix_pct"]  = df["fixed_file_count"]  / df["original_file_count"]
    df["new_errors_pct"]  = (df["fixed_error_count"]  - df["new_errors_count"]) / df["original_error_count"]
    print(df[["fixed_error_count", "original_error_count", "error_fix_pct"]].head(10))
    
    print(df["error_fix_pct"].describe())
    
    # mask = (df["fixed_error_count"] < df["new_errors_count"]) & (df["config"] == "rl_dense") & (df["model_name"] == "gemma12b")
    # print("gemma12b rl dense rows with minus REF:", mask.sum())
    
    mask = df["build_success"] == True
    print("build success:", mask.sum())

    # 2) 看看这些行长什么样
    cols = [
        "model_name", "config", "bu_id",
        "fixed_error_count", "original_error_count",
        "new_errors_count", "build_success"
    ]
    print(df.loc[mask, cols].head(60))

    # 3) 看一下这些 bu_id 是否重复出现多次（可能是聚合错误）
    print(
        df.loc[mask]
        .groupby(["model_name", "config"])["bu_id"]
        .nunique()
    )
    # 如果 original_xxx_count == 0，你可以选择设为 0 或 NaN，看你怎么理解
    df.loc[df["original_error_count"] == 0, "error_fix_pct"] = 0.0
    df.loc[df["original_file_count"] == 0,  "file_fix_pct"]  = 0.0

    # 可选：只看 off vs rl vs sft vs sft_rl 这几类
    config_order = ["off_the_shelf", "rl_sparse", "rl_dense", "sft", "sft_rl_sparse", "sft_rl_dense"]
    # config_order = ["off", "sft", "rl", "sft_rl"]
    df = df[df["config"].isin(config_order)].copy()

    # 把 config 变成有序类别，保证横轴顺序
    df["config"] = pd.Categorical(df["config"], categories=config_order, ordered=True)

    plot_violin_metric(
        df=df,
        metric_col="error_fix_pct",
        metric_label="Errors fixed (%)",
        config_order=config_order,
        save_path="violin_error_fix_pct.png",
    )
    
    plot_violin_metric(
        df=df,
        metric_col="file_fix_pct",
        metric_label="Files fixed (%)",
        config_order=config_order,
        save_path="violin_file_fix_pct.png",
    )

    
    plot_violin_metric(
        df=df,
        metric_col="new_errors_pct",
        metric_label="Relative Errors Fix Rate",
        config_order=config_order,
        save_path="violin_new_errors_pct.png",
        y_clip=(-200, 100)
    )
    # error_hist_path = "violin.png"
    # plt.savefig(error_hist_path, dpi=200, bbox_inches="tight")
    # plt.close()