import pandas as pd
import numpy as np
from scipy.stats import binomtest  # scipy>=1.7
from scipy.stats import wilcoxon
from pathlib import Path


def wilcoxon_metric_vs_off(df, model_name, metric_col, configs_to_compare, off_name="off"):
    """
    对某个 model_name + 某个 metric（error_fix_pct / file_fix_pct / new_errors_pct），
    使用 Wilcoxon signed-rank 检验每个 config_new vs off 的差异。

    返回 DataFrame，每行对应 (model, metric, config_new vs off) 的比较结果。
    """
    df_m = df[df["model_name"] == model_name].copy()

    # 透视：行 = bu_id，列 = config，值 = 该 metric
    wide = df_m.pivot_table(
        index="bu_id",
        columns="config",
        values=metric_col
    )

    results = []

    for cfg_new in configs_to_compare:
        if off_name not in wide.columns or cfg_new not in wide.columns:
            continue

        pair = wide[[off_name, cfg_new]].dropna()
        if pair.empty:
            continue

        off_vals = pair[off_name].astype(float)
        new_vals = pair[cfg_new].astype(float)

        diff = new_vals - off_vals

        # Wilcoxon 要求至少有一个非零差值，否则没法做
        # zero_method="pratt" 会保留 0 差值在样本数里，但不影响秩
        if np.all(diff == 0):
            p_value = np.nan
            W_stat  = np.nan
            n_effective = 0
        else:
            try:
                res = wilcoxon(
                    new_vals,
                    off_vals,
                    alternative="greater",  # 检验 new 是否显著大于 off
                    zero_method="pratt",    # 允许有部分 diff=0
                    mode="auto"
                )
                p_value = res.pvalue
                W_stat  = res.statistic
                # 有效样本 = 非零差值个数
                n_effective = np.count_nonzero(diff != 0)
            except ValueError:
                # 极端情况下（全 0 或其他异常）兜底
                p_value = np.nan
                W_stat  = np.nan
                n_effective = 0

        # 一些直观的 effect 描述
        median_off = off_vals.median()
        median_new = new_vals.median()
        median_diff = diff.median()

        wins   = (diff > 0).sum()
        losses = (diff < 0).sum()
        ties   = (diff == 0).sum()

        results.append({
            "model_name":   model_name,
            "metric":       metric_col,
            "config_new":   cfg_new,
            # "config_off":   off_name,
            "n_pairs":      len(pair),
            # "n_effective":  n_effective,    # 参与 Wilcoxon 的非零差值个数
            "wins_new":     wins,
            "losses_new":   losses,
            "ties":         ties,
            "W_stat":       W_stat,
            "p_value":      p_value,
            "median_off":   median_off,
            "median_new":   median_new,
            "median_diff":  median_diff,    # new - off
        })

    return pd.DataFrame(results)

def sign_test_vs_off(df, model_name, configs_to_compare, off_name="off_the_shelf"):
    """    
    """
    df_m = df[df["model_name"] == model_name].copy()

    # 透视成 wide：行=bu_id，列=config，值=build_success
    wide = (
        df_m.pivot_table(
            index="bu_id",
            columns="config",
            values="build_success"
        )
    )

    results = []

    for cfg_new in configs_to_compare:
        # 如果这个配置在这个模型里根本不存在，就跳过
        if cfg_new not in wide.columns or off_name not in wide.columns:
            continue

        pair = wide[[off_name, cfg_new]].dropna()  # 只保留两个配置都跑过的 BU
        off = pair[off_name].astype(int)
        new = pair[cfg_new].astype(int)

        diff = new - off
        wins   = (diff ==  1).sum()  # new=1, off=0
        losses = (diff == -1).sum()  # new=0, off=1
        ties   = (diff ==  0).sum()  # 00 或 11

        n_discordant = wins + losses

        if n_discordant == 0:
            p_value = np.nan   # 完全平局，没法做 Sign test
        else:
            test = binomtest(
                wins,
                n_discordant,
                p=0.5,
                alternative="greater",  # 检验 "new 赢得更多"
            )
            p_value = test.pvalue

        # 配对 odds ratio: wins / losses，加 0.5 防止除零
        or_paired = (wins + 0.5) / (losses + 0.5)

        results.append({
            "model_name": model_name,
            "config_new": cfg_new,
            "config_off": off_name,
            "n_pairs":    len(pair),
            "wins_new":   wins,
            "losses_new": losses,
            "ties":       ties,
            "p_value":    p_value,
            "or_paired":  or_paired,
        })

    return pd.DataFrame(results)

# def add_bonferroni_flags(df_in, alpha_list=(0.01, 0.05, 0.10)):
#     """
#     对 df_in（某个 model 的结果表）增加多列：
#       sig_0.01_bonf, sig_0.05_bonf, sig_0.1_bonf
#     表示在 Bonferroni 校正后是否显著。
#     """
#     df = df_in.copy()
#     # 只对有 p_value 的行计数（NaN 的跳过）
#     valid = df["p_value"].notna()
#     m = valid.sum()
#     if m == 0:
#         # 没有有效检验，直接返回
#         for a in alpha_list:
#             col = f"sig_{a:.2f}_bonf"
#             df[col] = False
#         return df

#     for a in alpha_list:
#         thresh = a / m
#         col = f"sig_{a:.2f}_bonf"  # 比如 "sig_0.01_bonf"
#         df[col] = False
#         df.loc[valid, col] = df.loc[valid, "p_value"] <= thresh

#     return df

result_file =  Path("/home/xchen6/breaking_updates_rl/results/aggregated_results.json")

df = pd.read_json(result_file)
configs_to_compare = ["rl_sparse", "rl_dense", "sft_rl_sparse", "sft_rl_dense"]

df_res_llama8b   = sign_test_vs_off(df, "llama8b",   configs_to_compare)
df_res_gemma4b   = sign_test_vs_off(df, "gemma4b",   configs_to_compare)
df_res_gemma12b  = sign_test_vs_off(df, "gemma12b",  configs_to_compare)

df_sign_all = pd.concat([df_res_llama8b, df_res_gemma4b, df_res_gemma12b], ignore_index=True)
print(df_sign_all)

### Holm-Bonferroni Correction
# def holm_bonferroni(pvals, alpha=0.05):
#     """
#     输入: pvals 为 list 或 array
#     输出: boolean 数组，表示每个 p 是否在 Holm–Bonferroni 下显著
#     """
#     pvals = np.asarray(pvals)
#     m = len(pvals)
#     # 按 p 值排序
#     order = np.argsort(pvals)
#     sorted_p = pvals[order]

#     # 逐个比较
#     passed = np.full(m, False)
#     for i, p in enumerate(sorted_p, start=1):  # i 从 1 到 m
#         threshold = alpha / (m - i + 1)
#         if p <= threshold:
#             passed[i-1] = True
#         else:
#             # 一旦某个不通过，后面的也都不通过
#             break

#     # 把结果映射回原始顺序
#     out = np.full(m, False)
#     out[order] = passed
#     return out

# df_sign_all["significant_holm"] = False

# alpha = 0.1

# for model in df_sign_all["model_name"].unique():
#     mask = df_sign_all["model_name"] == model
#     pvals = df_sign_all.loc[mask, "p_value"].values
#     # 去掉 NaN 的情况（比如全平局）
#     not_nan = ~np.isnan(pvals)
#     pvals_nonan = pvals[not_nan]

#     if len(pvals_nonan) == 0:
#         continue

#     sig_nonan = holm_bonferroni(pvals_nonan, alpha=alpha)

#     # 写回 DataFrame
#     idx_model = df_sign_all.index[mask][not_nan]
#     df_sign_all.loc[idx_model, "significant_holm"] = sig_nonan
# print("Sign test results with Holm-Bonferroni correction:")
# print(df_sign_all)


#### all models
# all_models = df["model_name"].unique()
# rows_all = []

# for mname in all_models:
#     df_m = sign_test_vs_off(df, mname, configs_to_compare)
#     df_m = add_bonferroni_flags(df_m, alpha_list=(0.01, 0.05, 0.10))
#     rows_all.append(df_m)

# df_sign_all = pd.concat(rows_all, ignore_index=True)
# print("Sign test results with Bonferroni correction:")
# print(df_sign_all)


### CEFR
df = df.copy()

# 避免除以 0：先算比例，然后把原始计数为 0 的设成 NaN 或 0
df["error_fix_pct"] = df["fixed_error_count"] / df["original_error_count"]
df["file_fix_pct"]  = df["fixed_file_count"]  / df["original_file_count"]
df["new_errors_pct"]  = (df["fixed_error_count"]  - df["new_errors_count"]) / df["original_error_count"]

metrics = ["new_errors_pct", ]#["file_fix_pct", "error_fix_pct", "new_errors_pct"]

all_models = df["model_name"].unique()

rows = []
for model in all_models:
    for metric in metrics:
        df_res = wilcoxon_metric_vs_off(
            df=df,
            model_name=model,
            metric_col=metric,
            configs_to_compare=configs_to_compare,
            off_name="off_the_shelf"
        )
        rows.append(df_res)

df_wilcoxon_all = pd.concat(rows, ignore_index=True)
print(df_wilcoxon_all)
