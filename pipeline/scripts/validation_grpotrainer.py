from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig

# 1) 带 metadata 的极简数据集
dataset = Dataset.from_list([
    {"prompt": "Say hello in French.", "path_to_file": "src/hello.py", "project": "demo_A"},
    {"prompt": "Name two prime numbers.", "path_to_file": "src/math.py",  "project": "demo_B"},
])

# 2) 调试型奖励函数：打印它收到的 kwargs（官方建议用 **kwargs）
def debug_reward_func(completions, **kwargs):
    # 官方文档：会传入 prompts/completions 以及数据集中额外的列和其他上下文；
    # 用 **kwargs 接收更稳妥。
    print("=== Inside reward_func ===")
    print("kwargs keys:", list(kwargs.keys()))
    for k, v in kwargs.items():
        # 仅截断打印，避免刷屏
        if isinstance(v, list):
            print(f"  {k}:", [str(x)[:40] for x in v[:2]])
        else:
            print(f"  {k}:", str(v)[:120])
    # 返回与 completions 数量对齐的分数
    return [0.0 for _ in completions]

# 3) 配置：把 max_steps 放进 GRPOConfig（不是传给 train()）
args = GRPOConfig(
    output_dir="./grpo_debug",
    per_device_train_batch_size=2,
    max_steps=1,                 # 关键：在这里设置步数
    # 如果你需要保留数据集中的列给 reward_func 用，确保不移除：
    remove_unused_columns=False, # 文档默认是 False，继续保留
    num_generations=2,           # 每个 prompt 生成候选数
    max_completion_length=16
)

# 4) 初始化并训练（官方示例：train() 不带 max_steps 参数）
trainer = GRPOTrainer(
    model="distilgpt2",              # 文档允许字符串模型 ID；内部 from_pretrained 加载
    reward_funcs=debug_reward_func,
    train_dataset=dataset,
    args=args
)

trainer.train()  # 正确：不传 max_steps