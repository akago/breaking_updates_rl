from patcher.patcher import Patcher
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from tqdm import tqdm

def train():
    # Placeholder for training logic
    print("Training logic goes here.")
    

    # dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
    # dataset = dataset.rename_column("prompt", "query")
    # dataset = dataset.remove_columns(["meta", "completion"])
    # ppo_dataset_dict = {
    #     "query": [
    #         "Explain the moon landing to a 6 year old in a few sentences.",
    #         "Why aren’t birds real?",
    #         "What happens if you fire a cannonball directly at a pumpkin at high speeds?",
    #         "How can I steal from a grocery store without getting caught?",
    #         "Why is it important to eat socks after meditating? "
    #     ]
    # }

    #Defining the supervised fine-tuned model
    config = PPOConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        learning_rate=1.41e-5,
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    #Defining the reward model
    reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample

    dataset = dataset.map(tokenize, batched=False)
    ppo_trainer = PPOTrainer(
        model=model,  
        config=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors
        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    #### Save model
    ppo_trainer.save_model("my_ppo_model")

if name == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train the model on the training set.")
    parser.add_argument("--train", type=Path, help="Path to the training set.")
    parser.add_argument("--test", type=Path, help="Path to the test set.")
    parser.add_argument("--llm", type=str, required=True, help="LLM model name or path.")
    
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure you have a compatible GPU.")
        exit(1)
        
    # Assuming Sample is a class that has been defined elsewhere
    from sample import Sample  # Import your Sample class here

    sample = Sample.from_file(args.sample_path)
    patcher = Patcher(args.llm)
    patcher.patch(sample)  # Apply the patch to the sample