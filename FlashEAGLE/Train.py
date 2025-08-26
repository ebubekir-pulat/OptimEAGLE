from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# Reference for below code: https://huggingface.co/docs/trl/main/en/sft_trainer
trainer = SFTTrainer(
    model="AngelSlim/Qwen3-1.7B_eagle3",
    train_dataset=load_dataset("PKU-Alignment/Align-Anything-Instruction-100K-zh", split="test"),
)
trainer.train()