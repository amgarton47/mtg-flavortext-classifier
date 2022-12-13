from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained("gpt2")


# <clue> clue_1 </clue> == <answer> answer_1 </answer>


# config =

# Training arguments

# Trainer

# Trainer.train...

# dataset = TextDataset
