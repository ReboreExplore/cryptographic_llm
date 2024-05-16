"""
This is the training script for the Llama-2-7b model. 
The script trains the model on a math dataset and saves the trained model to the disk.

Dataset used: hendrycks/competition_math

Usage:
```bash
python crypto-llama7b.py CUDA_VISIBLE_DEVICES=0
```
"""


# Import packages and modules
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from tqdm.notebook import tqdm
import logging
import gc
import pandas as pd

#Force garbage collection
gc.collect()

# Set up logging configuration
logging.basicConfig(filename='results.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

logging.info("Modules Loaded")

################################################################################
# Prompt Formatting
################################################################################

# Default system prompt for LLAMA2-style conversations
DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT = """You are a cryptography expert. You have good grasp on the cryptographic concepts and can solve medium 
to high level problems related to maths and cryptography."""


def format_conversation_llama2(dataset):
    '''
    Formats a conversation in LLAMA2 style.

    This function takes a dataset containing a problem and its solution, and formats it into a LLAMA2-style 
    conversation.

    Args:
    - dataset (dict): A dictionary containing the problem and solution of the conversation.

    Returns:
    dict: A dictionary containing the formatted conversation.

    Example:
    >>> dataset = {'problem': 'How can I improve my coding skills?', 'solution': 'You can improve your coding skills 
    by practicing regularly and working on challenging projects.'}
    >>> formatted_conversation = format_conversation_llama2(dataset)
    >>> print(formatted_conversation)
    {'text': '<s>[INST] <<SYS>> How can I improve my coding skills? <</SYS>> You can improve your coding skills by 
    practicing regularly and working on challenging projects. </s>'}

    '''

    template = """<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{question}[/INST] {answer}</s>"""

    conversation = template.format(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        question=dataset['problem'],
        answer=dataset['solution'],
    )

    return {"text": conversation}

################################################################################
# Model Definitions
################################################################################

# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Fine-tuned math model name (date-month-hour-minutes)
new_model = "llama7b-crypto-06-04-10-30"

# Output directory where the model predictions,configuration files and checkpoints will be stored
output_dir = f"./results/{new_model}"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Number of training epochs
num_train_epochs = 5

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 25

# Log every X updates steps
logging_steps = 25

# Logging Directory
logging_dir = f"./logs/{new_model}/"

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}
# # Set to "auto" while using mulptiple GPUs
# device_map = "auto"


def main():

    # Get the dataset from hugging_face
    # Get the crypto-dataset from hugging face
    math_dataset = load_dataset("hendrycks/competition_math",trust_remote_code=True, split= "train")
    logging.info("Dataset Loaded")

    # Format the dataset
    math_format_dataset = math_dataset.map(
    format_conversation_llama2,
    remove_columns=math_dataset.column_names, # remove all columns; only "text" will be left
    num_proc=os.cpu_count()  # multithreaded
)
    
    logging.info("Dataset Formatted")
    
    # Load LLaMA tokenizer
    # set_special_tokens = True, so the bos (beginning of sequence </s>) and eos (end of sequence </s>) token is added
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # not adding mask token for now - can be used to give attention to some part of the sequence
    # manually add the padding token and set it to the eos token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    tokenizer.save_pretrained(output_dir)

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        cache_dir="/projects/barman/cache", # Set the cache directory manually if your space in home folder is limited
    )
    
    logging.info("Model loaded")

    
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules= ["q_proj", "v_proj", "k_proj"],
        # modules_to_save= ["embed_tokens", "lm_head"],
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
        #load_best_model_at_end=True, # This option only saves the best model and no checkpoints
        #save_strategy="no", # use this when using the load_best_model_at_end (requires save_strategy and eval_strategy to be same)
        #evaluation_strategy="epoch", 
        #logging_strategy="epoch",
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=math_format_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    # Save model configuration into the output directory
    model.config.to_json_file(output_dir + "/config.json")
    
    logging.info("Training Started")
    # Train model
    trainer.train()

    logging.info("Training Successfully Completed")
    # Save trained model
    trainer.model.save_pretrained(new_model)

    logging.info("Model Successfully saved")

    
    loss_outputs = pd.DataFrame(trainer.state.log_history)

    # Select desired columns
    desired_columns = ["loss", "learning_rate", "epoch", "step"]

    # Keep only those columns in the DataFrame
    loss_outputs = loss_outputs[desired_columns]

    # Save the loss results
    filename = f"loss-results"
    filepath = os.path.join(f"results/{new_model}", filename)  # Replace with your desired path

    # Save DataFrame to CSV
    loss_outputs.to_csv(filepath, index=False)  # Avoid saving index column

if __name__ == "__main__":
    main()