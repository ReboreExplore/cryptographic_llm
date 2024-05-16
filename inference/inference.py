import warnings
warnings.filterwarnings("ignore")
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer,pipeline, logging, BitsAndBytesConfig, AutoModelForCausalLM
from datasets import load_dataset
from random import randrange


# Get the model
# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-chat-hf"


# Fine-tuned math model name (date-month-hour-minutes)
new_model = "llama-2-7b-chat-math-01-04-12-10"

# Output directory where the model predictions and checkpoints will be stored
output_dir = f"./results/{new_model}/"

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
# system prompt
################################################################################

DEFAULT_SYSTEM_PROMPT = """You are a fine-tuned AI model who is a math genious. 
You can solve simple to moderate level mathematics problems. 
Follow a chain of thought approach while answering. Answer in brief. """

def main():

    # Get the dataset from hugging_face
    math_dataset = load_dataset("hendrycks/competition_math",trust_remote_code=True, split="test")
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load finetuned LLM model and tokenizer
    ft_model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        cache_dir="/projects/barman/cache",
    )

    # Load finetuned LLM model and tokenizer
    ft_model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        cache_dir="/projects/barman/cache",
    )

    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        cache_dir="/projects/barman/cache",
    #     use_flash_attention_2=use_flash_attention,
    )


    ################################################################################
    # Run inference
    ################################################################################

    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)
    # Run text generation pipeline with our next model
    # prompt = "What is a large language model?"
    pipe = pipeline(task="text-generation", model=ft_model, tokenizer=tokenizer, max_length=512)
    result = pipe(f"<s>[INST] <<SYS>> {DEFAULT_SYSTEM_PROMPT} <</SYS>> {math_dataset[0]["problem"]} [/INST]")


    print(f"Instruction:\n{DEFAULT_SYSTEM_PROMPT}\n")
    print(f"Input:\n{math_dataset[0]["problem"]}\n")
    print(f"Generated Response with fine tuned model - epoch 1:\n {result[0]['generated_text'].split("[/INST]")[-1]}\n")
    print(f"Ground Truth:\n{math_dataset[0]["solution"]}")



if __name__ == "__main__":
    main()