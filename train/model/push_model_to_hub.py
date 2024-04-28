"""
This script is used to merge the adaptor weights with the base model weights and push the model to the Hugging Face hub.

Usage:
```bash
python push_model_to_hub.py -a <adaptor_path> -r <repository_path>
```

This can also be done using the Hugging Face CLI as well
"""

# # Uncomment this section if you need to specify the GPU to use
# # Set up the GPU
# import os
# gpu = os.environ["CUDA_VISIBLE_DEVICES"]="3"

# import the required libraries
import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse

# Get the adaptor paths from the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-a","--adaptor_path", type=str, help="Path to the adaptor weights")
# get the repository path from argumentparser
parser.add_argument("-r","--repository_path", type=str, help="Path to the repository to be uploaded")
args = parser.parse_args()

adapters_name = args.adaptor_path
repository_path = args.repository_path

# Get the model
# The model that you want to train from the Hugging Face hub
base_model_name = "mistralai/Mistral-7B-v0.1"

# Fine-tuned math model name (date-month-hour-minutes)
# This is where the adaptor weights are stored
adapters_name = "./results/meta_math_mistral-7b-09-04-20-55"

# Load the base model
basemodel = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    #load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    cache_dir="/projects/barman/cache", # Change this to your cache directory- make sure you have enough space to store the model
)

# Load the adaptor weights and merge them with the base model
model = PeftModel.from_pretrained(basemodel, adapters_name)
model = model.merge_and_unload()

print("Model loaded successfully!")

# Save the model and push it to the hub
model.save_pretrained(adapters_name)
model.push_to_hub(repository_path, token=True, max_shard_size="5GB", safe_serialization=True)

print(f"Model pushed to the hub successfully at {repository_path}")

# Push the tokenizer as well to the hub
tokenizer = AutoTokenizer.from_pretrained(adapters_name)
tokenizer.push_to_hub(repository_path, token=True)

print(f"Tokenizer pushed to the hub successfully at {repository_path}")

# # Uncomment the following code if you want to upload a folder directly to the hub

# from huggingface_hub import HfApi, upload_folder, create_branch

# # Initialize the HfApi class
# api = HfApi()

# path_to_upload = "./metadata"

# # # Optionally, create a new branch for 'nf4'. Beware this will copy all files from main.
# # create_branch(repo_id=new_hub_model_path, repo_type="model", branch="nf4")

# # Upload the entire folder to the specified branch in the repository
# upload_folder(
#     folder_path=path_to_upload,
#     repo_id=repository_path,
#     repo_type="model",  # Assuming it's a model; can be "dataset" or "space" as well
#     # revision="nf4",  # Specify the branch you want to push to
#     token=True,
# )

# print(f"Uploaded contents of {path_to_upload} to {repository_path} on HuggingFace Hub")