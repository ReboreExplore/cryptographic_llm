"""
This script creates a Hugging Face repository and uploads the initial files

Alternatively, you can use the Hugging Face CLI to create a repository and upload files.

Usage:
```bash
python create_hf_repository.py -r <repo_id>
```
"""

# Importing required libraries
from huggingface_hub import HfApi
import argparse

# Get the repository id from the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-r","--repo_id", type=str, help="Repository ID")
args = parser.parse_args()

repo_name = args.repo_id


# Create a hugging face repository
api = HfApi()
# Change the repo_type to "dataset" if you are uploading a dataset and to "space" if you are uploading a space
api.create_repo(repo_id=repo_name, private=True,repo_type="model") 

# The initial upload - uncomment this part if you want to upload specific things
api.upload_file(
    path_or_fileobj="./README.md",
    path_in_repo="README.md",
    repo_id=repo_name,
    commit_message= "This is my first commit"
)