#import datasets library
from datasets import load_dataset
import argparse

# Load the dataset to be pushed to the hub
# If you have a dataset in CSV format, with some specific options avoid uploading the csv file with the
# hugging face UI, use the load_dataset function to load the dataset and push it to the hub.


# get the dataset path from argumentparser 
parser = argparse.ArgumentParser()
parser.add_argument("-p","--dataset_path", type=str, help="Path to the dataset to be uploaded")
args = parser.parse_args()

dataset_path = args.dataset_path


# For infomation - https://huggingface.co/docs/datasets/en/upload_dataset
dataset = load_dataset('csv', data_files=[dataset_path], sep=',',quotechar='"',skipinitialspace=True,encoding='utf-8')

# Push the dataset to the hub
dataset.push_to_hub("Manpa/cryptollm-v1")

print("Dataset pushed to the hub successfully!")