from datasets import load_dataset
from huggingface_hub import login

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("laion/relaion400m")