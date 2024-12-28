from Gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    '''
    1. Load the pre-trained tokenizer from Huggingface
    2. Load the PaliGemma pre-trained weight and its config
    3. Return the pre-trained (model, tokenizer)
    '''
    # Load the tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files, these files contain the weight in a dictionary
    # ... and load them one by one in the tensors dictionary
    # So the tensors will contain all pre-trained weight of the model
    # https://huggingface.co/google/paligemma-3b-pt-224/tree/main    
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config from the config.json file
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device) # Initialize the PaliGemma model with config

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Copy the weight matrix from embedding layer to the language model head
    model.tie_weights()

    return (model, tokenizer)