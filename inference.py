from PIL import Image
import torch
import fire

from PaliGemma_Processor import PaliGemmaProcessor
from Gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    '''
    Move the input tensor into the device.
    '''
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    '''
    Call the PaliGemmaProcessor object with prompt and image, which return the model_inputs in device
    model_inputs=: {
            'pixel_values': (batch_size, channel, height, width) tensor for SigLip input
            'input_ids': (batch_size, seq_len) tensor for Gemma input, include the placeholder '<image>'
            'attention_mask': (batch_size, seq_len) tensor for Gemma input, just mask the padding
        }
    '''
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def _sample_top_p(probs: torch.Tensor, p: float):
    '''
    1. Sort the probability in descending
    2. Compute the cumulative sum of probability
    3. Pick the tokens where cumulative sum is p
    4. Rescale the probability as the sum is 1 and sample from this multinomial distribution
    '''
    # (batch_size, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # Sort in descending order
    probs_sum = torch.cumsum(probs_sort, dim=-1) # Calculate the cumulative sum
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p # We will keep the place when mask is True
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1) 
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device) # Get the input dict contains: 'pixel_values', 'input_ids', 'attention_mask'
    input_ids = model_inputs["input_ids"] # (batch_size, seq_len)
    attention_mask = model_inputs["attention_mask"] # (batch_size, seq_len)
    pixel_values = model_inputs["pixel_values"] # (batch_size, channel, height, width)

    kv_cache = KVCache() # Create an empty KV cache

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # Get the model outputs
        # TODO: remove the labels
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        '''
        outputs=: {
            'logits': (batch_size, seq_len, vocab_size) tensor as the PaliGemmaForConditionalGeneration output. The definition is in GemmaForCausalLM module.
            'kv_cache': current updated cache object, ie, contain a list of key and a list of value
        }
        '''
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :] # The logit of the last token: (batch_size, vocab_size)
        # Sample the next token
        if do_sample:
            # Apply temperature before softmax, reduce the gap between different tokens, then will get a more diverse tokens prediction
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            # Greedy pick the largest probability token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(next_token)
        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break
        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1) # In the inference loop, since we use KV cache, each time we just input 1 token, which is the token generated from last step
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    # Concatenate all generate token ids in a tensor
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # Decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(prompt + decoded)

def main(
    model_path: str = None, # The weight of the model
    prompt: str = None, # The prompt
    image_file_path: str = None, # The image we are going to conditional on
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"
    '''
    1. Load the PaliGemma model and tokenizer
    2. Use the model.config and tokenizer to initialize PaliGemmaProcessor object
    3. Pass the model, processor, prompt, image_file_path for inference
    '''
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path, device) # Load HuggingFace model
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens # The number of the image token, 256 for 224-models, since 16*16=256
    image_size = model.config.vision_config.image_size # The size of the image to fit the SigLip model (224*224)
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)