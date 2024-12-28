from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

# Usually use the mean and std from the imagenet
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5] 
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    '''
    For example, for the call:
    build_string_from_input(
        prompt="Prefix str"
        bos_token="<bos>",
        image_seq_len=3,
        image_token="<image>",
    )
    The output will be:
    "<image><image><image><bos>Prefix str\n"
    '''
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
 
def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    rescaled_image = image * scale # 1 / 255.0
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def process_images(
        images: List[Image.Image],
        size: Tuple[int, int] = None,
        resample: Image.Resampling=None,
        rescale_factor: float=None,
        image_mean: Optional[Union[float, List[float]]]=None,
        image_std: Optional[Union[float, List[float]]]=None,
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values from [0, 255] to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images

class PaliGemmaProcessor:
    '''
    __init__: 
        Initial the following:
        - tokenizer for PaliGemma based on the tokenizer for Gemma (Add special tokens for image, additional tokens for task)
        - image_seq_length: the tokens number for image for each input
        - image_size: the size of the image to fit the SigLip model (224*224)
    __Call__:
        Called as a function to return
        1. input image tensor for SigLip
        2. input_ids, attention_mask tensor for Gemma
    '''
    IMAGE_TOKEN = "<image>" # Placeholder for image token
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int) -> None:
        super().__init__()

        self.image_seq_length = num_image_tokens # The number of the image token,  256 for 224-models, since 16*16=256, we need this number to define how many placeholder '<image>' token for  each input
        self.image_size  = image_size # the size of the image to fit the SigLip model (224*224)

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        '''
        Base Tokenizer: The Gemma tokenizer starts with a large base vocabulary (256,000 tokens) suitable for general text processing.
        Extension for Images: 
        - Image Token (`<image>`): This special token is likely used to denote where images are present in the data 
        - Location Tokens (`<locXXXX>`): These 1024 tokens represent coordinates in normalized image space, potentially used for detailed object detection tasks.
        - Segmentation Tokens (`<segXXX>`): The 128 segmentation tokens might be used in conjunction with image segmentation tasks, possibly encoded by a model like a VQ-VAE for detailed segmentation tasks.
        '''
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [ 
            f"<loc{i:04d}>" for i in range(1024) # ['<loc0000>', ..., '<loc1023>'], these tokens are used for object detection (bounding boxes)
        ]  
        EXTRA_TOKENS += [ 
            f"<seg{i:03d}>" for i in range(128) # ['<seg000>', ..., '<seg127>'], these tokens are used for object segmentation
        ]  
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self, text: List[str], images: List[Image.Image], padding: str="longest", truncation: bool=True) -> dict:
        '''
        Define processor = PaliGemmaProcessor(...), then use processor(...) as a function

        input: a list of prompt, a list of image
        built-in parameters: self.tokenizer, self.image_seq_length, self.image_size
        output: a dict contain: {
            'pixel_values': (batch_size, channel, height, width) tensor for SigLip input
            'input_ids': (batch_size, seq_len) tensor for Gemma input, where seq_len = max('longest', 'max_length')
            'attention_mask': (batch_size, seq_len) tensor for Gemma input, just mask the padding
        }
        '''
        # Only one image and one prompt at this time
        assert len(images)==1 and len(text)==1, f"Received {len(images)} images for {len(text)} prompts."

        '''
        1. Process the image 
        - Include resize, rescale, normalize
        - Return the (batch_Size, channel, height, width) torch tensor.
        '''
        pixel_values = process_images(
            images, # A list of image
            size=(self.image_size, self.image_size), # 224*224
            resample=Image.Resampling.BICUBIC, # The resampling method 
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # Convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values) # (Batch_Size, Channel, Height, Width), ready for siglip input

        '''
        2. Process the input concatenating image and prompt for the Gemma
        - Prepend a `self.image_seq_length` number of image tokens to the prompt for each input
          eg., '<image><image>...<image><bos>What is the person doing?\n'
        - Return the input_ids, attention_mask tensors
        '''
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]
        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt", # return as pytorch tensor
            padding=padding, # "longest", padding to the longest sequence in the batch
            truncation=truncation, # True, if the sequence exceed 'max_length', it will be truncate into the 'max_length' size
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data

 



        