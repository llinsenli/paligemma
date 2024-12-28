from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    '''
    Setup the config for SiglipVision model
    '''
    def __init__(
        self,
        hidden_size = 768, # The embedding dim
        intermediate_size = 3072,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        num_channels = 3, # RGB in each image
        image_size = 224, # For any image need to resize into 224-by-224 
        patch_size = 16, # Image will be divide into patches, each patch is 16-by16
        layer_norm_eps = 1e-6,
        attention_dropout = 0.0,
        num_image_tokens: int = None, # How many image embedding for each image (num_patches)
        **kwargs           
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size =image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    '''
    The Embedding Layer of the SiglipVisionTransformer
    Include: 
        1. a 2-d convolution layer as the patch_embedding, 
        2. a position embedding match the patch_embedding
    The output is patch_embedding + position embedding
    '''
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        '''
        Consider just one instance(not a batch) for a 2-D convolutional layer: 
        input: (in_channels, image_size, image_size), an image
        filter:(in_channels, kernel_size, kernel_size), #param is in_channels*kernel_size*kernel_size
            The output for one filter is the inner product of the part of input(same size as the filter) and filter,
            which is a 2-d tensor ((image_size-kernel_size)/stride + 1, image_size-kernel_size)/stride + 1).
            If kernel_size=stride, then the output is (image_size/kernel_size, image_size/kernel_size).
        
        If there are out_channels of filters:
            input: (in_channels, image_size, image_size)
            filter: (out_channels, in_channels, kernel_size, kernel_size). #param is out_channels*in_channels*kernel_size*kernel_size
            output: (out_channels, image_size/kernel_size, image_size/kernel_size)
        '''
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels, # Number of input channels, typically 3 for RGB images
            out_channels = self.embed_dim, # The embedding dimension, determines the number of distinct feature detectors
            kernel_size = self.patch_size, # Size of each patch to be extracted and embedded
            stride = self.patch_size, # Stride size, matching the patch size to prevent overlap
            padding = "valid", # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 # The total number of tokens, or size of input_ids in NLP
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.tensor:
        '''
        input: (batch_size, channels, height, width), a batch of image
        output: (batch_size, num_patches, embed_dim), a batch of image, with each image is a list of embedding vector
        '''
        # [batch_size, channels, height, width] --> [batch_size, embed_dim, num_patch_h, num_patch_w]
        # where num_patch_h, num_patch_w = height/patch_size, width/patch_size
        _, _, height, width = pixel_values.shape 
        patch_embeds = self.patch_embedding(pixel_values)
        # [batch_size, embed_dim, num_patch_h, num_patch_w] --> [batch_size, embed_dim, num_patches]
        # where num_patches = num_patch_h*num_patch_w
        embeddings = patch_embeds.flatten(2) # keep the first 2 dim and flatten the later dim
        # [batch_size, embed_dim, num_patches] --> [batch_size, num_patches, embed_dim]
        embeddings = embeddings.transpose(1,2) # transformer need a sequence of embedding
        # Add the embeddings with the position embedding
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class SiglipAttention(nn.Module):
    '''
    The multi-head attention from 'Attention is all you need'
    '''
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**(-0.5) # 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor)->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        input: (batch_size, num_patches, embed_dim), from the layernorm layer
        output: (batch_size, num_patches, embed_dim)
        '''
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) # (batch_size, num_patches, embed_dim)
        key_states = self.k_proj(hidden_states) # (batch_size, num_patches, embed_dim)
        value_states = self.v_proj(hidden_states) # (batch_size, num_patches, embed_dim)

        # Prepare for the multi head attention tensor
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, num_head, head_dim) -> (batch_size, num_head, num_patches, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute the attention weight using formula Q * K^T / sqrt(dk)
        # (batch_size, num_head, num_patches, head_dim) * (batch_size, num_head, head_dim, num_patches) -->
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale) # (batch_size, num_head, num_patches, num_patches)
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        # Apply the softmax row-wise, (batch_size, num_head, num_patches, num_patches)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Multiply the attention weights by the value states. 
        # (batch_size, num_head, num_patches, num_patches) * (batch_size, num_head, num_patches, head_dim) -->
        attn_output = torch.matmul(attn_weights, value_states) # (batch_size, num_head, num_patches, head_dim) 
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Transpose back and concat the head to get the original tensor size
        # (batch_size, num_head, num_patches, head_dim) -> (batch_size, num_patches, num_head, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (batch_size, num_patches, num_head, head_dim) -> (batch_size, num_patches, embed_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim) 

        # Mix the information from the heads result for each token
        '''
        In the previous step, we only concatenate the head_embed computed independently in multi-head attention for each token. 
        There is no communication between different head_embed.
        Thus, apply this Linear layer Wo
        '''
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class SigliMLP(nn.Module):
    '''
    Define the fully connected layer. All position(patch) share the same parameter within each layer, but different parameter in different layer
    Include two linear layer and a activtion fucniton module between:
        1. fc1: W1+b
        2. gelu
        3. fc2: W2+b
    '''
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self,  hidden_states: torch.Tensor)->torch.Tensor:
        '''
        input: (batch_size, num_patches, embed_dim)
        output: (batch_size, num_patches, embed_dim)
        '''
        # (batch_size, num_patches, embed_dim) --> (batch_size, num_patches, intermediate_size)
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # (batch_size, num_patches, intermediate_size) --> (batch_size, num_patches, embed_dim)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    '''
    Define the transformer encoder block
    Include two skip connection: 
        1. x + self_attention(layernorm(x))
        2. x + mlp(layernorm(x))
    '''
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layernorm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigliMLP(config)
        self.layernorm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor)->torch.Tensor:
        '''
        input: (batch_size, num_patches, embed_dim), from embedding layer (SiglipVisionEmbeddings) or last encoder block
        output: (batch_size, num_patches, embed_dim)
        '''
        # residual: (batch_size, num_patches, embed_dim)
        residual = hidden_states
        # (batch_size, num_patches, embed_dim) --> (batch_size, num_patches, embed_dim)
        hidden_states = self.layernorm1(hidden_states)
        # (batch_size, num_patches, embed_dim) --> (batch_size, num_patches, embed_dim)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # (batch_size, num_patches, embed_dim) --> (batch_size, num_patches, embed_dim)
        hidden_states = residual + hidden_states

        # residual: (batch_size, num_patches, embed_dim)
        residual = hidden_states
        # (batch_size, num_patches, embed_dim) --> (batch_size, num_patches, embed_dim)
        hidden_states = self.layernorm2(hidden_states)
        # (batch_size, num_patches, embed_dim) --> (batch_size, num_patches, embed_dim)
        hidden_states = self.mlp(hidden_states)
        # (batch_size, num_patches, embed_dim) --> (batch_size, num_patches, embed_dim)
        hidden_states = residual + hidden_states
        return hidden_states

class SiglipEncoder(nn.Module):
    '''
    Define the sequence of the SiglipEncoderLayer
    '''
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
            )
        # self.layernorm  = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, inputs_embeds: torch.Tensor)->torch.Tensor:
        '''
        input: (batch_size, num_patches, embed_dim), from the SiglipVisionEmbeddings layer
        output: (batch_size, num_patches, embed_dim)
        '''
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # hidden_states = self.layernorm(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    '''
    Define the structure of the SiglipVisionTransformer model
    Include: 
        1. embedding layer (SiglipVisionEmbeddings)
        2. transformer encoder
        3. layernorm layer
    '''
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor):
        # pix_values: [batch_size, channels, height, width] --> [batch_size, num_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_states = self.encoder(inputs_embeds = hidden_states)

        last_hidden_states = self.post_layernorm(last_hidden_states)

        return last_hidden_states

class SiglipVisionModel(nn.Module):
    '''
    Build the SiglipVisionModel by calling the pre-defined SiglipVisionTransformer class
    '''
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor):
        '''
        Take a batch of image and return a batch of list of embedding
        input: [batch_size, channels, height, width]
        output: [batch_size, num_patches, embed_dim]
        '''
        # [batch_size, channels, height, width] --> [batch_size, num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)

