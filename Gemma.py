import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import math
from SigLip import SiglipVisionConfig, SiglipVisionModel


########################### Model Configuation ###########################
class KVCache():
    '''
    Store the keys and values tensor when in inference part to reduce the computation workload.
    In the inference part, we only care about the last position of the hidden state since it generate the next token.
    So in each time step, we only input the current single token (output from last inference) instead of all toknes including the previous tokens.
    And the linear projection will generate the corresponding q, k, v for this single token as normal. 
    Before the attention mechanism, q is for single tokens, while k/v will extract the previous k/v states for previous tokens 
    and then append the current k/v state as the new k/v state.

    In attention mechanism, assume current time step is t, then Q is the query for the current token, 
    K/V are the keys/values for the previous t-1 tokens extract from the KV-cache and append the current key/value token:
        Q(t): (..., ..., 1, head_dim)
        K(t): (..., ..., t, head_dim) = K(t-1).append(k_t) on seq_len dim
        V(t): (..., ..., t, head_dim) = V(t-1).append(v_t) on seq_len dim
    Once the inference done, it generate the (t+1)-th token, then this token will be as input for the next inference:
        Q(t+1): (..., ..., 1, head_dim)
        K(t+1): (..., ..., t+1, head_dim) = K(t).append(k_t+1) on seq_len dim
        V(t+1): (..., ..., t+1, head_dim) = V(t).append(v_t+1) on seq_len dim
    '''
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,  # (batch_size, num_KV_heads, seq_len, head_dim)
        value_states: torch.Tensor,# (batch_size, num_KV_heads, seq_len, head_dim)
        layer_idx: int, # The layer index, each layer need to store its own KV cache
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        First recognize if there exist created KV-Cache for current layer, if no, append it to the empty list and return it.
        If exist created KV-Cache, then first append the current K,V state into it and then return it for attention computation
        '''
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            # Usually the first input is a prompt, which means seq_len > 1,
            # then just prefill these tokens k/v states into the cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones on the seq_len dim
            # each tensor has shape: (batch_size, num_KV_heads, seq_len, head_dim)
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
class GemmaConfig():
    '''
    The configuration of the GemmaModel
    '''
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size, # In the feed forward layer
        num_hidden_layers,
        # Group query attention, which have different number of head for query, key and value
        num_attention_heads, # The number of head for query
        num_key_value_heads, # The number of head for key
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
    '''
    The configuration of the PaliGemmaForConditionalGeneration
    '''
    def __init__(self,
                 vision_config = None,
                 text_config = None,
                 ignore_index = -100, # Used as the label when training
                 image_token_index = 256000, # ids of "<image>"
                 vocab_size = 257152,
                 projection_dim = 2048, # The output size of the linear projector layer, the same as the hidden_size in Gemma
                 hidden_size = 2048, # The embedding dim for the language model
                 pad_token_id = None,
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.text_config = text_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id = pad_token_id)
        self.vocab_size = self.text_config.vocab_size
    
        # The number of patches for each image
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

########################### Model Architecture ###########################

class PaliGemmaMultiModelProjector(nn.Module):
    '''
    The PaliGemmaMultiModelProjector layer applied after SiglipVisionModel
    (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, projection_dim or hidden_size) 768 -> 2048
    '''
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class GemmaRMSNorm(nn.Module):
    '''
    The RMS-Normalization layer for Gemma model
    '''
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim)) # The learnable parameter, acts as a scaling factor after normalization

    def _norm(self, x):
        '''
        Takes an input tensor x and applies root mean square (RMS)
        '''
        mean_square = x.pow(2).mean(-1, keepdim=True) # The mean of square on the last dimention, keep the original dimention
        return x * torch.rsqrt(mean_square + self.eps) # 1 / sqrt(...), add self.eps avoid division by zero.

    def forward(self, x):
        '''
        First normalize the tensor, then adapt it with a learnable scaling factor
        (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, embedding_dim)
        '''
        output = self._norm(x.float()) # The normalized output
        output = output * (1.0 + self.weight.float()) # The normalized output adjust by a learnable scaling factor, use (1 + scaling_factor) * normalized_output is becasue we initial the scaling_factor as zero tensor
        return output.type_as(x)
    
class GemmaRotaryEmbedding(nn.Module):
    '''
    Realize the rotary position embedding in paper https://arxiv.org/pdf/2104.09864 formula (34)
    '''
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim # head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base # 

        # Calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, ..., dim // 2
        '''
        Long-term decay: 
        theta_i = 1 / base^(2*i/dim) where i = 0, 1, 2, ..., dim // 2, or theta_i = 1 / base^(i/dim) where i = 0, 2, 4, 6..., dim
        This setting provides a long-term decay property, which means the inner-product will decay when
        the relative position increase, i.e., a pair of tokens with a long relative distance
        should have less connection.
        '''
        # Theta: [theta_1, theta_2, ...theta_d/2], which shape is (head_dim // 2)
        Theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)) # (self.dim // 2)
        self.register_buffer("Theta", tensor=Theta, persistent=False) # No training, no save in state_dict()

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        '''
        Compute the cos(m), sin(m) term in formula (34) in paper, m is the position of the token in seq_len dimension
        x: (batch_size, num_head, seq_len, head_dim), we only use its device, dtype here
        position_ids: (batch_size, seq_len)
        '''
        self.Theta.to(x.device)
        # Copy the Theta tensor for batch in the sequence: (head_dim // 2) -> (batch_size, head_dim // 2, 1)
        Theta_expanded = self.Theta[None, :, None].float().expand(position_ids.shape[0], -1, 1) # [[...], [theta_1, theta_2, ...theta_d/2], 1]
        # position_ids_expanded: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        position_ids_expanded = position_ids[:, None, :].float() # [[...], 1, [1, 2, 3, ... , seq_len]]
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            '''
            freqs: (batch_size, head_dim // 2, 1) @ (batch_size, 1, seq_len) --> (batch_size, seq_len, head_dim // 2)
            For each item in batch, consider a column [theta_1, theta_2, ...theta_d/2] times a row [1, 2, 3, ... , seq_len], which get a (head_dim // 2, seq_len) matrix.
            Then transpose, so get (seq_len, head_dim // 2) matrix:
                [[1*theta_1, 1*theta_2, 1*theta_3, ..., 1*theta_d/2],
                 [2*theta_1, 2*theta_2, 2*theta_3, ..., 2*theta_d/2],
                 [3*theta_1, 3*theta_2, 3*theta_3, ..., 3*theta_d/2],
                 [4*theta_1, 4*theta_2, 4*theta_3, ..., 4*theta_d/2],
                 ...
                 [seq_len*theta_1, seq_len*theta_2, seq_len*theta_3, ..., seq_len*theta_d/2]]
            '''
            freqs = (Theta_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: （batch_Size, seq_len, head_dim）
            emb = torch.cat((freqs, freqs), dim=-1) 
            '''
            Difference from the paper:
            In paper: [m*theta_1, m*theta_1, m*theta_2, m*theta_2,  ...]
            But here: [m*theta_1, m*theta_2, m*theta_3, ..., m*theta_1, m*theta_2, m*theta_3, ...]
            Reason: in the llama huggingface, they permute the weight (W_q, W_k) for sliced rotary
                    https://github.com/huggingface/transformers/issues/25199#issuecomment-1687720247
            Hence 
                [[1*theta_1, 1*theta_2, 1*theta_3, ..., 1*theta_d/2, 1*theta_1, 1*theta_2, 1*theta_3, ..., 1*theta_d/2],
                 [2*theta_1, 2*theta_2, 2*theta_3, ..., 2*theta_d/2, 2*theta_1, 2*theta_2, 2*theta_3, ..., 2*theta_d/2],
                 [3*theta_1, 3*theta_2, 3*theta_3, ..., 3*theta_d/2, 3*theta_1, 3*theta_2, 3*theta_3, ..., 3*theta_d/2],
                 [4*theta_1, 4*theta_2, 4*theta_3, ..., 4*theta_d/2, 4*theta_1, 4*theta_2, 4*theta_3, ..., 4*theta_d/2],
                 ...
                 [seq_len*theta_1, seq_len*theta_2, seq_len*theta_3, ..., seq_len*theta_d/2, seq_len*theta_1, seq_len*theta_2, seq_len*theta_3, ..., seq_len*theta_d/2]]
            '''
            # cos, sin: （batch_size, seq_len, head_dim）
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
    @staticmethod
    def rotate_half(x):
        '''
        x: (batch_size, num_Q_heads, seq_len, head_dim)
        According to https://github.com/huggingface/transformers/issues/25199#issuecomment-1687720247, the rotate for input x is changed.
        Hence
            [[-x_d/2+1, -x_d/2+2, -x_d/2+3, ..., x_0, x_1, x_2, x_3, ...],
             [-x_d/2+1, -x_d/2+2, -x_d/2+3, ..., x_0, x_1, x_2, x_3, ...],
             [-x_d/2+1, -x_d/2+2, -x_d/2+3, ..., x_0, x_1, x_2, x_3, ...],
             ...
             [-x_d/2+1, -x_d/2+2, -x_d/2+3, ..., x_0, x_1, x_2, x_3, ...]]
        
        '''
        # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
        x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
        x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        '''
        Apply the formula (34) in paper.
        q: (batch_size, num_Q_heads, seq_len, head_dim)
        v: (batch_size, num_KV_heads, seq_len, head_dim)
        cos, sin: (batch_size, seq_len, head_dim）
        '''
        # Add the head dimension
        cos = cos.unsqueeze(unsqueeze_dim) # (batch_size, 1, seq_len, head_dim）
        sin = sin.unsqueeze(unsqueeze_dim) # (batch_size, 1, seq_len, head_dim）
        
        q_embed = (q * cos) + (GemmaRotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (GemmaRotaryEmbedding.rotate_half(k) * sin)
        return q_embed, k_embed

class GemmaMLP(nn.Module):
    '''
    The MLP layer for Gemma model
    Include 3 nn.Linear() layers and an activtion fucniton module:
        1. gate_proj: (hidden_size, intermediate_size)
        2. up_proj: (hidden_size, intermediate_size)
        3. down_proj: (intermediate_size, hidden_size)
        4. nn.functional.gelu

    Also include a Gating Mechanism:
        The use of a gating mechanism (gate_proj output followed by GELU, then multiplied by up_proj output) allows the network to control the flow of information more dynamically. 
        It can decide how much of the transformed input should influence the final output.
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        '''
        input: (batch_size, seq_len, hidden_size) -> output: (batch_size, seq_len, hidden_size)
        '''
        # Equivalent to: self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        y = F.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return z

class GemmaAttention(nn.Module):
    '''
    The attention layer of the Gemma, which applies Grouped Query Attention.
    The motivation is: 
        The bottleneck of the attention mechanism computation is not in how many dot product we are doing,
        but how much time it takes to copy the memory from the high bandwidth memory to the local memory. 
    The implementation is:
        Select several heads of key, value, then divide the all heads of query into group such that in each group, these heads of quary share the same key, value. 
        Then in same group of query dot product compution, only need to copy key, value to local memory once.
    Eg:
        Query: Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8
        Key: K1, K2
        Value: V1, V2
        Then, Q1, Q2, Q3, Q4 share K1, V1; Q5, Q6, Q7, Q8 share K2, V2
    Btw, this approach also reduce the size of KV cache
    '''
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads # number of the head for query, eg: 8
        self.head_dim = config.head_dim # head dimension
        self.num_key_value_heads = config.num_key_value_heads # number of the head for key, value, eg: 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # number of query in each group, eg: 4
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0   

        # Number of heads = 8
        # Hidden_Size = 1024
        # Head_dim = 1024 / 8 = 128
        # Wq: [1024, 8 * 128] = [1024, 1024]
        # Wk: [1024, 2 * 128] = [1024, 256]
        # Wv: [1024, 2 * 128] = [1024, 256]
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        '''
        Expend the tensor from (batch_size, num_KV_heads, seq_len, head_dim) to (batch_size, num_Q_heads, seq_len, head_dim) 
        Note that n_rep = num_Q_heads // num_KV_heads
        '''
        batch_size, num_KV_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_KV_heads, n_rep, seq_len, head_dim) # Create a new dimension and repeat the tensor after this position
        return hidden_states.reshape(batch_size, num_KV_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor, # (batch_size, seq_len, hidden_size) from the RMSnormal layer
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size() # [Batch_Size, Seq_Len, Hidden_Size]

        query_states = self.q_proj(hidden_states) # (batch_size, seq_len, num_Q_heads * head_dim)
        key_states = self.k_proj(hidden_states) # (batch_size, seq_len, num_KV_heads * head_dim)
        value_states = self.v_proj(hidden_states) # (batch_size, seq_len, num_KV_heads * head_dim)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_Q_heads, seq_len, head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # (batch_size, num_KV_heads, seq_len, head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # (batch_size, num_KV_heads, seq_len, head_dim)

        # Apply Rotary Embedding
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None) # (batch_Size, seq_len, head_dim), (batch_size, seq_len, head_dim)
        query_states, key_states = GemmaRotaryEmbedding.apply_rotary_pos_emb(query_states, key_states, cos, sin) # (batch_size, num_Q_heads, seq_len, head_dim), (batch_size, num_KV_heads, seq_len, head_dim)

        # Apply KV cache
        if kv_cache is not None:
            '''
            Prefilling phase: 
                When inference, the initial input is a prompt, which is a list of tokens (include image and text),
                then the seq_len > 1 for key_states, value_states. All tokens' key_states, value_states will prefill into the KV cache.
            Token generation phase:
                After prefilling, each input is one token, then the seq_len = 1 for key_states, value_states. 
                The current token's key_state, value_state will append into the KV cache.
            '''
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Repeat the key and values to match the number of heads of the query
        '''
        Since we don't have custom cuda kernal to realize the real Grouped Query Attention,
        we are actually pretend that we use the same key/value for the grouped query. We need to make the key/value have the same head_dim as query.
        '''
        key_states = GemmaAttention.repeat_kv(key_states, self.num_key_value_groups) # (batch_size, num_Q_heads, seq_len, head_dim)
        value_states = GemmaAttention.repeat_kv(value_states, self.num_key_value_groups) # (batch_size, num_Q_heads, seq_len, head_dim)

        # Perform the calculation as usual, Q * K^T / sqrt(head_dim). 
        # (batch_size, num_Q_heads, seq_len, head_dim) * (batch_size, num_Q_heads, head_dim, seq_len) / scalar - > (batch_size, num_Q_heads, seq_len, seq_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Apply the softmax on the last dim
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply the dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # Multiply by the values.
        # (batch_size, num_Q_heads, seq_len, seq_len) * (batch_size, num_Q_heads, seq_len, head_dim) -> (batch_size, num_Q_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # (batch_size, num_Q_heads, seq_len, head_dim) -> (batch_size, seq_len, num_Q_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (batch_size, seq_len, num_Q_heads, head_dim) -> (batch_size, seq_len, num_Q_heads * head_dim)
        attn_output = attn_output.view(batch_size, seq_len, -1)
        # Multiply by W_o. 
        attn_output = self.o_proj(attn_output) # (batch_size, seq_len, hidden_size)

        return attn_output, attn_weights
        
class GemmaDecoderLayer(nn.Module):
    '''
    The decoder block for Gemma model
    Include two skip connection:
        1. x + self_attention(layernorm(x))
        2. x + mlp(layernorm(x)) 
    '''
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # Normalization before attention layer
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # Normalization before feed forward layer

    def forward(
        self,
        hidden_states: torch.Tensor, # (batch_size, seq_len, hidden_size)
        attention_mask: Optional[torch.Tensor] = None, # attention mask that send to the attention mechanism
        position_ids: Optional[torch.LongTensor] = None, # The Rotary Positional Encoding, also send to attention mechanism
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    '''
    The Gemma decoder, include:
    1. Embedding layer
    2. A list of decoder block
    3. GemmaRMSNorm normalization layer
    '''
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        '''
        Return the defined Embedding layer nn.Embedding(vocab_size, hidden_size, padding_idx)
        '''
        return self.embed_tokens
    
    def forward(self,
                 attention_mask: Optional[torch.Tensor]=None,
                 position_ids: Optional[torch.Tensor]=None, # The Rotary Positional Encoding, applied during the calculation of the attention
                 inputs_embeds: Optional[torch.FloatTensor] = None, # images feature + text tokens
                 kv_cache: Optional[KVCache] = None,
                )->Tuple:
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        '''
        To make the magnitude the same even if the model dimention increase, normalize the embedding
        '''
        normalizer = torch.tensor(self.config.hidden_size**(-0.5), dtype=hidden_states.dtype) 
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states



class GemmaForCausalLM(nn.Module):
    '''
    Assemble the Gemma language model (GemmaModel) and language model head (lm_head) which project embedding into logits
    '''
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GemmaModel(config) # The Gemma language model
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # The language model head

    def get_input_embeddings(self):
        '''
        The nn.Embedding(vocab_size, hidden_size, padding_idx) object
        '''
        return self.model.embed_tokens
    
    def tie_weights(self):
        '''
        Copy the weight matrix from embedding layer to the language model head
        '''
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self,
                 attention_mask: Optional[torch.Tensor]=None,
                 position_ids: Optional[torch.Tensor]=None,
                 inputs_embeds: Optional[torch.FloatTensor] = None,
                 kv_cache: Optional[KVCache] = None,
                )->Tuple:
        # inputs_embeds: (batch_size, seq_len, hidden_size) -> outputs: (batch_size, seq_len, hidden_size)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            'logits': logits # (batch_size, seq_len, vocab_size)
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data
    
class PaliGemmaForConditionalGeneration(nn.Module):
    '''
    The PaliGemma model itself, including:
        - 1. The SiglipVisionModel as the Contrastive Vision Encoder
        - 2. The PaliGemmaMultiModelProjector fed by the output of SiglipVisionModel to align the embedding dim
        - 3. The GemmaForCausalLM as the language model
    '''
    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config) # The Contrastive Vision Encoder
        self.multi_model_projector = PaliGemmaMultiModelProjector(config) # The linear projection connect the Contrastive Vision Encoder, to align the embedding dim of the patch the same as the language model
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config) # The transformer decoder 
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1 

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(self, 
                                             image_features: torch.Tensor, # (batch_size, num_patches, hidden_size), ready to insert
                                             inputs_embeds: torch.Tensor, # (batch_size, seq_len, hidden_size)
                                             input_ids: torch.Tensor,  # (batch_size, seq_len), include the placeholder token for image
                                             attention_mask: torch.Tensor, 
                                             kv_cache: Optional[KVCache]=None):
        '''
        1. Put the image embedding from image_features into the placeholder <image> in inputs_embeds
        2. Create the attention mask: 
            softmax(att_matrix + mask): mask = 0 means not mask, -inf means mask
        '''
        _, _, embed_dim = image_features.shape # Extract the embedding dim
        batch_size, seq_len = input_ids.shape # Extract the batch_size and seq_len
        # dtype, device = inputs_embeds.dtype, inputs_embeds.device
        scaled_image_features = image_features / (self.config.hidden_size**0.5) # To keep the same magnitude of the feature in image when changing the embedding size
    
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, seq_len, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # text_mask: (batch_size, seq_len), true for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.config.pad_token_id)
        # image_mask: (batch_size, seq_len), true for image tokens
        image_mask = input_ids == self.config.image_token_index # Locate the indice where "<image>"
        # pad_mask: (batch_size, seq_len), true for padding tokens
        pad_mask = input_ids == self.config.pad_token_id

        # Expend the mask to the embedding dim otherwise we can't use them in torch.where
        '''
        unsqueeze(-1): Add an extra dimension at the last axis: (batch_size, seq_len) -> (batch_size, seq_len, 1)
        expand(-1, -1, embed_dim): -1 means not changing the size of that dimension, expand the last dim into embed_dim
        -> (batch_size, seq_len, embed_dim)
        '''
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embedding
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### Create the attention mask ####
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len =  inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            '''
            Prefill Phase
            '''
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device # 0 means no mask
            )
        else:
            '''
            Token Generation Phase
            '''
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device # 0 means no mask
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids
 
    def forward(self, 
                input_ids: torch.LongTensor=None, # input_ids from PaliGemmaProcessor
                pixel_values: torch.FloatTensor=None,  # pixel_values from PaliGemmaProcessor
                attention_mask: Optional[torch.Tensor]=None, # attention_mask from PaliGemmaProcessor
                kv_cache: Optional[KVCache]=None,
                )->Tuple:
        '''
        Take the input tensors from the PaliGemmaProcessor: 'pixel_values', 'input_ids', 'attention_mask'
        '''
        assert torch.all(attention_mask == 1), "The input can not be padded" 

        # 1. Extra the input embeddings
        # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        '''
        We define the get_input_embeddings() method in the GemmaModel language model, which returrn a nn.Embedding(vocab_size, hidden_size) object.
        Then call it with input_ids as it is the Embedding layer
        '''
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids) 

        # 2. Merge text and image
        # (batch_size, channels, height, width) -> (batch_size, num_patches, embed_dim)
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype)) # The forward method in SiglipVisionModel
        #  (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, hidden_size)
        image_features = self.multi_model_projector(selected_image_feature) # Align the image embedding dim the same as the language model

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        # The forward method for Gemma language model
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache
        )
        return outputs
