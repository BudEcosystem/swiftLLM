import torch
from swiftllm.worker.kernels.quick_gelu import quick_gelu
from swiftllm.worker.kernels.linear import linear

def get_clip_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0
    return image_size // patch_size

def get_clip_num_patches(image_size: int, patch_size: int) -> int:
    grid_length = get_clip_patch_grid_length(image_size=image_size,
                                             patch_size=patch_size)
    return grid_length * grid_length

class CLIPEncoderLayer:
    def __init__(self, model_config, weight, layer_id):
        self.weight = weight
        self.model_config = model_config
        self.layer_id = layer_id

        self.num_heads = model_config.num_q_heads
        self.head_dim = model_config.head_dim

    
    def forward(self, hidden_states):
        
        
        residual = hidden_states
        #layernorm_1
        hidden_states = torch.nn.functional.layer_norm(hidden_states, (self.model_config.hidden_size,), self.weight.layer_norm1)

        bsz, tgt_len, embed_dim = hidden_states.size()
        #attention
        q = linear(hidden_states, self.weight.q_proj)
        k = linear(hidden_states, self.weight.k_proj)
        v = linear(hidden_states, self.weight.v_proj)
        q = q.view(-1, self.model_config.num_q_heads,  self.model_config.head_dim)
        k = k.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)
        v = v.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1,2))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = torch.nn.functional.dropout(attn_weights)

        attn_output = torch.bmm(attn_probs, v)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        hidden_states = linear(attn_output, self.weight.o_proj)

        hidden_states = residual + hidden_states

        residual = hidden_states
        
        hidden_states = torch.nn.functional.layer_norm(hidden_states, (self.model_config.hidden_size,), self.weight.layer_norm2)
        #mlp
        hidden_states = linear(hidden_states, self.weight.fc1)
        quick_gelu(hidden_states)
        # hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = linear(hidden_states, self.weight.fc2)
        
        hidden_states = residual + hidden_states

        return hidden_states

class CLIPVisionEmbeddings:
    def __init__(self, model_config, weights):
        self.weights = weights
        self.model_config = model_config
        self.num_channels = 3

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        class_embedding = torch.nn.Parameter(torch.randn(self.model_config.hidden_size, device="cuda"))
        
        patch_embedding = torch.nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.model_config.hidden_size,
            kernel_size=self.model_config.patch_size,
            stride=self.model_config.patch_size,
            bias=False
        )
        # patch_embedding.load_state_dict(self.weights.vision_tower.vison_patch_embedding)
        patch_embedding.weight.data = self.weights.vision_tower.vison_patch_embedding.to('cuda')
        
        num_patches = get_clip_num_patches(image_size=self.model_config.image_size, patch_size=self.model_config.patch_size)
        num_positions = num_patches + 1
        position_ids = torch.arange(num_positions).expand((1, -1)).to('cuda')

        position_embedding = torch.nn.Embedding(num_positions, self.model_config.hidden_size)
        # position_embedding.load_state_dict(self.weights.vision_tower.vison_position_embedding)
        position_embedding.weight.data = self.weights.vision_tower.vison_position_embedding.to('cuda')
        patch_embeds = patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + position_embedding(position_ids)

        return embeddings.half()


class CLIPVisionModel:

    def __init__(self, model_config, weights):
        self.weights = weights
        self.model_config = model_config

        self.embedding = CLIPVisionEmbeddings(self.model_config, self.weights)

        self.encoder_layers = [
            CLIPEncoderLayer(
                self.model_config,
                self.weights.vision_tower.layers[layer_id],
                layer_id
            )
            for layer_id in range(self.model_config.num_layers)
        ]
    
    def forward(self, pixel_values):
        
        hidden_states = self.embedding.forward(pixel_values)
        
        hidden_states = torch.nn.functional.layer_norm(hidden_states, (self.model_config.hidden_size,), self.weights.vision_tower.vison_pre_layrnorm)
        
        for layer in self.encoder_layers:
            hidden_states = layer.forward(hidden_states)
        
        return hidden_states

class LlavaMultiModalProjector:

    def __init__(self, model_config, weights):
        self.model_config = model_config
        self.weights = weights

    def forward(self, image_features: torch.Tensor):
        
        hidden_states = linear(image_features, self.weights.multi_modal_projector.proj_linear_1)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = linear(hidden_states, self.weights.multi_modal_projector.proj_linear_2)
        
        return hidden_states
