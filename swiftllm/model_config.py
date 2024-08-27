import os
import json
import torch

class LlamaModelConfig:
    """
    The configuration of a LLaMA model (including LLaMA 1/2/3).
    """
    
    def __init__(
        self,
        model_config: dict
    ):
        """
        Initialize a LLaMA model configuration from a dict, which should be generated
        from a huggingface transformers config.json file.
        """
        
        assert model_config["model_type"] == "llava"
        self.num_layers = model_config["num_hidden_layers"]
        self.num_q_heads = model_config["num_attention_heads"]
        self.num_kv_heads = model_config.get("num_key_value_heads", self.num_q_heads)
        self.hidden_size = model_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_q_heads
        self.vocab_size = model_config["vocab_size"]
        self.max_position_embeddings = model_config["max_position_embeddings"]
        self.ffn_inter_dim = model_config["intermediate_size"]
        self.rotary_base = model_config.get("rope_theta", model_config.get("rotary_base", 10000))
        self.rms_norm_eps = model_config["rms_norm_eps"]
        self.rope_scaling = model_config.get("rope_scaling", 1.0)
        self.rope_theta = model_config.get("rope_theta", 10000)
        if self.rope_scaling is None:
            self.rope_scaling = 1.0
        assert model_config["hidden_act"] == "silu"

    def get_kvslot_size(self, dtype: torch.dtype = torch.float16) -> int:
        """
        Get the size of one kv slot (the kv cache of one token) (in bytes)
        """
        return (2 * self.num_layers * self.num_kv_heads * self.head_dim) * dtype.itemsize
    
    @staticmethod
    def load_from_model_path(model_path: str) -> "LlamaModelConfig":
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            model_config_dict = json.loads(f.read())
        return LlamaModelConfig(model_config_dict)

class LlavaVisionConfig:
    def __init__(self, vision_config):
        self.num_channels = 3
        self.hidden_size = vision_config['hidden_size']
        self.patch_size = vision_config['patch_size']
        self.image_size = vision_config['image_size']
        self.num_layers = vision_config['num_hidden_layers']
        self.num_q_heads = vision_config["num_attention_heads"]
        self.num_kv_heads = vision_config.get("num_key_value_heads", self.num_q_heads)
        self.head_dim = self.hidden_size // self.num_q_heads
        self.intermediate_size = vision_config["intermediate_size"]
        self.layer_norm_eps = 1e-05

class LlavaConfig:

    def __init__(self, model_config):

        self.text_config = {
            "_name_or_path": "vicuna-7b-v1.5",
            "architectures": [
                "LlamaForCausalLM"
            ],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pad_token_id": 0,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.31.0",
            "use_cache": True,
            "vocab_size": 32064
        }
        self.text_config = LlamaModelConfig(self.text_config)
        # self.vision_config = model_config['vision_config']
        self.vision_config = LlavaVisionConfig(model_config['vision_config'])

        self.vocab_size = model_config["vocab_size"]
        self.image_token_index = model_config["image_token_index"]

    
    def get_kvslot_size(self, dtype: torch.dtype = torch.float16) -> int:
        """
        Get the size of one kv slot (the kv cache of one token) (in bytes)
        """
        language_model = 2 * self.text_config.num_layers * self.text_config.num_kv_heads * self.text_config.head_dim
        vision_model = 2 * self.vision_config.num_layers * self.vision_config.num_kv_heads * self.vision_config.head_dim
        return (language_model + vision_model) * dtype.itemsize
    
    @staticmethod
    def load_from_model_path(model_path: str) -> "LlavaConfig":
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            model_config_dict = json.loads(f.read())
        return LlavaConfig(model_config_dict)
        
