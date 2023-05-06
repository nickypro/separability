# Code mostly from TransformerLens
# https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py
from typing import Callable, Any, Optional
from dataclasses import dataclass
import einops
from transformers import AutoConfig

@dataclass
class ConfigClass:
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    n_layers: int
    n_ctx: int
    eps: float
    d_vocab: int
    act_fn: str
    use_attn_scale: bool
    use_local_attn: bool
    scale_attn_by_inverse_layer_idx: bool
    normalization_type: str
    architecture: str
    tokenizer_name: str
    parallel_attn_mlp: bool = False
    positional_embedding_type: str = "standard"
    rotary_dim: Optional[int] = None

def convert_hf_model_config(official_model_name: str):
    """
    Returns the model config for a HuggingFace model, converted to a dictionary
    in the HookedTransformerConfig format.

    Takes the official_model_name as an input.
    """
    # Load HuggingFace model config
    if 'llama' not in official_model_name:
        hf_config = AutoConfig.from_pretrained(official_model_name)
        architecture = hf_config.architectures[0]
    else:
        architecture = "LLaMAForCausalLM"
    if 'llama-7b' in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif 'llama-13b' in official_model_name:
        cfg_dict = {
            "d_model": 5120,
            "d_head": 5120 // 40,
            "n_heads": 40,
            "d_mlp": 13824,
            "n_layers": 40,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 5120 // 40,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif 'llama-30b' in official_model_name:
        cfg_dict = {
            "d_model": 6656,
            "d_head": 6656 // 52,
            "n_heads": 52,
            "d_mlp": 17920,
            "n_layers": 60,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 6656 // 52,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif 'llama-65b' in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 8192 // 64,
            "n_heads": 64,
            "d_mlp": 22016,
            "n_layers": 80,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 8192 // 64,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "GPTNeoForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_heads,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "attn_types": hf_config.attention_layers,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": False,
            "use_local_attn": True,
            "window_size": hf_config.window_size,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPT2LMHeadModel":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_ctx,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "OPTForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.ffn_dim,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPTJForCausalLM":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": 4 * hf_config.n_embd,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.rotary_dim,
            "normalization_type": "LN",
        }
    elif architecture == "GPTNeoXForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "normalization_type": "LN",
        }
        rotary_pct = hf_config.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    # All of these models use LayerNorm
    cfg_dict["architecture"] = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg_dict["tokenizer_name"] = official_model_name
    return ConfigClass(**cfg_dict)


#####################################################################################
# Define Architecture Maps
#####################################################################################

# Meta OPT and Galactica Models
###############################

opt_model_map = {
    "model"           : "model",
    "layers"          : "model.decoder.layers",
    "embed"           : "model.decoder.embed_tokens",
    "embed.W_E"       : "model.decoder.embed_tokens.weight",
    "pos_embed.W_pos" : "model.decoder.embed_positions",
    # in OPT, ln_final is only added for backwards compatibility
    "ln_final.w"      : "model.decoder.final_layer_norm.weight",
    "ln_final.b"      : "model.decoder.final_layer_norm.bias",
    "unembed.W_U"     : "lm_head.weight.T",
    "unembed.b_U"     : None,
}

opt_layer_map = {
    "ln1"           : "self_attn_layer_norm",
    "ln1.w"         : "self_attn_layer_norm.weight",
    "ln1.b"         : "self_attn_layer_norm.bias",

    "attn"          : "self_attn",
    "attn.q_proj"   : "self_attn.q_proj",
    "attn.k_proj"   : "self_attn.k_proj",
    "attn.v_proj"   : "self_attn.v_proj",

    "attn.W_Q"      : "self_attn.q_proj.weight",
    "attn.W_K"      : "self_attn.k_proj.weight",
    "attn.W_V"      : "self_attn.v_proj.weight",
    "attn.b_Q"      : "self_attn.q_proj.bias",
    "attn.b_K"      : "self_attn.k_proj.bias",
    "attn.b_V"      : "self_attn.v_proj.bias",

    "attn.out_proj" : "self_attn.out_proj",
    "attn.W_O"      : "self_attn.out_proj.weight",
    "attn.b_O"      : "self_attn.out_proj.bias",

    "attn.inv_out_proj" : "self_attn.out_proj",
    "attn.W_O_inv"  : "self_attn.inv_out_proj.weight",
    "attn.b_O_inv"  : "self_attn.inv_out_proj.bias",

    "ln2"           : "final_layer_norm",
    "ln2.w"         : "final_layer_norm.weight",
    "ln2.b"         : "final_layer_norm.bias",

    "fc1"           : "fc1",
    "mlp.W_in"      : "fc1.weight",
    "mlp.b_in"      : "fc1.bias",

    "activation_fn" : "activation_fn",

    "fc2"           : "fc2",
    "mlp.W_out"     : "fc2.weight",
    "mlp.b_out"     : "fc2.bias",
}


# GPT NEO X and Pythia Models
#############################

gpt_neox_model_map = {
    "model"           : "base_model",
    "layers"          : "base_model.layers",
    "embed"           : "base_model.embed_in",
    "embed.W_E"       : "base_model.embed_in.weight",
    "pos_embed.W_pos" : "base_model.embed_pos.weight",
    "ln_final.w"      : "base_model.final_layer_norm.weight",
    "ln_final.b"      : "base_model.final_layer_norm.bias",
    "unembed.W_U"     : "base_model.embed_out.weight",
    "unembed.b_U"     : "base_model.embed_out.bias",
}

def build_gpt_neox_layer_map(cfg):
    def gpt_neox_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        qkv_heads = layer.attention.query_key_value
        W = qkv_heads.weight
        #W = einops.rearrange(W, "(i qkv h) m->qkv i m h", i=cfg.n_heads, qkv=3)
        W = einops.rearrange(W, "(i qkv h) m->qkv m (i h)", i=cfg.n_heads, qkv=3)
        qkv_map = {"q": 0, "k": 1, "v": 2}
        index = qkv_map[key]

        # Get mode
        if inpt is None:
            return W[index]

        # Set mode
        params = qkv_heads.state_dict()
        W[index] = inpt
        W = einops.rearrange(W, "qkv m (i h) -> (i qkv h) m", i=cfg.n_heads, qkv=3)
        params["weight"] = W
        qkv_heads.load_state_dict(params)

    def gpt_neox_qkv_bias(layer, key: str, inpt: Optional[Any]=None):
        qkv_head = layer.attention.query_key_value
        qkv_bias = qkv_head.bias
        qkv_bias = einops.rearrange(
            qkv_bias, "(index qkv head)->qkv (index head)", qkv=3, index=cfg.n_heads,
        )
        qkv_map = {"q": 0, "k": 1, "v": 2}
        index = qkv_map[key]

        # Get mode
        if inpt is None:
            return qkv_bias[index]

        # Set mode
        params = qkv_head.state_dict()
        qkv_bias[index] = inpt
        qkv_bias = einops.rearrange(
            qkv_bias, "qkv (index head) -> (index qkv head)", qkv=3, index=cfg.n_heads,
        )
        params["bias"] = qkv_bias
        qkv_head.load_state_dict(params)

    gpt_neox_layer_map = {
        "ln1"       : "input_layernorm",
        "ln1.w"     : "input_layernorm.weight",
        "ln1.b"     : "input_layernorm.bias",

        "attn"      : "attention",
        "attn.q_proj"   : None,
        "attn.k_proj"   : None,
        "attn.v_proj"   : None,

        "attn.W_Q"  : lambda layer, inpt=None: gpt_neox_qkv_weight(layer, "q", inpt),
        "attn.W_K"  : lambda layer, inpt=None: gpt_neox_qkv_weight(layer, "k", inpt),
        "attn.W_V"  : lambda layer, inpt=None: gpt_neox_qkv_weight(layer, "v", inpt),
        "attn.b_Q"  : lambda layer, inpt=None: gpt_neox_qkv_bias(layer, "q", inpt),
        "attn.b_K"  : lambda layer, inpt=None: gpt_neox_qkv_bias(layer, "q", inpt),
        "attn.b_V"  : lambda layer, inpt=None: gpt_neox_qkv_bias(layer, "q", inpt),

        "attn.out_proj" : "attention.dense",
        "attn.W_O"      : "attention.dense.weight",
        "attn.b_O"      : "attention.dense.bias",

        "attn.inv_out_proj" : "attention.inv_out_proj",
        "attn.W_O_inv"      : "attention.inv_out_proj.weight",
        "attn.b_O_inv"      : "attention.inv_out_proj.bias",

        "ln2"       : "post_attention_layernorm",
        "ln2.w"     : "post_attention_layernorm.weight",
        "ln2.b"     : "post_attention_layernorm.bias",

        "fc1"       : "mlp.dense_h_to_4h",
        "fc2"       : "mlp.dense_4h_to_h",
        "mlp.W_in"  : "mlp.dense_h_to_4h.weight",
        "mlp.W_out" : "mlp.dense_4h_to_h.weight",
        "mlp.b_in"  : "mlp.dense_h_to_4h.bias",
        "mlp.b_out" : "mlp.dense_4h_to_h.bias",
        "activation_fn" : "mlp.act",
    }
    return gpt_neox_layer_map

#####################################################################################
# Build Model Layer Map interfaces
#####################################################################################

# Define Helper Functions
#########################

def get_attrs(obj, attr_string):
    nested_attributes = attr_string.split('.')
    current_attr = obj
    for attr_name in nested_attributes:
        current_attr = getattr(current_attr, attr_name)
    return current_attr

def get_model_key_map(config):
    architecture = config.architecture
    if architecture == "OPTForCausalLM":
        return opt_model_map
    if architecture == "GPTNeoXForCausalLM":
        return gpt_neox_model_map

    raise NotImplementedError(f"Architecture {architecture} not implemented")

def get_layer_key_map(config):
    architecture = config.architecture

    if architecture == "OPTForCausalLM":
        return opt_layer_map
    if architecture == "GPTNeoXForCausalLM":
        return build_gpt_neox_layer_map(config)

    raise NotImplementedError(f"Architecture {architecture} not implemented")


# Define Real Model Map and Layer Maps
######################################

class ModelMap:
    def __init__(self, model, cfg):
        self.cfg         = cfg
        self.model       = model
        self.key_map     = get_model_key_map(cfg)

        # Handle layers
        self.orig_layers = self["layers"]
        self.layers = [
            ModelLayerMap(self.cfg, layer) for layer in self.orig_layers
        ]

    def __getitem__(self, __name: str):
        key = self.key_map[__name]
        return get_attrs(self.model, key)

    def __setitem__(self, key, inpt):
        keys = key.split('.')
        attr = keys[-1]
        module = get_attrs(self.model, ".".join(keys[:-1]))
        params = module.state_dict()
        params[attr] = inpt
        module.load_state_dict(params)

class ModelLayerMap:
    def __init__(self, cfg, layer):
        self.cfg   = cfg
        self.layer = layer
        self.key_map = get_layer_key_map(cfg)

    def __getitem__(self, __name):
        key = self.key_map[__name]

        if isinstance(key, str):
            return get_attrs(self.layer, key)

        if isinstance(key, Callable):
            return key(self.layer)

    def __setitem__(self, __name: str, __value: Any) -> None:
        key = self.key_map[__name]
        if isinstance(key, Callable):
            return key(self.layer, __value)

        if not isinstance(key, str):
            raise ValueError("Invalid key, must be string or callable")

        # Get the module and attribute name
        keys = key.split('.')
        module = get_attrs(self.layer, ".".join(keys[:-1]))
        attr = keys[-1]

        if attr == "inv_out_proj":
            setattr(module, attr, __value)
            return

        # If setting an attribute of a module (eg: weights or biases), update
        params = module.state_dict()
        params[attr] = __value
        module.load_state_dict(params)
        return

    def __str__(self):
        out_str  = "Wrapper for Transformer Layer\n"
        out_str += self.key_map.keys().__str__()
        out_str += "\nOriginal Layer Structure:\n"
        out_str += self.layer.__str__()
        return out_str
