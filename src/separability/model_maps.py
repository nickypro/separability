# Code mostly from TransformerLens
# https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py
import torch
import einops

from transformers.models.gpt2.modeling_gpt2 import GPT2Block

def get_qkv_weights_gpt2(self,
        gpt_layer: GPT2Block,
    ):
    state_dict = {}

    # In GPT-2, q,k,v are produced by one big linear map, whose output is
    # concat([q, k, v])
    W = gpt_layer.attn.c_attn.weight
    W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)

    W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=self.n_heads)
    W_K = einops.rearrange(W_K, "m (i h)->i m h", i=self.n_heads)
    W_V = einops.rearrange(W_V, "m (i h)->i m h", i=self.n_heads)

    state_dict["W_Q"] = W_Q
    state_dict["W_K"] = W_K
    state_dict["W_V"] = W_V

    qkv_bias = gpt_layer.attn.c_attn.bias
    qkv_bias = einops.rearrange(
        qkv_bias,
        "(qkv index head)->qkv index head",
        qkv=3,
        index=self.n_heads,
        head=self.d_head,
    )
    state_dict["b_Q"] = qkv_bias[0]
    state_dict["b_K"] = qkv_bias[1]
    state_dict["b_V"] = qkv_bias[2]

    return state_dict

def set_qkv_weights_gpt2(self,
        gpt_layer: GPT2Block,
        updated_state_dict: dict
    ):
    # not we undo what we did in the previous function
    W_Q = updated_state_dict["W_Q"]
    W_K = updated_state_dict["W_K"]
    W_V = updated_state_dict["W_V"]

    W_Q = einops.rearrange(W_Q, "i m h->m (i h)", i=self.n_heads)
    W_K = einops.rearrange(W_K, "i m h->m (i h)", i=self.n_heads)
    W_V = einops.rearrange(W_V, "i m h->m (i h)", i=self.n_heads)

    W = torch.cat([W_Q, W_K, W_V], dim=1)

    q_bias = updated_state_dict["b_Q"]
    k_bias = updated_state_dict["b_K"]
    v_bias = updated_state_dict["b_V"]
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=1)
    qkv_bias = einops.rearrange(
        qkv_bias,
        "qkv index head->(qkv index head)",
        qkv=3,
        index=self.n_heads,
        head=self.d_head,
    )

    # Update Q K V based on state dict
    attn_state_dict = {"weight": W, "bias": qkv_bias}
    gpt_layer.attn.c_attn.load_state_dict(attn_state_dict)

    return

opt_map = {
    "embed.W_E"       : "decoder.embed_tokens.weight",
    "pos_embed.W_pos" : "decoder.embed_positions",
    # in OPT, ln_final is only added for backwards compatibility
    "ln_final.w"      : "decoder.final_layer_norm.weight",
    "ln_final.b"      : "opt.model.decoder.final_layer_norm.bias",
    "unembed.W_U"     : "opt.lm_head.weight.T",
    "unembed.b_U"     : None,
}

opt_layer_map = {
        "ln1.w"     : "self_attn_layer_norm.weight",
        "ln1.b"     : "self_attn_layer_norm.bias",

        "attn.W_Q"  : "self_attn.q_proj.weight",
        "attn.W_K"  : "self_attn.k_proj.weight",
        "attn.W_V"  : "self_attn.v_proj.weight",
        "attn.b_Q"  : "self_attn.q_proj.bias",
        "attn.b_K"  : "self_attn.k_proj.bias",
        "attn.b_V"  : "self_attn.v_proj.bias",

        "attn.W_O"  : "self_attn.out_proj.weight",
        "attn.b_O"  : "self_attn.out_proj.bias",

        "ln2.w"     : "final_layer_norm.weight",
        "ln2.b"     : "final_layer_norm.bias",

        "mlp.W_in"  : "fc1.weight",
        "mlp.W_out" : "fc2.weight",

        "mlp.b_in"  : "fc1.bias",
        "mlp.b_out" : "fc2.bias",
}
