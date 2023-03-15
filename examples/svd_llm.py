import torch
from separability import Model
from separability.nn import mlp_svd_two_layer, InverseLinear

def check_inverse_linear_bias():
    with torch.no_grad():
        opt = Model("facebook/opt-125m", limit=1000, use_accelerator=False)
        attn = opt.model.decoder.layers[0].self_attn
        v_proj = attn.v_proj
        out_proj = attn.out_proj
        device = v_proj.weight.device
        inv_out_proj = InverseLinear(out_proj).to(device)

        vector = torch.randn((opt.d_model)).to(device)
        output = out_proj(v_proj(vector))

        v_bias = torch.tensor(v_proj.bias).to(device)
        v_bias_out = out_proj(v_bias)
        v_bias_restored = inv_out_proj(v_bias_out)

        print( v_bias[:5] )
        print( v_bias_restored[:5] )

        params = v_proj.state_dict()
        params.update({'bias': v_bias_restored})
        v_proj.load_state_dict(params)

        new_output = out_proj(v_proj(vector))
        print(output[:5])
        print(new_output[:5])

# check_inverse_linear_bias()

def svd_model_attention(use_zero_biases=False):
    with torch.no_grad():
        opt = Model("facebook/opt-125m", limit=1000, use_accelerator=False, dtype=torch.float16)
        attn = opt.model.decoder.layers[0].self_attn
        v_proj = attn.v_proj
        out_proj = attn.out_proj
        device = v_proj.weight.device
        dtype = v_proj.weight.dtype

        if use_zero_biases:
            v_proj.load_state_dict({
                'weight': v_proj.weight,
                'bias': torch.zeros_like(v_proj.bias)
            })
            out_proj.load_state_dict({
                'weight': out_proj.weight,
                'bias': torch.zeros_like(out_proj.bias)
            })

        vector = torch.randn((opt.d_model), dtype=dtype, device=device)
        output = out_proj(v_proj(vector))

        print('before')
        print( vector[:3])
        print( output[:3])
        big = torch.matmul(out_proj.weight, v_proj.weight)
        out = torch.matmul(big, vector)
        print(out[:3])
        print(big[:5, :3])

        inv_out = mlp_svd_two_layer(v_proj, out_proj, opt.d_head,
            svd_dtype=torch.float32, combine_biases=True)

        new_output = out_proj(v_proj(vector))

        print('after')
        print( vector[:3])
        print( new_output[:3] )
        print(torch.matmul(out_proj.weight, v_proj.weight)[:5, :3])

svd_model_attention()