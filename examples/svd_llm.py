import torch
from separability import Model
from separability.nn import mlp_svd_two_layer, InverseLinear

def svd_model_attention():
    with torch.no_grad():
        opt = Model("facebook/opt-125m", limit=1000, use_accelerator=False)
        attn = opt.model.decoder.layers[0].self_attn
        v_proj = attn.v_proj
        out_proj = attn.out_proj
        device = v_proj.weight.device
        """
        v_proj.load_state_dict({
            'weight': v_proj.weight,
            'bias': torch.zeros_like(v_proj.bias)
        })
        out_proj.load_state_dict({
            'weight': out_proj.weight,
            'bias': torch.zeros_like(out_proj.bias)
        })
        #"""

        vector = torch.randn((opt.d_model)).to(device)
        output = out_proj(v_proj(vector))

        print('before')
        print( vector[:3])
        print( output[:3])
        big = torch.matmul(out_proj.weight, v_proj.weight)
        out = torch.matmul(big, vector)
        print(out[:3])
        print(big[:5, :3])

        mlp_svd_two_layer(v_proj, out_proj, opt.d_head)

        new_output = out_proj(v_proj(vector))

        print('after')
        print( vector[:3])
        print( new_output[:3] )
        print(torch.matmul(out_proj.weight, v_proj.weight)[:5, :3])

svd_model_attention()