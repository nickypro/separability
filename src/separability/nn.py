""" This file contains utils for working with single layers of dense neural networks.
"""

from typing import Optional, Union
from torch import Tensor
import torch

class InverseLinear(torch.nn.Module):
    """_summary_
    Produces a torch layer which undoes what the "original" linear layer does.

    This was built to get the output right before out_proj, and I thought
    it would be easier and/or faster than using the values matrix
    (it probably is not actually easier or faster)

    Args:
        original: The original Linear layer we are getting the inverse of
    """
    def __init__(self,
            original: torch.nn.Linear=None,
            original_weights: Tensor=None,
            original_biases: Tensor=None,
        ):
        super(InverseLinear, self).__init__()
        if original is not None:
            weights, bias = original.parameters()
        elif original_weights is not None:
            weights = original_weights
            bias    = original_biases
        else:
            raise ValueError("Either original or original_weights must be provided")

        # Check that the original transformation is square
        original_size = weights.size()
        if original_size[0] != original_size[1]:
            raise ValueError("Original Linear Layer must be square")

        # Define the Inverse Bias vector
        if bias is None:
            bias = torch.zeros(weights.shape[0], device=weights.device)
        self.inverse_bias = - bias.clone()

        # Define the Inverse Linear Layer
        _dtype, _device = weights.dtype, weights.device
        inverse_weights = weights.cpu().to(dtype=torch.float64).inverse()
        inverse_weights = inverse_weights.to(dtype=_dtype).to(_device)
        size = inverse_weights.size()
        self.fc = torch.nn.Linear( size[0], size[1], bias=False )
        self.fc.load_state_dict({'weight': inverse_weights})

    def forward(self, x):
        y = x + self.inverse_bias
        y = self.fc( y )
        return y

    # pylint: disable=arguments-differ
    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           **kwargs
        ):
        super( InverseLinear, self ).to( device, dtype=dtype, **kwargs )
        if device is not None:
            self.inverse_bias = self.inverse_bias.to( device, **kwargs )
            self.fc = self.fc.to( device, **kwargs )
        if dtype is not None:
            self.inverse_bias = self.inverse_bias.to( dtype=dtype, **kwargs )
            self.fc = self.fc.to( dtype=dtype, **kwargs )
        return self

def mlp_delete_rows(mlp:
        torch.nn.Linear,
        deletion_indices: Tensor,
        delete_biases:bool = True,
    ):
    """Deletes (in place) the weights and biases of rows of the MLP that
    are marked True in deletion_rows

    Args:
        mlp (torch.nn.Linear): The Multi-Layer Perceptron to delete rows from
        deletion_rows (Tensor): Tensor of booleans indicating which rows to delete
    """
    # Get the parameters from the MLP
    params = mlp.state_dict()
    weights: Tensor = params['weight']
    biases: Tensor  = params['bias']

    # Delete the weights and biases from the rows
    n_rows = len(weights)
    for row_index in range(n_rows):
        if deletion_indices[row_index]:
            weights[row_index] = torch.zeros_like(weights[row_index])
            if delete_biases:
                biases[row_index]  = torch.zeros_like(biases[row_index])

    # Update the model to have the deleted rows
    params.update({'weight': weights, 'bias': biases})
    mlp.load_state_dict(params)

    return mlp

def mlp_adjust_biases(
        mlp: torch.nn.Linear,
        deletion_indices: Tensor,
        mean_activations: Tensor,
    ):
    """ Calculates the bias adjustement needed to compensate for the deletion of
    neurons in the MLP, and applies it to the MLP

    Args:
        mlp (torch.nn.Linear): The Multi-Layer Perceptron to adjust the biases of
        deletion_rows (Tensor): The
        mean_activations (Tensor):
    """
    if mean_activations is None:
        return mlp

    # Load parameters
    params = mlp.state_dict()
    biases: Tensor = params['bias']
    device = biases.device

    # Make a temporary copy of the MLP with only the weights (no biases).
    params.update({'bias': torch.zeros_like(biases)})
    mlp.load_state_dict(params)

    # adjust bias of out_proj by mean activations
    mean_activations  = mean_activations.detach().clone().to(device)
    mean_activations *= deletion_indices.to(device).reshape(mean_activations.shape)
    bias_adjustement = mlp( mean_activations )
    biases += bias_adjustement

    # Place biases back into the MLP, with adjustements from above
    params.update({'bias': biases})
    mlp.load_state_dict(params)

    return mlp

def mlp_delete_columns_raw(weights: Tensor, deletion_indices: Tensor):
    """Deletes (in place) the columns of weights in the MLP that are
    marked as True in deletion_indices

    Args:
        weights (Tensor): The MLP Weights to delete columns from
        deletion_indices (Tensor): The indices of the columns to delete
    """
    # Load parameters
    device = weights.device

    # Transpose the weights, then delete row by row
    weights_t = weights.transpose(0, 1)

    for row_index, delete_row in enumerate(deletion_indices):
        if delete_row:
            weights_t[row_index] = torch.zeros_like(weights_t[row_index])

    weights = weights_t.transpose(0, 1)

    return weights

def mlp_delete_columns(mlp: torch.nn.Linear, deletion_indices: Tensor):
    """Deletes (in place) the columns of weights in the MLP that are
    marked as True in deletion_indices

    Args:
        mlp (torch.nn.Linear): The Multi-Layer Perceptron to delete columns from
        deletion_indices (Tensor): The indices of the columns to delete
    """
    # Load parameters
    params = mlp.state_dict()

    # Delete the columns
    weights = mlp_delete_columns_raw(params['weight'], deletion_indices)

    # Update the model to have the deleted columns
    params.update({'weight': weights})
    mlp.load_state_dict(params)

    return mlp

def mlp_svd_two_layer_raw(
        W_in: Tensor,
        W_out: Tensor,
        b_in: Tensor,
        b_out: Tensor,
        d_head: float,
        svd_dtype: torch.dtype = torch.float32,
        combine_biases: bool = False,
    ):
    """Calculates the SVD of the two layers, and alters the weights of the
    layers to be sqrt(S)*U and sqrt(S)*V, and alters the biases of layer_1
    so that it always gives the same output. Only works if there is no
    activation function between the two layers.

    Args:
        layer_1 (torch.nn.Linear): The first layer of the MLP
        layer_2 (torch.nn.Linear): The second layer of the MLP
        d_head (float): The dimension of the head
        svd_dtype (torch.dtype, optional): The dtype to use for the SVD.
            Defaults to torch.float32.

    Returns:
        inv_out (InverseLinear): The inverse of the new layer_2
    """
    # Get the parameters from the MLP
    orig_shape = W_in.shape
    assert orig_shape[0] % d_head == 0, "d_head must divide d_model"

    n_heads = orig_shape[0] // d_head
    d_model = orig_shape[1]

    # pre-compute the effect of the layer 1 biases, so that we can reconstruct
    # the bias again in the new basis later
    b_in_effect = torch.matmul(W_out, b_in)

    # Get layer 1 and layer 2 weights in correct shapes for SVD
    W_in_heads  = W_in.reshape((n_heads, d_head, d_model))
    W_out_heads = W_out.reshape((d_model, n_heads, d_head))

    dtype, device = W_in.dtype, W_in.device

    # perform SVD using only the weights
    for head in range(n_heads):
        # Get the weights
        in_weights  = W_in_heads[head]
        out_weights = W_out_heads[:, head]

        # Perform SVD
        big_matrix = torch.matmul(out_weights, in_weights).to(dtype=svd_dtype)
        u_out, s, v_in = torch.linalg.svd(big_matrix, full_matrices=True)

        # Remove rows/columns which we know should have zero rank
        s = s[:d_head].to(dtype=dtype)
        v_in  = v_in[:d_head, :].to(dtype=dtype)  # in_matrix,  eg [64, 768]
        u_out = u_out[:, :d_head].to(dtype=dtype) # out_matrix, eg [768, 64]

        # Scale the v_in and u_out matrices by sqrt(S) each
        s = s.sqrt()
        u_out *= s
        v_in  *= s.unsqueeze(dim=-1)

        # return new weights to original head matrix
        in_weights  = v_in
        out_weights = u_out

        # move to actual original matrices for load_state_dict later
        W_in_heads[head] = in_weights
        W_out_heads[:, head] = out_weights

    # Get new weights in original format
    W_in  = W_in_heads.reshape(orig_shape)
    W_out = W_out_heads.reshape(orig_shape)

    # Get new biases in original format
    # Use inverse linear to reconstruct new biases for layer_1
    inv_out = InverseLinear(
        original_weights=W_out, original_biases=b_out
        ).to(dtype=dtype).to(device)

    if combine_biases:
        # Combine the biases of layer_1 into layer_2
        b_out = b_out + b_in_effect
        b_in = torch.zeros_like(b_in)

    else:
        # otherwise, reconstruct the biases for layer_1, keep layer_2 unchanged
        b_in = inv_out.fc(b_in_effect)

    updated_weights = {
        "W_in": W_in, "W_out": W_out, "b_in": b_in, "b_out": b_out,
    }

    return inv_out, updated_weights


def mlp_svd_two_layer(
        layer_1: torch.nn.Linear,
        layer_2: torch.nn.Linear,
        d_head: float,
        svd_dtype: torch.dtype = torch.float32,
        combine_biases: bool = False,
    ):
    """Calculates the SVD of the two layers, and alters the weights of the
    layers to be sqrt(S)*U and sqrt(S)*V, and alters the biases of layer_1
    so that it always gives the same output. Only works if there is no
    activation function between the two layers.

    Args:
        layer_1 (torch.nn.Linear): The first layer of the MLP
        layer_2 (torch.nn.Linear): The second layer of the MLP
        d_head (float): The dimension of the head
        svd_dtype (torch.dtype, optional): The dtype to use for the SVD.
            Defaults to torch.float32.

    Returns:
        inv_out (InverseLinear): The inverse of the new layer_2
    """
    params_1 = layer_1.state_dict()
    params_2 = layer_2.state_dict()
    W_in,  B_in  = layer_1.weight, layer_1.bias
    W_out, B_out = layer_2.weight, layer_2.bias

    inv_out, updated_weights = mlp_svd_two_layer_raw(
        W_in, W_out, B_in, B_out, d_head, svd_dtype, combine_biases )

    params_1["weight"] = updated_weights["W_in"]
    params_1["bias"]   = updated_weights["b_in"]
    layer_1.load_state_dict(params_1)

    params_2["weight"] = updated_weights["W_out"]
    params_2["bias"]   = updated_weights["b_out"]
    layer_2.load_state_dict(params_2)

    return inv_out
