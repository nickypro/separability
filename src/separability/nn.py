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
    def __init__(self, original: torch.nn.Linear):
        super(InverseLinear, self).__init__()
        weights, bias = original.parameters()

        # Check that the original transformation is square
        original_size = weights.size()
        if original_size[0] != original_size[1]:
            raise ValueError("Original Linear Layer must be square")

        # Define the Inverse Bias vector
        self.inverse_bias = -bias

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

def mlp_delete_rows(mlp: torch.nn.Linear, deletion_indices: Tensor):
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
    mean_activations *= deletion_indices.to(device)
    bias_adjustement = mlp( mean_activations )
    biases += bias_adjustement

    # Place biases back into the MLP, with adjustements from above
    params.update({'bias': biases})
    mlp.load_state_dict(params)

    return mlp

def mlp_delete_columns(mlp: torch.nn.Linear, deletion_indices: Tensor):
    """Deletes (in place) the columns of weights in the MLP that are
    marked as True in deletion_indices

    Args:
        mlp (torch.nn.Linear): The Multi-Layer Perceptron to delete columns from
        deletion_indices (Tensor): The indices of the columns to delete
    """
    # Load parameters
    params = mlp.state_dict()
    weights: Tensor = params['weight']
    device = weights.device

    # Transpose the weights, then delete row by row
    weights_t = weights.transpose(0, 1)

    for row_index, delete_row in enumerate(deletion_indices):
        if delete_row:
            weights_t[row_index] = torch.zeros_like(weights_t[row_index])

    weights = weights_t.transpose(0, 1)

    # Update the model to have the deleted columns
    params.update({'weight': weights})
    mlp.load_state_dict(params)

    return mlp