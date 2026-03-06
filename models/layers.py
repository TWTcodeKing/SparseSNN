"""
Utility layers for SNN models.
Replaces spikingjelly.clock_driven.layer utilities.
"""

import torch
import torch.nn as nn



# Spatial Container for sequential data
class SeqToANNContainer(nn.Module):
    """
    Wraps standard ANN layers to process sequential (T, B, ...) input.
    Flattens T and B, applies the wrapped modules, then reshapes back.
    Replaces spikingjelly.clock_driven.layer.SeqToANNContainer.
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq):
        """x_seq: (T, B, C, H, W)"""
        T = x_seq.shape[0]
        y = self.module(x_seq.flatten(0, 1))
        y_shape = y.shape
        return y.view(T, -1, *y_shape[1:])
    
# temporal container for sequential data
class SeqToANNContainerT(nn.Module):
    """
    Wraps standard ANN layers to process sequential (T, B, ...) input.
    Flattens T and B, applies the wrapped modules, then reshapes back.
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq):
        """x_seq: (T, B, C, H, W)"""
        T = x_seq.shape[0]
        y = []
        for t in range(T):
            y.append(self.module(x_seq[t]))
        y = torch.stack(y)
        return y