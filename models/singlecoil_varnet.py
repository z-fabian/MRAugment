"""
Code is modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/models/varnet.py
to support singlecoil models. The sensitivity map estimator network is removed and the slice is assumed
to have a single channel.
"""
from fastmri.models import VarNet, VarNetBlock,  NormUnet
from typing import List, Tuple
import fastmri
import torch
import torch.nn as nn

class SinglecoilVarNetBlock(VarNetBlock):
    """
    Model block for the single-coil version of the end-to-end variational network.
    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = fastmri.fft2c(self.model(fastmri.ifft2c(current_kspace)))

        return current_kspace - soft_dc - model_term
    
class SinglecoilVarNet(VarNet):
    """
    A single-coil version of the full variational network model.
    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super(VarNet, self).__init__()

        self.cascades = nn.ModuleList(
            [SinglecoilVarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)