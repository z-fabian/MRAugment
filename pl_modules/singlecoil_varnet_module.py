"""
Handles training simplified VarNet models on single-coil data. The sensitivity map estimator network is removed.
"""
from models.singlecoil_varnet import SinglecoilVarNet
from fastmri.pl_modules import VarNetModule
import fastmri

class SinglecoilVarNetModule(VarNetModule):
    """
    Single-coil version of the VarNet training module.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
        """
        super(VarNetModule, self).__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.varnet = SinglecoilVarNet(
            num_cascades=self.num_cascades,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = fastmri.SSIMLoss()