"""
Train a VarNet model on one of the Stanford MRI datasets with MRAugment data augmentation. 

The steps to add MRAugment to any training code is simple:
    1) Initialize a DataAugmentor with desired augmentation parameters and probabilities
    2) Pass augmentor to DataTransform that is applied to the training data
    3) You are all set!

Code based on https://github.com/facebookresearch/fastMRI/fastmri_examples/varnet/train_varnet_demo.py
"""

import os, sys
import pathlib
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute())   )

import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.pl_modules import VarNetModule

# MRAugment-specific imports
from mraugment.data_augment import DataAugmentor
from mraugment.data_transforms import VarNetDataTransform
from pl_modules.stanford_data_module import StanfordDataModule

# Imports for logging and other utility
from pytorch_lightning.plugins import DDPPlugin
import yaml
from utils import load_args_from_config
import torch.distributed


def cli_main(args):
    if args.verbose:
        print(args.__dict__)
        
    pl.seed_everything(args.seed)

    # ------------
    # model
    # ------------
    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )
    
    # -----------------
    # data augmentation
    # -----------------
    
    # pass an external function to DataAugmentor that 
    # returns the current epoch for p scheduling
    current_epoch_fn = lambda: model.current_epoch
    
    # initialize data augmentation pipeline
    augmentor = DataAugmentor(args, current_epoch_fn)
    
    # ------------
    # data
    # ------------
    
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    
    # use random masks for train transform, fixed masks for val transform
    # pass data augmentor to train transform only
    train_transform = VarNetDataTransform(augmentor=augmentor, mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform()
    
    # ptl data module - this handles data loaders
    data_module = StanfordDataModule(
        data_path=args.data_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
        train_val_seed=args.train_val_seed,
        train_val_split=args.train_val_split,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, 
                                            plugins=DDPPlugin(find_unused_parameters=False),
                                            checkpoint_callback=True,
                                            callbacks=args.checkpoint_callback)
    
    # Save all hyperparameters to .yaml file in the current log dir
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                save_all_hparams(trainer, args)
    else: 
         save_all_hparams(trainer, args)
            
    # ------------
    # run
    # ------------
    trainer.fit(model, datamodule=data_module)


def save_all_hparams(trainer, args):
    if not os.path.exists(trainer.logger.log_dir):
        os.makedirs(trainer.logger.log_dir)
    save_dict = args.__dict__
    save_dict.pop('checkpoint_callback')
    with open(trainer.logger.log_dir + '/hparams.yaml', 'w') as f:
        yaml.dump(save_dict, f)
    
def build_args():
    parser = ArgumentParser()

    # basic args
    backend = "ddp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1

    # client arguments
    parser.add_argument(
        '--config_file', 
        default=None,   
        type=pathlib.Path,          
        help='If given, experiment configuration will be loaded from this yaml file.',
    )
    parser.add_argument(
        '--verbose', 
        default=False,   
        action='store_true',          
        help='If set, print all command line arguments at startup.',
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = StanfordDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        mask_type="random",  # random masks for knee data
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )
    
    # data augmentation config
    parser = DataAugmentor.add_augmentation_specific_args(parser)

    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=12,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=0.0003,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
    )

    args = parser.parse_args()
    
    # Load args if config file is given
    if args.config_file is not None:
        args = load_args_from_config(args)
        

    args.checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_metrics/ssim",
        mode="max",
        filename='epoch{epoch}-ssim{val_metrics/ssim:.4f}',
        auto_insert_metric_name=False,
        save_last=True
    )

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()