import argparse

import torch
import argparse
import lightning as L

from networks.model import IceModel
from networks.dataset import OccupancyDataModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test IceNet model')
    
    parser.add_argument('--data_dir', type=str, default='./data/trials', help='Directory containing the dataset')
    parser.add_argument('-b', '--batchsize', type=int, default=128)
    parser.add_argument('-c', '--chunksize', type=int, default=25, help='Chunk size for data loading')
    parser.add_argument('-nc', '--num_chunks', type=int, default=1, help='Number of chunks to load per concentration')
    
    parser.add_argument("--conc", nargs="+", type=int, choices=[20, 30, 40, 50], required=True, help="Concentrations, values must be from [20, 30, 40, 50].")
    parser.add_argument("--test_trial_range", type=int, nargs=2, default=(9500, 10000), help="Range of test trials for each concentration [start, end).")
    parser.add_argument('--cpus', type=int, default=1, help='Number of CPU cores to use as workers')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')

    parser.add_argument('--in_channels', type=int, default=6, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=4, help='Number of output channels')
    parser.add_argument('--hidden_dims_cls', nargs='*', type=int, default=[], help='Hidden dimensions for classification head')
    parser.add_argument('--hidden_dims_reg', nargs='*', type=int, default=[], help='Hidden dimensions for regression head')
    
    parser.add_argument('--stage', type=str, choices=['map', 'cls', 'reg'], default=None, required=True, help='Stage to test')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to a checkpoint file for testing')

    parser.add_argument('--jobid', type=str, default=None, help='Job ID for tracking logs')

    parser.add_argument('--cost_log', action='store_true', help='Use log of cost in the model')

    args = parser.parse_args()
    
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint['hyper_parameters']
    
    model = IceModel.load_from_checkpoint(args.ckpt_path, args=argparse.Namespace(**hparams))
    
    data_module = OccupancyDataModule(
                        data_dir=args.data_dir,
                        concentrations=args.conc,
                        train_trial_range=None,
                        val_trial_range=None,
                        test_trial_range=args.test_trial_range,
                        chunk_size=args.chunksize,
                        num_chunks=args.num_chunks,
                        batch_size=args.batchsize,
                        num_workers=args.cpus,
                        stage=args.stage,
                        cost_log=args.cost_log
                        )
    
    trainer = L.Trainer(
        accelerator= 'gpu' if torch.cuda.is_available() else 'cpu',
        devices = args.gpus,
        strategy = 'ddp',
        logger=False,
        enable_progress_bar=False,
    )
    
    scores = trainer.test(model, datamodule=data_module, verbose=True)
    print(scores)
    
    
    
    