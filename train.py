import json
import torch
import argparse
import lightning as L
import yaml
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from networks.architectures import architectures
from networks.network_utils import loss_fns
from networks.dataset import OccupancyDataModule
from networks.model import IceModel

def args_to_str(args):
    parts = [
        f"{args.arch}",
        f"s({args.seed})",
        f"e({args.epoch})",
        f"b({args.batchsize})",
        f"c({args.chunksize},{args.num_chunks})",
        f"lr({args.lr})",
        f"wd({args.weight_decay})",
        f"conc({','.join(map(str, args.conc))})",
        f"tr({args.train_trial_range[0]}-{args.train_trial_range[1]})",
        f"vr({args.val_trial_range[0]}-{args.val_trial_range[1]})",
        f"w({args.window_height}x{args.window_width})",
        f"loss({args.main_loss_fn},{args.cons_loss_fn or 'none'},{args.cls_loss_fn or 'none'},{args.reg_loss_fn or 'none'})",
        f"loss_args({args.main_loss_args},{args.cons_loss_args},{args.cls_loss_args},{args.reg_loss_args})",
        f"scale({args.cons_scale})",
        f"hdims_cls({','.join(map(str, args.hidden_dims_cls))})",
        f"hdims_reg({','.join(map(str, args.hidden_dims_reg))})",
    ]
    if args.debug:
        parts.insert(0, "debug")
    if args.only_occ_mode:
        parts.append("only_occ")
    if args.cost_log:
        parts.append("cost_log")
    return "_".join(parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train IceNet model')

    parser.add_argument('--data_dir', type=str, default='./data/trials', help='Directory containing the dataset')
    parser.add_argument('--arch', choices=architectures.keys(), default='unet')
    parser.add_argument('-b', '--batchsize', type=int, default=128)
    parser.add_argument('-c', '--chunksize', type=int, default=25, help='Chunk size for data loading')
    parser.add_argument('-nc', '--num_chunks', type=int, default=1, help='Number of chunks to load per concentration')
    
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('-wd', '--weight-decay', type=float, default=0)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('--valfreq', type=int, default=10, help='Validation frequency over epochs')
    
    parser.add_argument('--cpus', type=int, default=1, help='Number of CPU cores to use as workers')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes to use for distributed training')
    
    parser.add_argument("--conc", nargs="+", type=int, choices=[20, 30, 40, 50], required=True, help="Concentrations, values must be from [20, 30, 40, 50].")

    parser.add_argument("--train_trial_range", type=int, nargs=2, default=(0, 100), help="Range of training trials for each concentration [start, end).")
    parser.add_argument("--val_trial_range", type=int, nargs=2, default=(0, 100), help="Range of validation trials for each concentration [start, end).")
    parser.add_argument("--test_trial_range", type=int, nargs=2, default=(0, 100), help="Range of test trials for each concentration [start, end).")
    
    parser.add_argument('-wh', '--window_height', type=int, default=80, help='Height of the observation window')
    parser.add_argument('-ww', '--window_width', type=int, default=80, help='Width of the observation window')
    
    parser.add_argument('--main_loss_fn', type=str, choices=loss_fns.keys(), default=' ')
    parser.add_argument('--main_loss_args', type=json.loads, default='{}')
    parser.add_argument('--cons_loss_fn', type=str, choices=loss_fns.keys(), default=None)
    parser.add_argument('--cons_loss_args', type=json.loads, default='{}')
    parser.add_argument('--cls_loss_fn', type=str, choices=loss_fns.keys(), default=None)
    parser.add_argument('--cls_loss_args', type=json.loads, default='{}')
    parser.add_argument('--reg_loss_fn', type=str, choices=loss_fns.keys(), default=None)
    parser.add_argument('--reg_loss_args', type=json.loads, default='{}')
    
    parser.add_argument('--cost_log', action='store_true', help='Whether to use log cost values during training')
    parser.add_argument('--cons_scale', type=float, default=0, help='Scale for conservation loss')
    parser.add_argument('--only_occ_mode', action='store_true', help='Only use occupancy channel (no thickness, speed or cost)')
        
    parser.add_argument('--in_channels', type=int, default=6, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=4, help='Number of output channels')
    parser.add_argument('--hidden_dims_cls', nargs='*', type=int, default=[], help='Hidden dimensions for classification head')
    parser.add_argument('--hidden_dims_reg', nargs='*', type=int, default=[], help='Hidden dimensions for regression head')
    
    parser.add_argument('--stage', type=str, choices=['map', 'cls', 'reg'], default=None, required=True, help='Stage to train')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Checkpoint path to resume from')
    
    parser.add_argument('-pb', '--progress_bar', action='store_true', help='Enable progress bar')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--jobid', type=str, default=None, help='Job ID for tracking logs')
    parser.add_argument('--desc', type=str, default=None, help='Additional Description for the run')
    
    args = parser.parse_args()
    
    
    L.pytorch.seed_everything(args.seed, workers=True)
    
    wandb_config = yaml.safe_load(open('configs/wandb.yaml', 'r'))
    wandb.login(key=wandb_config['api_key'])

    print('args:', vars(args))
    print('name:', args_to_str(args))
    if torch.cuda.is_available():
        print(f'device cuda {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}, bf16 {torch.cuda.is_bf16_supported()}')
    else:
        print('device cpu')

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",       
        mode="min",               
        save_top_k=1,              
        save_last=True,           
    )
    
    stage_trainer_args = {
        'max_epochs': args.epoch,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': args.gpus,
        'num_nodes': args.nodes,
        'strategy': 'ddp',
        'sync_batchnorm': True,
        'log_every_n_steps': 50,
        'enable_progress_bar': args.progress_bar,
        'check_val_every_n_epoch': args.valfreq,
        'num_sanity_val_steps': 0,
        'precision': 'bf16-mixed' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else '16-mixed',
        'benchmark': True,
        'deterministic': False,
    }
    
    print(f"\n{'='*30}\nStarting stage: {args.stage}\n{'='*30}")
            
    data_module = OccupancyDataModule(
                        data_dir=args.data_dir,
                        concentrations=args.conc,
                        train_trial_range=args.train_trial_range,
                        val_trial_range=args.val_trial_range,
                        test_trial_range=args.test_trial_range,
                        chunk_size=args.chunksize,
                        num_chunks=args.num_chunks,
                        batch_size=args.batchsize,
                        num_workers=args.cpus,
                        stage=args.stage,
                        cost_log=args.cost_log
                        )
    
    stage_name = f"{args_to_str(args)}_{args.stage}"
    stage_wandb_logger = L.pytorch.loggers.WandbLogger(project=wandb_config['project'], config=vars(args), name=stage_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor = f"val_{args.stage}_loss",
        mode="min",
        save_top_k=1,
        dirpath=f"checkpoints{'/debug' if args.debug else ''}/{stage_wandb_logger.experiment.id}",
        filename=f"{args.stage}-{{epoch}}",
    )
    
    stage_trainer_args['logger'] = stage_wandb_logger
    stage_trainer_args['callbacks'] = [
        LearningRateMonitor(logging_interval='epoch'),
        checkpoint_callback,
    ]
    
    model = IceModel(args=args)
    if args.resume_ckpt:
        print(f"Loading checkpoint: {args.resume_ckpt}")
        checkpoint = torch.load(args.resume_ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        
    print(f"Starting training: {args.stage}\n{'='*30}")
    trainer = L.Trainer(**stage_trainer_args)
    trainer.fit(model, datamodule=data_module)
    print(f"Completed stage: {args.stage}\n{'='*30}")
    
    print(f"Best checkpoint for stage {args.stage}: {checkpoint_callback.best_model_path}")
    
    test_results = trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
    print(f"Test results for stage {args.stage} : {test_results}")
    
    wandb.finish()
