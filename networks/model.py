import torch
import lightning as L
from networks.architectures import architectures
from networks.network_utils import get_loss_fn, freeze_module, unfreeze_module
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import RelativeSquaredError, SpearmanCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC


class IceModel(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args.__dict__)
        
        if args.arch == 'unet_occ':
            args.only_occ_mode = True 
            self.net = architectures[args.arch](in_channels=args.in_channels, out_channels=args.out_channels)
        else:
            if args.stage == "map":
                use_reg_head = False
                use_cls_head = False
            elif args.stage == "cls":
                use_reg_head = False
                use_cls_head = True
            elif args.stage == "reg":
                use_reg_head = True
                use_cls_head = True

            self.net = architectures[args.arch](in_channels=args.in_channels, out_channels=args.out_channels,
                                                hidden_dims_reg=args.hidden_dims_reg, hidden_dims_cls=args.hidden_dims_cls,
                                                use_cls_head=use_cls_head, use_reg_head=use_reg_head)
        if hasattr(args, 'stage'):
            self.setup_stage()
        self.setup_loss_fn(args)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.rse = RelativeSquaredError()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()
        self.auroc = BinaryAUROC()
        self.spearman = SpearmanCorrCoef()
        
    def setup_loss_fn(self, args):
        self.main_loss_fn = get_loss_fn(args.main_loss_fn, **args.main_loss_args)
        
        self.cons_loss_fn = get_loss_fn(args.cons_loss_fn, **args.cons_loss_args) if args.cons_loss_fn is not None else None
        
        self.cls_loss_fn = get_loss_fn(args.cls_loss_fn, **args.cls_loss_args) if args.cls_loss_fn is not None else None
        
        self.reg_loss_fn = get_loss_fn(args.reg_loss_fn, **args.reg_loss_args) if args.reg_loss_fn is not None else None

        self.cons_scale = args.cons_scale
        
    def setup_stage(self):
        if self.hparams.stage=='map':
            if hasattr(self.net, 'classification_head'):
                freeze_module(self.net.classification_head)
            if hasattr(self.net, 'regression_head'):
                freeze_module(self.net.regression_head)
        elif self.hparams.stage=='cls':
            freeze_module(self.net)
            unfreeze_module(self.net.classification_head)
        elif self.hparams.stage=='reg':
            freeze_module(self.net)
            unfreeze_module(self.net.regression_head)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch, eta_min=1e-6)
        return [optimizer], [scheduler]
    
    def calc_loss_only_occ(self, y_hat_occ, y_occ, prefix):
        main_loss = self.main_loss_fn(y_hat_occ, y_occ)
        cons_loss = self.cons_loss_fn(y_hat_occ, y_occ) if self.cons_loss_fn is not None else torch.tensor(0.0, device=self.device)
        loss = main_loss + self.cons_scale * cons_loss
        self.log(f'{prefix}_main_loss', main_loss, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_cons_loss', cons_loss, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_map_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def calc_loss(self, y_hat_occ, y_hat_cls, y_hat_reg, y_occ, y_cost, prefix):
        if self.hparams.stage == 'map':
            main_loss = self.main_loss_fn(y_hat_occ, y_occ)
            # conservation loss is for occupancy and thickness channels only
            cons_loss = self.cons_loss_fn(y_hat_occ[:, :2], y_occ[:, :2]) if self.cons_loss_fn is not None else torch.tensor(0.0, device=self.device)
            loss = main_loss + self.cons_scale * cons_loss 
            self.log(f'{prefix}_main_loss', main_loss, on_epoch=True, sync_dist=True)
            self.log(f'{prefix}_cons_loss', cons_loss, on_epoch=True, sync_dist=True)
            self.log(f'{prefix}_map_loss', loss, on_epoch=True, sync_dist=True)
        elif self.hparams.stage == 'cls':
            loss = self.cls_loss_fn(y_hat_cls, y_cost)
            self.log(f'{prefix}_cls_loss', loss, on_epoch=True, sync_dist=True)
        elif self.hparams.stage == 'reg':
            loss = self.reg_loss_fn(y_hat_reg, y_cost)
            self.log(f'{prefix}_reg_loss', loss, on_epoch=True, sync_dist=True)
            if self.hparams.cost_log:
                cost_mse = torch.nn.functional.mse_loss(y_hat_reg.exp(), y_cost.exp())
                self.log(f'{prefix}_cost_mse_exp', cost_mse, on_epoch=True, sync_dist=True)
            
        return loss
    
    def _shared_step(self, batch, prefix):
        x, (y_occ, y_cost) = batch
        
        if self.hparams.only_occ_mode:
            y_occ = y_occ[:, [0]]  # only occupancy channel
            x = x[:, [0, 4, 5]]
            y_hat_occ = self(x)
            loss = self.calc_loss_only_occ(y_hat_occ, y_occ, prefix)
        else:
            y_hat_occ, y_hat_cls, y_hat_reg = self(x)
            loss = self.calc_loss(y_hat_occ, y_hat_cls, y_hat_reg, y_occ, y_cost, prefix)
            
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        x, (y_occ, y_cost) = batch
        if self.hparams.only_occ_mode:
            y_occ = y_occ[:, [0]]
            x = x[:, [0, 4, 5]]
            y_hat_occ = self(x)
    
            occ_mse = torch.nn.functional.mse_loss(y_hat_occ, y_occ)
            occ_ssim = self.ssim(y_hat_occ, y_occ)
            occ_diff = torch.nn.functional.mse_loss(x[:, 0], y_occ[:, 0], reduction='none').mean(dim=(1, 2))
            diff_corr = torch.corrcoef(torch.stack([occ_diff, y_occ[:, 0]]))[0, 1]
            self.log('test_occ_mse', occ_mse, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test_occ_ssim', occ_ssim, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test_diff_corr', diff_corr, on_step=False, on_epoch=True, sync_dist=True)
            return {
                'occ_mse': occ_mse,
                'occ_ssim': occ_ssim,
                'diff_corr': diff_corr,
            }
        else:
            y_hat_occ, y_hat_cls, y_hat_reg = self(x)
            
            if self.hparams.stage == 'map':
                occ_mse = torch.nn.functional.mse_loss(y_hat_occ, y_occ)
                occ_ssim = self.ssim(y_hat_occ, y_occ)
                self.log('test_occ_mse', occ_mse, on_step=False, on_epoch=True, sync_dist=True)
                self.log('test_occ_ssim', occ_ssim, on_step=False, on_epoch=True, sync_dist=True)
                return {
                    'map_mse': occ_mse,
                    'map_ssim': occ_ssim,
                }
                
            elif self.hparams.stage == 'cls':
                y_hat_cls_prob = torch.sigmoid(y_hat_cls)
                y_cost = y_cost.int()
                cls_acc = self.accuracy(y_hat_cls_prob, y_cost)
                cls_prec = self.precision(y_hat_cls_prob, y_cost)
                cls_rec = self.recall(y_hat_cls_prob, y_cost)
                cls_f1 = self.f1(y_hat_cls_prob, y_cost)
                cls_auroc = self.auroc(y_hat_cls_prob, y_cost)
                self.log('test_cls_acc', cls_acc, on_step=False, on_epoch=True, sync_dist=True)
                self.log('test_cls_prec', cls_prec, on_step=False, on_epoch=True, sync_dist=True)
                self.log('test_cls_rec', cls_rec, on_step=False, on_epoch=True, sync_dist=True)
                self.log('test_cls_f1', cls_f1, on_step=False, on_epoch=True, sync_dist=True)
                self.log('test_cls_auroc', cls_auroc, on_step=False, on_epoch=True, sync_dist=True)
                return {
                    'cls_acc': cls_acc,
                    'cls_prec': cls_prec,
                    'cls_rec': cls_rec,
                    'cls_f1': cls_f1,
                    'cls_auroc': cls_auroc,
                }
                
            elif self.hparams.stage == 'reg':
                if self.hparams.cost_log:
                    cost_mse_exp = torch.nn.functional.mse_loss(y_hat_reg.exp(), y_cost.exp())
                    self.log('test_reg_mse_exp', cost_mse_exp, on_epoch=True, sync_dist=True)
                else:
                    y_hat_reg = torch.clamp(y_hat_reg, min=0.0)
                cost_mse = torch.nn.functional.mse_loss(y_hat_reg, y_cost)
                cost_rse = self.rse(y_hat_reg, y_cost)
                cost_pearson = torch.corrcoef(torch.stack([y_hat_reg[:, 0], y_cost[:, 0]]))[0, 1]
                cost_spearman = self.spearman(y_hat_reg[:, 0], y_cost[:, 0])
                
                occ_diff = torch.nn.functional.mse_loss(x[:, 0], y_occ[:, 0], reduction='none').mean(dim=(1, 2))
                diff_pearson = torch.corrcoef(torch.stack([occ_diff, y_cost[:, 0]]))[0, 1]
                diff_spearman = self.spearman(occ_diff, y_cost[:, 0])
                
                self.log('test_reg_mse', cost_mse, on_step=False, on_epoch=True, sync_dist=True)
                self.log('test_reg_rse', cost_rse, on_step=False, on_epoch=True, sync_dist=True)
                self.log('test_reg_pearson', cost_pearson, on_step=False, on_epoch=True, sync_dist=True)
                self.log('test_reg_spearman', cost_spearman, on_step=False, on_epoch=True, sync_dist=True)
                self.log('test_diff_pearson', diff_pearson, on_step=False, on_epoch=True, sync_dist=True)
                self.log('test_diff_spearman', diff_spearman, on_step=False, on_epoch=True, sync_dist=True)
            
                return {
                    'reg_mse': cost_mse,
                    'reg_rse': cost_rse,
                    'reg_pearson': cost_pearson,
                    'reg_spearman': cost_spearman,
                    'diff_pearson': diff_pearson,
                    'diff_spearman': diff_spearman,
                    'reg_mse_exp': cost_mse_exp if self.hparams.cost_log else None,
                }
        