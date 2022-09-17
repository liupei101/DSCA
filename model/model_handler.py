import os.path as osp
from types import SimpleNamespace
import torch
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .HierNet import WSIGenericNet, WSIHierNet, WSIGenericCAPNet
from .model_utils import init_weights
from utils import *
from loss import create_survloss, loss_reg_l1
from optim import create_optimizer
from dataset import prepare_dataset
from eval import evaluator


class MyHandler(object):
    """Deep Risk Predition Model Handler.
    Handler the model train/val/test for: HierSurv
    """
    def __init__(self, cfg):
        # set up for seed and device
        torch.cuda.set_device(cfg['cuda_id'])
        seed_everything(cfg['seed'])

        # set up for path
        self.writer = SummaryWriter(cfg['save_path'])
        self.last_ckpt_path = osp.join(cfg['save_path'], 'model-last.pth')
        self.best_ckpt_path = osp.join(cfg['save_path'], 'model-best.pth')
        self.metrics_path   = osp.join(cfg['save_path'], 'metrics.txt')
        self.config_path    = osp.join(cfg['save_path'], 'print_config.txt')

        # in_dim / hid1_dim / hid2_dim / out_dim
        dims = [int(_) for _ in cfg['dims'].split('-')] 

        # set up for model
        if cfg['task'] == 'GenericSurv':
            cfg['scale'] = 4 if 'x20' in cfg['magnification'] else 1
            cfg_emb_backbone = SimpleNamespace(in_dim=dims[0], out_dim=dims[1], scale=cfg['scale'], dropout=cfg['dropout'], dw_conv=cfg['emb_dw_conv'], ksize=cfg['emb_ksize'])
            cfg_tra_backbone = SimpleNamespace(d_model=dims[1], d_out=dims[2], nhead=cfg['tra_nhead'], dropout=cfg['dropout'], num_layers=cfg['tra_num_layers'],
                ksize=cfg['tra_ksize'], dw_conv=cfg['tra_dw_conv'], epsilon=cfg['tra_epsilon'])
            self.model = WSIGenericNet(
                dims, cfg['emb_backbone'], cfg_emb_backbone, cfg['tra_backbone'], cfg_tra_backbone, 
                dropout=cfg['dropout'], pool=cfg['pool']
            )
            self.model.apply(init_weights) # model parameter init
            print(self.model)
        elif cfg['task'] == 'GenericCAPSurv':
            cfg['magnification'] = 'x5_x20'
            cfg_emb_backbone = SimpleNamespace(in_dim=dims[0], out_dim=dims[1], scale=4, dropout=cfg['dropout'], dw_conv=cfg['emb_dw_conv'], ksize=cfg['emb_ksize'])
            cfg_tra_backbone = SimpleNamespace(d_model=dims[1], d_out=dims[2], nhead=cfg['tra_nhead'], dropout=cfg['dropout'], num_layers=cfg['tra_num_layers'],
                ksize=cfg['tra_ksize'], dw_conv=cfg['tra_dw_conv'], epsilon=cfg['tra_epsilon'])
            self.model = WSIGenericCAPNet(
                dims, cfg['emb_backbone'], cfg_emb_backbone, cfg['tra_backbone'], cfg_tra_backbone, 
                dropout=cfg['dropout'], pool=cfg['pool']
            )
            self.model.apply(init_weights) # model parameter init
        elif cfg['task'] == 'HierSurv':
            cfg_x20_emb = SimpleNamespace(backbone=cfg['emb_x20_backbone'], 
                in_dim=dims[0], out_dim=dims[1], scale=4, dropout=cfg['dropout'], dw_conv=cfg['emb_x20_dw_conv'], ksize=cfg['emb_x20_ksize'])
            cfg_x5_emb = SimpleNamespace(backbone=cfg['emb_x5_backbone'], 
                in_dim=dims[0], out_dim=dims[1], scale=1, dropout=cfg['dropout'], dw_conv=False, ksize=cfg['emb_x5_ksize'])
            cfg_tra_backbone = SimpleNamespace(backbone=cfg['tra_backbone'], ksize=cfg['tra_ksize'], dw_conv=cfg['tra_dw_conv'],
                d_model=dims[1], d_out=dims[2], nhead=cfg['tra_nhead'], dropout=cfg['dropout'], num_layers=cfg['tra_num_layers'], epsilon=cfg['tra_epsilon'])
            self.model = WSIHierNet(
                dims, cfg_x20_emb, cfg_x5_emb, cfg_tra_backbone, 
                dropout=cfg['dropout'], pool=cfg['pool'], join=cfg['join'], fusion=cfg['fusion']
            )
        else:
            raise ValueError(f"Expected HierSurv/GenericSurv, but got {cfg['task']}")
        self.model = self.model.cuda()
        print_network(self.model)
        self.model_pe = cfg['tra_position_emb']
        print("[model] Transformer Position Embedding: {}".format('Yes' if self.model_pe else 'No'))
        
        # set up for loss, optimizer, and lr scheduler
        self.loss = create_survloss(cfg['loss'], argv={'alpha': cfg['alpha']})
        self.loss_l1 = loss_reg_l1(cfg['reg_l1'])
        cfg_optimizer = SimpleNamespace(opt=cfg['opt'], weight_decay=cfg['weight_decay'], lr=cfg['lr'], 
            opt_eps=cfg['opt_eps'], opt_betas=cfg['opt_betas'], momentum=cfg['opt_momentum'])
        self.optimizer = create_optimizer(cfg_optimizer, self.model)
        
        # 1. Early stopping: patience = 30
        # 2. LR scheduler: lr * 0.5 if val_loss is not decreased in 10 epochs.
        if cfg['es_patience'] is not None:
            self.early_stop = EarlyStopping(warmup=cfg['es_warmup'], patience=cfg['es_patience'], start_epoch=cfg['es_start_epoch'], verbose=cfg['es_verbose'])
        else:
            self.early_stop = None
        self.steplr = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.cfg = cfg
        print_config(cfg, print_to_path=self.config_path)

    def exec(self):
        task = self.cfg['task']
        experiment = self.cfg['experiment']
        print('[exec] start experiment {} on {}.'.format(experiment, task))
        
        path_split = self.cfg['path_data_split'].format(self.cfg['seed_data_split'])
        pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
        print('[exec] read patient IDs from {}'.format(path_split))
        
        # For reporting results
        if experiment == 'sim':
            # Prepare datasets 
            train_set  = prepare_dataset(pids_train, self.cfg, self.cfg['magnification'])
            train_pids = train_set.pids
            val_set    = prepare_dataset(pids_val, self.cfg, self.cfg['magnification'])
            val_pids   = val_set.pids
            train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], generator=seed_generator(self.cfg['seed']),
                num_workers=self.cfg['num_workers'], shuffle=True,  worker_init_fn=seed_worker)
            val_loader   = DataLoader(val_set,   batch_size=self.cfg['batch_size'], generator=seed_generator(self.cfg['seed']),
                num_workers=self.cfg['num_workers'], shuffle=False, worker_init_fn=seed_worker)
            if pids_test is not None:
                test_set    = prepare_dataset(pids_test, self.cfg, self.cfg['magnification'])
                test_pids   = test_set.pids
                test_loader = DataLoader(test_set,  batch_size=self.cfg['batch_size'], generator=seed_generator(self.cfg['seed']),
                    num_workers=self.cfg['num_workers'], shuffle=False, worker_init_fn=seed_worker)
            else:
                test_set = None 
                test_pids = None
                test_loader = None

            # Train
            val_name = 'validation'
            val_loaders = {'validation': val_loader, 'test': test_loader}
            self._run_training(train_loader, val_loaders=val_loaders, val_name=val_name, measure=True, save=False)

            # Evals
            metrics = dict()
            evals_loader = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
            for k, loader in evals_loader.items():
                if loader is None:
                    continue
                cur_pids = [train_pids, val_pids, test_pids][['train', 'validation', 'test'].index(k)]
                # cltor is on cpu
                cltor = self.test_model(self.model, loader, self.cfg['task'], checkpoint=self.best_ckpt_path, model_pe=self.model_pe)
                ci, loss = evaluator(cltor['y'], cltor['y_hat'], metrics='cindex'), self.loss(cltor['y'], cltor['y_hat'])
                metrics[k] = [('cindex', ci), ('loss', loss)]

                if self.cfg['save_prediction']:
                    path_save_pred = osp.join(self.cfg['save_path'], 'surv_pred_{}.csv'.format(k))
                    save_prediction(cur_pids, cltor['y'], cltor['y_hat'], path_save_pred)

        print_metrics(metrics, print_to_path=self.metrics_path)

        return metrics

    def _run_training(self, train_loader, val_loaders=None, val_name=None, measure=True, save=True, **kws):
        """Traing model.

        Args:
            train_loader ('DataLoader'): DatasetLoader of training set.
            val_loaders (dict): A dict like {'val': loader1, 'test': loader2}, gives the datasets
                to evaluate at each epoch.
            val_name (string): The dataset used to perform early stopping and optimal model saving.
            measure (bool): If measure training set at each epoch.
        """
        epochs = self.cfg['epochs']
        assert self.cfg['bp_every_iters'] % self.cfg['batch_size'] == 0, "Batch size must be divided by bp_every_iters."
        if val_name is not None and self.early_stop is not None:
            assert val_name in val_loaders.keys(), "Not specify the dataloader to perform early stopping."
            print("[training] {} epochs, with early stopping on {}.".format(epochs, val_name))
        else:
            print("[training] {} epochs, without early stopping.".format(epochs))
        
        last_epoch = -1
        for epoch in range(epochs):
            last_epoch = epoch

            train_cltor, batch_avg_loss = self._train_each_epoch(train_loader)
            self.writer.add_scalar('loss/train_batch_avg_loss', batch_avg_loss, epoch+1)
            
            if measure:
                train_ci, train_loss = evaluator(train_cltor['y'], train_cltor['y_hat'], metrics='cindex'), self.loss(train_cltor['y'], train_cltor['y_hat'])
                steplr_monitor_loss = train_loss
                self.writer.add_scalar('loss/train_overall_loss', train_loss, epoch+1)
                self.writer.add_scalar('c_index/train_ci', train_ci, epoch+1)
                print('[training] training epoch {}, avg. batch loss: {:.8f}, loss: {:.8f}, c_index: {:.5f}'.format(epoch+1, batch_avg_loss, train_loss, train_ci))

            val_metrics = None
            if val_loaders is not None:
                for k in val_loaders.keys():
                    if val_loaders[k] is None:
                        continue
                    val_cltor = self.test_model(self.model, val_loaders[k], self.cfg['task'], model_pe=self.model_pe)
                    # If it is at eval mode, then set alpha in SurvMLE to 0
                    met_ci, met_loss = evaluator(val_cltor['y'], val_cltor['y_hat'], metrics='cindex'), self.loss(val_cltor['y'], val_cltor['y_hat'], cur_alpha=0.0)
                    self.writer.add_scalar('loss/%s_overall_loss'%k, met_loss, epoch+1)
                    self.writer.add_scalar('c_index/%s_ci'%k, met_ci, epoch+1)
                    print("[training] {} epoch {}, loss: {:.8f}, c_index: {:.5f}".format(k, epoch+1, met_loss, met_ci))

                    if k == val_name:
                        # monitor ci 
                        val_metrics = met_ci if self.cfg['monitor_metrics'] == 'ci' else met_loss
            
            if val_metrics is not None and self.early_stop is not None:
                self.early_stop(epoch, val_metrics, self.model, ckpt_name=self.best_ckpt_path)
                self.steplr.step(val_metrics)
                if self.early_stop.if_stop():
                    last_epoch = epoch + 1
                    break
            
            self.writer.flush()

        if save:
            torch.save(self.model.state_dict(), self.last_ckpt_path)
            print("[training] last model saved at epoch {}".format(last_epoch))

    def _train_each_epoch(self, train_loader):
        bp_every_iters = self.cfg['bp_every_iters']
        collector = {'y': None, 'y_hat': None}
        bp_collector = {'y': None, 'y_hat': None}
        all_loss  = []

        self.model.train()
        i_batch = 0
        for fx, fx5, cx5, y in train_loader:
            i_batch += 1
            # 1. forward propagation
            fx = fx.cuda()
            fx5 = fx5.cuda()
            cx5 = cx5.cuda() if self.model_pe else None
            y = y.cuda()

            if self.cfg['task'] == 'HierSurv':
                y_hat = self.model(fx, fx5, cx5)
            elif self.cfg['task'] == 'GenericCAPSurv':
                y_hat = self.model(fx, fx5, cx5)
            elif self.cfg['task'] == 'GenericSurv':
                y_hat = self.model(fx, cx5)

            # PLE: y_hat.shape = (B, 1),    y.shape = (B, 1)
            # MLE: y_hat.shape = (B, BINS), y.shape = (B, 2)
            collector = collect_tensor(collector, y.detach().cpu(), y_hat.detach().cpu())
            bp_collector = collect_tensor(bp_collector, y, y_hat)

            if bp_collector['y'].size(0) % bp_every_iters == 0:
                # 2. backward propagation
                if self.cfg['loss'] == 'survple' and torch.sum(bp_collector['y'] > 0).item() <= 0:
                    print("[warning] batch {}, event count <= 0, skipped.".format(i_batch))
                    bp_collector = {'y': None, 'y_hat': None}
                    continue
                
                # 2.1 zero gradients buffer
                self.optimizer.zero_grad()
                
                # 2.2 calculate loss
                loss = self.loss(bp_collector['y'], bp_collector['y_hat'])
                loss += self.loss_l1(self.model.parameters())
                all_loss.append(loss.item())
                print("[training epoch] training batch {}, loss: {:.6f}".format(i_batch, loss.item()))

                # 2.3 backwards gradients and update networks
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
                
                bp_collector = {'y': None, 'y_hat': None}

        return collector, sum(all_loss)/len(all_loss)

    @staticmethod
    def test_model(model, loader, task, checkpoint=None, model_pe=False):
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))

        model.eval()
        res = {'y': None, 'y_hat': None}
        with torch.no_grad():
            for x1, x2, c, y in loader:
                x1 = x1.cuda()
                x2 = x2.cuda()
                c = c.cuda() if model_pe else None
                y  = y.cuda()
                if task == 'HierSurv':
                    y_hat = model(x1, x2, c)
                elif task == 'GenericCAPSurv':
                    y_hat = model(x1, x2, c)
                elif task == 'GenericSurv':
                    y_hat = model(x1, c)
                res = collect_tensor(res, y.detach().cpu(), y_hat.detach().cpu())
        return res
