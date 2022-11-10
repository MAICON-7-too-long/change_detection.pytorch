import os.path as osp
import sys
import math

import torch
from torch import autocast
from torch.cuda.amp import GradScaler

from tqdm import tqdm as tqdm

from .meter import AverageValueMeter


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', classes=1, wandb=None, verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.classes = classes
        self.wandb = wandb

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        # for metric in self.metrics:
        #     metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x1, x2, y, scaler):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def check_tensor(self, data, is_label):
        if not is_label:
            return data if data.ndim <= 4 else data.squeeze()
        return data.long() if data.ndim <= 3 else data.squeeze().long()

    def predict(self, best_model, dataloader, save_dir='./infer_res', suffix='.png'):
        import cv2

        self.model = best_model

        self.model.eval()

        self.infer_table = self.wandb.Table(columns=['Name', 'Image1', 'Image2'])

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for (x1, x2, filename) in iterator:
                x1, x2 = self.check_tensor(x1, False), self.check_tensor(x2, False)
                x1, x2 = x1.float(), x2.float()
                x1, x2 = x1.to(self.device), x2.to(self.device)
                y_pred = self.model.forward(x1, x2)

                y_pred = torch.argmax(y_pred, dim=1).squeeze().cpu().numpy().round()

                for (fname, xx1, xx2, yy_pred) in zip(filename, x1, x2, y_pred):
                    fname = fname.split('.')[0] + suffix

                    img1 = self.wandb.Image(xx1)
                    img2 = self.wandb.Image(xx2, masks = {
                        "prediction" : {
                            "mask_data" : yy_pred,
                            "class_labels" : { i+1 : str(i+1) for i in range(self.classes)} 
                        },
                    })
                    
                    self.infer_table.add_data(fname, img1, img2)
                    
                    if fname in ["1033.png", "1621.png", "1787.png", "1315.png", "2446.png", "1820.png"]:
                        yy_pred = cv2.resize(yy_pred, (750, 750), interpolation=cv2.INTER_NEAREST)
                    else:
                        yy_pred = cv2.resize(yy_pred, (754, 754), interpolation=cv2.INTER_NEAREST)
                    
                    cv2.imwrite(osp.join(save_dir, fname), yy_pred)

        self.wandb.log({"Table" : self.infer_table})

    def infer_vis(self, dataloader, save=True, evaluate=False, slide=False, image_size=1024, 
                    window_size=256, save_dir='./infer_res', suffix='.png'):
        """
        Infer and save results. (debugging)
        Note: Currently only batch_size=1 is supported.
        Weakly robust.
        'image_size' and 'window_size' work when slide is True.
        """
        import cv2
        import numpy as np

        self.model.eval()
        logs = {}
        metrics_meters = {metric.name: AverageValueMeter() for metric in self.metrics}

        self.infer_table = self.wandb.Table(columns=['Name', 'Image1', 'Image2'])

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            
            # for (x1, x2, y, filename) in iterator:
            for (x1, x2, filename) in iterator:
                y = None

                assert y is not None or not evaluate, "When the label is None, the evaluation mode cannot be turned on."

                if y is not None:
                    x1, x2, y = self.check_tensor(x1, False), self.check_tensor(x2, False), \
                                self.check_tensor(y, True)
                    x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                    y_pred = self.model.forward(x1, x2)
                else:
                    x1, x2 = self.check_tensor(x1, False), self.check_tensor(x2, False)
                    x1, x2 = x1.float(), x2.float()
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    y_pred = self.model.forward(x1, x2)

                if evaluate:
                    # update metrics logs
                    for metric in self.metrics:
                        # metric_value = metric_fn(y_pred, y).detach().cpu().numpy()
                        metric_value = metric.get_iou(y_pred.argmax(1), y).detach().cpu().numpy()
                        # metrics_meters[metric_fn.__name__].add(metric_value)
                        metrics_meters[metric.name].add(metric_value)
                    metrics_logs = {"evaluate_" + k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)
                    self.wandb.log(metrics_logs)

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)

                if save:
                    y_pred = torch.argmax(y_pred, dim=1).squeeze().cpu().numpy().round()
                    # y_pred = y_pred * 255

                    for (fname, xx1, xx2, yy_pred) in zip(filename, x1, x2, y_pred):
                        fname = fname.split('.')[0] + suffix

                        img1 = self.wandb.Image(xx1)
                        # img2 = self.wandb.Image(xx2)
                        img2 = self.wandb.Image(xx2, masks = {
                            "prediction" : {
                                "mask_data" : yy_pred,
                                "class_labels" : { i+1 : str(i+1) for i in range(self.classes)} 
                            },
                        })
                        
                        self.infer_table.add_data(fname, img1, img2)

                        cv2.imwrite(osp.join(save_dir, fname), yy_pred)
                        # save_image(torch.tensor(yy_pred, dtype=torch.long), osp.join(save_dir, fname + "by_cv2" + suffix))

                    # if slide:
                    #     inf_seg_maps = []
                    #     window_num = image_size // window_size
                    #     window_idx = [i for i in range(0, window_num ** 2 + 1, window_num)]
                    #     for row_idx in range(len(window_idx) - 1):
                    #         inf_seg_maps.append(np.concatenate([y_pred[i] for i in range(window_idx[row_idx],
                    #                                                                      window_idx[row_idx + 1])], axis=1))
                    #     inf_seg_maps = np.concatenate([row for row in inf_seg_maps], axis=0)
                    #     cv2.imwrite(osp.join(save_dir, filename), inf_seg_maps)
                    # else:
                    #     # To be verified
                    #     cv2.imwrite(osp.join(save_dir, filename), y_pred)
            
            self.wandb.log({"Table" : self.infer_table})

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.name: AverageValueMeter() for metric in self.metrics}
        scaler = GradScaler()
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for (x1, x2, y, filename) in iterator:

                x1, x2, y = self.check_tensor(x1, False), self.check_tensor(x2, False), \
                            self.check_tensor(y, True)
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x1, x2, y, scaler)

                # update loss logs
                loss_value = loss.detach().cpu().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.stage_name + "_" + self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)
                self.wandb.log(loss_logs)

                # update metrics logs
                for metric in self.metrics:
                        # metric_value = metric_fn(y_pred, y).detach().cpu().numpy()
                        metric_value = metric.get_iou(y_pred.argmax(1), y).detach().cpu().numpy()
                        # metrics_meters[metric_fn.__name__].add(metric_value)
                        metrics_meters[metric.name].add(metric_value)
                metrics_logs = {self.stage_name + "_" + k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
                self.wandb.log(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', classes=1, wandb=None, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            classes=1,
            wandb=wandb,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x1, x2, y, scaler):
        self.optimizer.zero_grad()
	
        with autocast(device_type='cuda', dtype=torch.float16):
            prediction = self.model.forward(x1, x2)
            loss = self.loss(prediction, y)

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', classes=1, wandb=None, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            classes=1,
            wandb=wandb,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x1, x2, y, scaler):
        with torch.no_grad():
            prediction = self.model.forward(x1, x2)
            loss = self.loss(prediction, y)
        return loss, prediction
