import os
from os.path import join
import time
import logging
from typing import Callable, Optional, Union, Dict, Tuple
import gin
from fire import Fire
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from exp.base import Experiment
from data_provider.data_loader import ForecastDataset
from model import get_model
from utils.checkpoint import Checkpoint
from utils.ops import default_device
from utils.losses import get_loss_fn
from utils.metrics import calc_metrics
from utils.tools import adjust_learning_rate, save_to_txt, visual


class ForecastExperiment(Experiment):
    @gin.configurable()
    def instance(self,
                 model_type: str,
                 save_vals: Optional[bool] = True,
                 save_txt: Optional[bool] = True):
        train_set, train_loader = get_data(flag='train')
        val_set, val_loader = get_data(flag='val')
        test_set, test_loader = get_data(flag='test')
        logging.info(f"train:{len(train_set)} val:{len(val_set)} test:{len(test_set)}")
        model = get_model(model_type, default_device, inputs_feats=train_set.timestamps.shape[-1])
        checkpoint = Checkpoint(self.root)
        model = train(model, checkpoint, train_loader, val_loader, test_loader)
        val_metrics = validate(model, loader=val_loader, report_metrics=True)
        test_metrics = validate(model, loader=test_loader, report_metrics=True,
                                save_path=self.root if save_vals else None)
        np.save(join(self.root, 'metrics.npy'), {'val': val_metrics, 'test': test_metrics})
        save_to_txt(self.root) if save_txt else None
        val_metrics = {f'ValMetric/{k}': v for k, v in val_metrics.items()}
        test_metrics = {f'TestMetric/{k}': v for k, v in test_metrics.items()}
        checkpoint.close({**val_metrics, **test_metrics})


@gin.configurable()
def get_optimizer(model: nn.Module,
                  lr: Optional[float] = 1e-3) -> optim.Optimizer:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer, lr


@gin.configurable()
def get_data(flag: bool,
             batch_size: int,
             num_workers: Optional[int] = 10) -> Tuple[ForecastDataset, DataLoader]:
    if flag in ('val', 'test'):
        shuffle = False
        drop_last = False
    elif flag == 'train':
        shuffle = True
        drop_last = True
    else:
        raise ValueError(f'no such flag {flag}')
    dataset = ForecastDataset(flag)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             drop_last=drop_last)
    return dataset, data_loader


@gin.configurable()
def train(model: nn.Module,
          checkpoint: Checkpoint,
          train_loader: DataLoader,
          val_loader: DataLoader,
          test_loader: DataLoader,
          loss_name: str,
          epochs: int) -> nn.Module:
    optimizer, lr = get_optimizer(model)
    training_loss_fn = get_loss_fn(loss_name)
    train_steps = len(train_loader)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f'parameters: {num_params/1e6:.3f}M')
    time_now = time.time()

    for epoch in range(epochs):
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()
        for it, (x, y, x_inputs, y_inputs) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()
            x, y, x_inputs, y_inputs = x.float().to(default_device()), y.float().to(
                default_device()), x_inputs.float().to(default_device()), y_inputs.float().to(default_device())
            forecast = model(x, x_inputs, y_inputs)
            loss = training_loss_fn(forecast, y)

            train_loss.append(loss.item())
            if (it + 1) % 100 == 0:
                logging.info(f"epochs: {epoch + 1}, iters: {it + 1} | loss: {loss.item():.5f}")

                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((epochs - epoch) * train_steps - it)
                logging.info(f"\tspeed: {speed:.5f}s/iter; left time: {left_time:.5f}s")
                iter_count = 0
                time_now = time.time()

            loss.backward()
            optimizer.step()

        logging.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")

        train_loss = np.average(train_loss)
        val_loss = validate(model, loader=val_loader, loss_fn=training_loss_fn)
        test_loss = validate(model, loader=test_loader, loss_fn=training_loss_fn)

        logging.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.5f} "
                     f"Vali Loss: {val_loss:.5f} Test Loss: {test_loss:.5f}")

        scalars = {'Loss/Train': train_loss,
                   'Loss/Val': val_loss,
                   'Loss/Test': test_loss}
        checkpoint(epoch + 1, model, scalars=scalars)
        if checkpoint.early_stop:
            logging.info("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch + 1, lr)

    if epochs > 0:
        model.load_state_dict(torch.load(checkpoint.model_path))
    return model


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             loss_fn: Optional[Callable] = None,
             report_metrics: Optional[bool] = False,
             save_path: Optional[str] = None) -> Union[Dict[str, float], float]:
    if save_path is not None:
        folder_path = save_path / 'test_visual'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    model.eval()
    preds = []
    trues = []
    inps = []
    total_loss = []
    for it, (x, y, x_inputs, y_inputs) in enumerate(loader):
        x, y, x_inputs, y_inputs = x.float().to(default_device()), y.float().to(default_device()), x_inputs.float().to(
            default_device()), y_inputs.float().to(default_device())
        if x.shape[0] == 1:
            continue
        forecast = model(x, x_inputs, y_inputs)
        if report_metrics:
            preds.append(forecast)
            trues.append(y)
            if save_path is not None:
                inps.append(x)
                if it % 20 == 0:
                    inp, true, pred = x.detach().cpu().numpy(), y.detach().cpu().numpy(), forecast.detach().cpu().numpy()
                    gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(it) + '.pdf'))
        else:
            loss = loss_fn(forecast, y, reduction='none')
            total_loss.append(loss)

    if report_metrics:
        preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        trues = torch.cat(trues, dim=0).detach().cpu().numpy()
        if save_path is not None:
            inps = torch.cat(inps, dim=0).detach().cpu().numpy()
            np.save(join(save_path, 'inps.npy'), inps)
            np.save(join(save_path, 'preds.npy'), preds)
            np.save(join(save_path, 'trues.npy'), trues)
        metrics = calc_metrics(preds, trues)
        return metrics
    total_loss = torch.cat(total_loss, dim=0).cpu()
    return np.average(total_loss)


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(ForecastExperiment)