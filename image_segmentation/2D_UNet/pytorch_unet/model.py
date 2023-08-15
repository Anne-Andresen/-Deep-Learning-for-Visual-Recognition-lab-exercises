import os

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader

from skimage import io

from time import time

from .utils import Logger, MetricList
from .dataset import ImageToImage2D, Image2D


def maybe_mkdir(path):

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

class Model:

    def __init__(self, net: nn.Module, loss, optimizer,  checkpoint_folder: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 device: torch.device = torch.device('cpu')):

        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_folder = checkpoint_folder
        maybe_mkdir(self.checkpoint_folder)

        self.device = device

        self.net.to(device=self.device)

        try:
            self.loss.to(device=self.device)
        except:
            pass


    def fit_epoch(self, dataset, n_batch, shuffle=False):

        self.net.train(True)
        epoch_running_loss = 0

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=shuffle)):
            X_batch = Variable(X_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))

            # training
            self.optimizer.zero_grad()
            output = self.net(X_batch)
            train_loss = self.loss(output, y_batch)
            train_loss.backwards()
            self.optimizer.step()
            epoch_running_loss += train_loss.item()
        self.net.train(False)

        del X_batch, y_batch
        logs = {'train_loss': epoch_running_loss / (batch_idx + 1)}
        return logs

    def val_epoch(self, dataset, n_batch, metric_list=MetricList({})):

        self.net.train(False)
        metric_list.reset()
        val_epoch_loss = 0.0

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch)):
            X_batch = Variable(X_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))

            out_val = self.net(X_batch)
            val_loss =self.loss(out_val, y_batch)
            val_epoch_loss += val_loss
            metric_list(out_val, y_batch)
        del X_batch, y_batch
        logs = {'val loss': val_epoch_loss/(batch_idx+1), **metric_list.get_results(normalize=batch_idx+1)}
        return  logs


    def fit_dataset(self, dataset: ImageToImage2D, n_epochs: int, n_batch: int = 1, shuffle: bool = False,
                    val_dataset: ImageToImage2D = None, save_freq: int = 100, save_model: bool = False,
                    predict_dataset: Image2D = None, metric_list: MetricList = MetricList({}),
                    verbose: bool = False):
        logger = Logger(verbose=verbose)
        minimum_loss = np.inf
        train_start = time()

        for epoch_idx in range(1, n_epochs+1):
            train_logs = self.fit_epoch(dataset, n_batch=n_batch, shuffle=shuffle)

            if self.scheduler is not None:
                self.scheduler.step(train_logs['train loss'])
            if val_dataset is not None:
                val_logs = self.val_epoch(val_dataset, n_batch=n_batch, metric_list=metric_list)
                loss = val_logs['val loss']
            else:
                loss = train_logs['train loss']
             #Early stoppijh / saving best model
            if save_model:
                if loss < minimum_loss:
                    torch.save(self.net, os.path.join(self.checkpoint_folder, 'best_model.pt'))
                    minimum_loss = val_logs['val loss']
                # we are paranoid so we also save latest model
                torch.save(self.net, os.path.join(self.checkpoint_folder, 'latest_model.pt'))

            epoch_end = time()
            logs = {'':epoch_idx, 'time': epoch_end-train_start, 'memory': torch.cuda.memory_allocated(), **val_logs, **train_logs}
            logger.log(logs)
            logger.to_csv(os.path.join(self.checkpoint_folder, 'logs.csv'))

            if save_freq and (epoch_idx % save_freq == 0):
                epoch_save_path = os.path.join(self.checkpoint_folder, str(epoch_idx).zfill(4))
                maybe_mkdir(epoch_save_path)
                torch.save(self.net, os.path.join(epoch_save_path, 'model.pt'))
                if predict_dataset:
                    self.predict_dataset(predict_dataset, epoch_save_path)
        self.logger = logger

        return logger


    def predict_dataset(self, dataset, path):

        self.net.train(False)

        maybe_mkdir(path)

        for batch_idx, (X_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1)):

            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx +1).zfill(3)
            X_batch = Variable(X_batch.to(device=self.device))
            pred_out = self.net(X_batch)
            io.imsave(os.path.join(path, image_filename), pred_out[0, 1, :, :])

















