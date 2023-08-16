import numpy as np
import torch
import torch.nn as nn
from blocks import *
import skimage as io # compatible with simple itk so nifti and dicom okay
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader


from time import time

from .utils import Logger, MetricList
from .dataset import ImageToImage3D, Image3D



def maybe_mkdir(path):

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)



class Model:

    def __init__(self, model: nn.Module, loss, optimizer,checkpoint_folder: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 device: torch.device = torch.device('cpu')):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.checkpoint_folder = checkpoint_folder
        maybe_mkdir(checkpoint_folder)
        self.scheduler = scheduler
        self.device = device

        self.model.to(device)

        try:
            self.loss.to(device)
        except:
            pass


    def fit_epoch(self, dataset, n_batch, shuffle=False):
        self.model.train(True)
        epoch_running_loss = 0.0

        for batch, (x_batch, y_batch, *rest) in enumerate(DataLoader(dataset,batch_size=n_batch, shuffle=shuffle)):
            x_batch = Variable(x_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))
            self.optimizer.zero_grad()

            out = self.model(x_batch)
            train_loss = self.loss(out, y_batch)
            train_loss.backwards()
            self.optimizer.step()
            epoch_running_loss += train_loss.item()

        self.model.train(False)

        del x_batch, y_batch

        logs = {'train_loss': epoch_running_loss / batch +1}

        return logs

    def val_epoch(self, dataset, n_batch, metric_list=MetricList({})):
        self.model.train(False)
        metric_list.reset()
        val_epoch_loss = 0.0

        for batch, (x_batch, y_batch, *rest) in enumerate(DataLoader(dataset, n_batch)):
            x_batch = Variable(x_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))

            out = self.model(x_batch)
            val_loss = self.loss(val_loss, y_batch)
            val_epoch_loss += val_loss

        del x_batch, y_batch

        logs = {'val loss': val_epoch_loss/(batch +1), **metric_list.get_results(normalize=batch+1)}

        return logs

    def fit_dataset(self, dataset: ImageToImage3D, n_epochs: int, n_batch: int=1, shuffle: bool=False,
                    val_dataset: ImageToImage3D=None, save_freq:int = 10, save_model:bool = False,
                    predict_dataset: Image3D=None, metric_list:MetricList=MetricList({}), verbose: bool=False):

        logger = Logger(verbose=verbose)
        minimum_loss = np.inf
        train_start = time()

        for epoch in range(1, n_epochs+1):
            train_logs = self.fit_epoch(dataset, n_batch=n_batch, shuffle=shuffle)

            if self.scheduler is not None:
                self.scheduler.step(train_logs['train_loss'])
            if val_dataset is not None:
                val_logs = self.val_epoch(val_dataset, n_batch=n_batch, metric_list=metric_list)
                loss = val_logs["val loss"]
            else:
                loss = train_logs["train_loss"]

            if save_model:
                if loss < minimum_loss:
                    torch.save(self.model, os.path.join(self.checkpoint_folder, 'best_model.pt'))

                epoch_end = time()
                logs = {'epoch: ': epoch, 'time': epoch_end-train_start, 'memory: ': torch.cuda.memory_allocated(), **val_logs, **train_logs}
                logger.log(logs)
                logger.to_csv(os.path.join(self.checkpoint_folder, 'csv_log.csv'))

                if save_freq and (epoch % save_freq == 0):
                    epoch_save_path = os.path.join(self.checkpoint_folder, str(epoch).zfill(4))
                    maybe_mkdir(epoch_save_path)
                    if predict_dataset:
                        self.predict_dataset(predict_dataset, epoch_save_path)

        self.logger = logger

        return logger

    def predict_dataset(self, dataset, path):

        self.model.train(False)
        maybe_mkdir(path)

        for batch, (x_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1)):

            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch + 1).zfill(3)

            x_batch= Variable(x_batch.to(device=self.device))
            out = self.model(x_batch)
            io.imsave(os.path.join(path, image_filename), out[0, 1, :, :])












