import torch.nn as nn
import torch
import numpy as np
import glob
import os
from torch import optim
from utils import *
import datetime
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import parameter_file as pf
from tqdm import tqdm
from model import Generator, Discriminator

def maybe_mkdir(path):
    ''
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)



def save_train_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['train_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['train_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def save_val_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['val_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['val_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def train():


    save_file_path = './save_model/model.pt'

    dataset_path = pf.data_dir + '/train/'
    model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-/S")
    writer = SummaryWriter(pf.output_dir + '/model/' + '/logs/')
    image_save_path = pf.output_dir + '/images'
    maybe_mkdir(image_save_path)
    model_save_path = pf.output_dir + '/save_model/'
    maybe_mkdir(model_save_path)
    print('path to dataset: ', dataset_path)

    # create dataset
    train_ds = ShapeNetDataset(dataset_path, train_or_val="train")
    train_loader = torch.utils.data.Dataloader(train_ds, batch_size=pf.batch_size, shuffle=False, num_workers=1)
    dset_len = {"train": len(train_ds)}
    dset_loaders = {"train": train_loader}
    gen = Generator()
    disk = Discriminator()
    G_solve = optim.Adam(gen.parameters(), lr=pf.g_lr, betas=pf.beta)

    D_solve = optim.Adam(disk.parameters(), lr=pf.d_lr, betas=pf.beta)

    gen.to(pf.device)
    disk.to(pf.device)
    criterion_gen = nn.L1Loss()
    criterion_disk = nn.MSELoss()

    itr_val = -1
    itr_train = -1

    for epoch in range(pf.epochs):
        start = time.time()

        for phase in ['train']:
            if phase == 'train':
                disk.train()
                gen.train()
            else:
                disk.eval()
                gen.eval()

            running_loss_gen = 0.0
            running_loss_disk = 0.0
            running_loss_adv = 0.0


            for i, X in enumerate(tqdm(train_loader[phase])):

                if phase == 'train':
                    itr_train +=1

                    X = X.to(pf.device)

                    batch = X.size()[0]
                    Z = generateZ(batch=batch)


                    d_real = disk(X)
                    fake = gen(Z)
                    d_fake = disk(fake)

                    real_label = torch.ones_like(d_real).to(pf.device)
                    fake_label = torch.zeros_like(d_fake).to(pf.device)

                if pf.soft_label:
                    real_label = torch.Tensor(batch).uniform_(0.7, 1.2).to(pf.device)
                    fake_label = torch.Tensor(batch).uniform_(0.0, 0.3).to(pf.device)

                d_real_loss = criterion_disk(real_label, real_label)
                d_fake_loss = criterion_disk(d_fake, fake_label)
                d_loss = d_real_loss + d_fake_loss


                d_acu_real = torch.ge(d_real.squeeze(), 0.5).float()
                d_acu_fake = torch.le(d_fake.squeeze(), 0.5).float()
                d_acu_total = torch.mean(torch.cat((d_acu_real, d_acu_fake), 0))

                if d_acu_total < pf.d_thresh:
                    disk.zero_grad()
                    d_loss.backward()
                    D_solve.step()


                Z = generateZ(batch=batch)
                fake = gen(Z)
                d_fake = disk(fake)

                adv_g_loss = criterion_disk(d_fake, real_label)

                recon_g_loss = criterion_gen(fake, X)

                g_loss = adv_g_loss

                    #if local_test:
                    #    print('')
                disk.zero_grad()
                gen.zero_grad()
                g_loss.backward()
                G_solve.step()


                running_loss_gen += recon_g_loss.item() * X.size(0)
                running_loss_disk += d_loss.item() *X.size(0)
                running_loss_adv += adv_g_loss.item()*X.size(0)

            epoch_loss_gen = running_loss_gen /dset_len[phase]

            epoch_loss_disk = running_loss_disk/dset_len[phase]
            epoch_loss_adv_G = running_loss_adv/dset_len[phase]

            end = time.time()
            epoch_time = end-start


            if (epoch+1) % pf.model_save_step == 0:

                torch.save(gen.state_dict(), model_save_path + 'G.pth')
                torch.save(disk.state_dict(), model_save_path + 'D.pth')

                samples = fake.cpu().data[:8].squeeze().numpy()

                SavePloat_Voxels(samples, image_save_path, epoch)




























    #for batch_icx, epoch in glob.glob(dataset_path)

