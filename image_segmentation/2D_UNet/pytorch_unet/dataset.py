import os
import numpy as np
import torch

from skimage import io

from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable


def to_long_tensor(pic):
    img = torch.from_numpy(np.array(pic, np.uint8))
    return img.long()

def correct_dims(*images):
    corr_imgs = []
    for img in images:
        if len(img.shape) == 2:
            corr_imgs.append(np.expand_dims(img, axis=2))
        else:
            corr_imgs.append(img)

    if len(corr_imgs)==1:
        return corr_imgs[0]
    else:
        return corr_imgs

# Doing a bit of image augmentation
class JointTransform2D:
    def __init__(self, crop=(256, 256), flip_in=0.5, color_jit=(0.1,0.1), p_random_affine=0, long_mask=False):
        super(JointTransform2D).__init__()
        self.crop = crop
        self.flip_in = flip_in
        self.color_jit = color_jit
        if color_jit:
            self.color_tf = T.ColorJitter(*color_jit)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        if np.random.rand() <= self.flip_in:
            image, mask = F.hflip(image), F.hflip(mask)

        if self.color_jit:
            image = self.color_tf(image)

        if np.random.rand() <= self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        #kears, pytorch so. only works with tensors so we transform to tensors
        image = F.to_tensor(image)

        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image,mask


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'images')
        self.output_path = os.path.join(self.dataset_path, 'labels')
        self.image_list = os.listdir(self.input_path)
        self.one_hot_label = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda  x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, item):
        image_filename = self.image_list[item]
        image = io.imread(os.path.join(self.input_path, image_filename))
        label = io.imread(os.path.join(self.output_path, image_filename))
        image , label = correct_dims(image, label)

        if self.joint_transform:
            image, label = self.joint_transform(image, label)

        if self.one_hot_label:
            assert self.one_hot_label > 0, 'one hot must be none negative'
            label = torch.zeros((self.one_hot_label, label.shape[1], label.shape[2])).scatter_(0,label.long(), 1)

        return image, label, image_filename






class Image2D(Dataset):

    def __init__(self, dataset_path: str, transform: Callable = None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'images')
        self.output_path = os.path.join(self.dataset_path, 'labels')
        self.image_lst = os.listdir(self.input_path)
        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor


    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, item):
        image_filename = self.image_lst[item]
        image = io.imread(os.path.join(self.input_path, image_filename))
        image = correct_dims(image)
        image = self.transform(image)
        return image, image_filename







