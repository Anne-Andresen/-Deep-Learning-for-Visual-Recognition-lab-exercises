import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch
from typing import Callable
import SimpleITK as sitk
#imread replaed by sitk(read) --> np.array(sitk.GetArrayFromImage())


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 3:
            corr_images.append(np.expand_dims(img, axis=3))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images






class ImageToImage3D(Dataset):


    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        # read image
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.input_path, image_filename)))
        # read mask image
        mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.output_path, image_filename)))

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        '''
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        '''
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        return image, mask, image_filename


class Image3D(Dataset):


    def __init__(self, dataset_path: str, transform: Callable = None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.images_list = os.listdir(self.input_path)

        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.input_path, image_filename)))

        # correct dimensions if needed
        image = correct_dims(image)

        #image = self.transform(image)

        return image, image_filename

