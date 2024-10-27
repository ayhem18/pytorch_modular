import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tr
from tqdm import tqdm

from mypt.shortcuts import P
from mypt.data.datasets.parallel_augmentation.parallel_aug_ds_wrapper import Food101Wrapper, ImagenetterWrapper
from mypt.data.datasets.genericFolderDs import Food101GenericWrapper, ImagenetteGenericWrapper

OUTPUT_SHAPE = (200, 200)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def check_ds_wrapper(folder: P,
                    train: bool, 
                    num_samples_per_cls: int):

    ds_aug = ImagenetterWrapper(root_dir=folder, 
                        output_shape=OUTPUT_SHAPE,
                        augs_per_sample=2, 
                        sampled_data_augs=[],
                        uniform_augs_before=[],
                        uniform_augs_after=[],
                        train=train,
                        samples_per_cls=num_samples_per_cls,
                        )

    ds_cls = ImagenetteGenericWrapper(root_dir=folder, 
                                   augmentations=[tr.ToTensor(), tr.Resize(size=OUTPUT_SHAPE)], 
                                   train=train, 
                                   samples_per_cls=num_samples_per_cls)
    
    assert len(ds_cls) == len(ds_aug), "The two datasets are of different sizes"

    # for i in range(len(ds_cls)[:5]):        
    #     _, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))  

    #     im1, im2, im3 = ds_cls[i].numpy(), ds_aug[i][0].numpy(), ds_aug[i][1].numpy()
    #     im1, im2, im3 = np.moveaxis(im1, 0,-1), np.moveaxis(im2, 0,-1), np.moveaxis(im3, 0,-1),

    #     ax[0].imshow(im1)
    #     ax[0].set_title("Augmentation Wrapper")

    #     ax[1].imshow(im2)
    #     ax[1].set_title("Augmentation Wrapper img 1")

    #     ax[2].imshow(im3)
    #     ax[2].set_title("Augmentation Wrapper img 2")
        
    #     plt.show()

    for i in tqdm(range(len(ds_cls)), desc='iterating through the datasets'):
        im1, im2, im3 = ds_cls[i].numpy(), ds_aug[i][0].numpy(), ds_aug[i][1].numpy()
        assert np.all(im1 == im2), "the images are different"
        assert np.all(im2 == im3), "the images are different"
        assert np.all(im3 == im1), "the images are different"

def check_ds_different_modes():
    # this function is visual inspection rather than an actual test
    t = os.path.join(SCRIPT_DIR, 'data', 'food101', 'train')
    v = os.path.join(SCRIPT_DIR, 'data', 'food101', 'val')

    check_ds_wrapper(t, 
                     train=True, 
                     classification_mode=True,
                     num_samples_per_cls=10)    

    check_ds_wrapper(t, 
                     train=True, 
                     classification_mode=False,
                     num_samples_per_cls=10)    

    check_ds_wrapper(v, 
                     train=False, 
                     classification_mode=True,
                     num_samples_per_cls=10)    

    check_ds_wrapper(v, 
                     train=False, 
                     classification_mode=False,
                     num_samples_per_cls=10)    


if __name__ == '__main__':
    t = os.path.join(SCRIPT_DIR, 'data', 'food101', 'train')
    v = os.path.join(SCRIPT_DIR, 'data', 'food101', 'val')


    check_ds_wrapper(t, train=False, num_samples_per_cls=None)

    for i in [10, 25, 50, 100]:
        check_ds_wrapper(t, train=False, num_samples_per_cls=i)
