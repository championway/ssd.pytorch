"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import os
import sys
import torch
import torch.utils.data as data
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import pandas as pd
import scipy.misc
import random

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

MMFCN_CLASSES = ['3m', 'andes', 'cocacola', 'crayola', 'folgers','heineken','hunts','kellogg','kleenex',\
               'kotex','libava','macadamia','milo','mm','pocky','raisins','stax','swissmiss','vanish','viva']

# note: if you used our download scripts, this should be right
#MMFCNREAL_ROOT = osp.join("/media/arg_ws3/5E703E3A703E18EB/data/mm_FCN/", "box_gan_LAMDAID_1")
MMFCNREAL_ROOT = osp.join("/media/arg_ws3/5E703E3A703E18EB/data/mm_FCN/", "unity")


class MMFCNAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(MMFCN_CLASSES, range(len(MMFCN_CLASSES))))
        self.keep_difficult = keep_difficult
    def __call__(self, mask, width, height):
        res = []
        x, y, w, h = cv2.boundingRect(mask)
        if (x, y, w, h) != (0, 0, 0, 0): # if there are objects exist
            res += [[x / width, y / height, (x+w) / width, (y+h) / height, int(mask[mask!=0][0])-1]]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class MMFCNDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('train'), ('test')],
                 transform=None, target_transform=MMFCNAnnotationTransform(),
                 dataset_name='unity',
                 csv_file = os.path.join("/media/arg_ws3/5E703E3A703E18EB/data/mm_FCN/unity/train.csv")):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.data_dir  = os.path.join("/media/arg_ws3/5E703E3A703E18EB/data/mm_FCN", dataset_name)
        if not os.path.exists(self.data_dir):
            print("Data not found!")
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.png')
        self.data = pd.read_csv(csv_file)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.data)

    def pull_item(self, index):
        img_id   = self.data.iloc[index, 0]
        label_id = self.data.iloc[index, 1]

        img = cv2.imread(os.path.join(self.data_dir, img_id),cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(os.path.join(self.data_dir, label_id), cv2.IMREAD_GRAYSCALE)

        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(mask, width, height)
        else:
            print("WRONG!!!")

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.data.iloc[index, 0]
        return cv2.imread(os.path.join(self.data_dir, img_id),cv2.IMREAD_UNCHANGED)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        label_id = self.data.iloc[index, 1]
        mask = cv2.imread(os.path.join(self.data_dir, label_id), cv2.IMREAD_GRAYSCALE)
        gt = self.target_transform(mask, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
