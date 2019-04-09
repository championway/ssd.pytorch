import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import os.path as osp
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

class ImageDataset(Dataset):
    def __init__(self, root = "/media/arg_ws3/5E703E3A703E18EB/data/subt_real_ssd/", transform_=None, mode='train'):
        self.transform = transforms.Compose(transform_)
        self.root = root
        self.files = []
        for line in open(osp.join(self.root, 'ImageSets/Main', mode + '.txt')):
            self.files.append(line.strip().split(' ')[0])
        #self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        #if mode == 'train':
        #    self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

    def __getitem__(self, index):
        file = self.files[index % len(self.files)]
        path = self.root + 'JPEGImages/' + file + '.png'
        img = Image.open(path)
        bbxs = self.get_bbx(file)
        '''target = []
        for i in bbxs[0]:
            target.append(torch.DoubleTensor([i]))'''
        img = np.array(img)/10000.
        #print(np.min(img), np.max(img))
        img = Image.fromarray(img)
        #img = Image.fromarray(img).convert('RGB')

        #if np.random.random() < 0.5:
        #    img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
        #    img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img = self.transform(img)
        return img, bbxs[0]

    def get_bbx(self, file, width=640., height=480.):
        path = self.root + 'Annotations/' + file + '.xml'
        target = ET.parse(path).getroot()
        res = []
        for obj in target.iter('object'):
            #difficult = int(obj.find('difficult').text) == 1
            #if not self.keep_difficult and difficult:
            #    continue
            name = obj.find('name').text.lower().strip()
            if name != 'bb_extinguisher':
                continue
            bbox = obj.find('bndbox')
            if bbox is not None:
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(bbox.find(pt).text) - 1
                    # scale height or width
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                    bndbox.append(cur_pt)
                #label_idx = self.class_to_ind[name]
                bndbox.append(1)
                res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            else: # For LabelMe tool
                polygons = obj.find('polygon')
                x = []
                y = []
                bndbox = []
                for polygon in polygons.iter('pt'):
                    # scale height or width
                    x.append(int(polygon.find('x').text) / width)
                    y.append(int(polygon.find('y').text) / height)
                bndbox.append(min(x))
                bndbox.append(min(y))
                bndbox.append(max(x))
                bndbox.append(max(y))
                bndbox.append(0)
                res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        return res # [[xmin, ymin, xmax, ymax, label_ind], ... ]

    '''def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}'''

    def __len__(self):
        return len(self.files)
