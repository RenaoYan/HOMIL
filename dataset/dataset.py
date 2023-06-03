import os
import glob
import torch
import random
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset


# M2's dataset
class M2Dataset(Dataset):
    def __init__(self, split, feat_dir, discard_rate=0, mixup_num=0, fixed_feat_dir=None):
        self.slide_ids = list(split.keys())
        self.labels = list(split.values())
        self.feat_dir = feat_dir
        self.feat_files = self.get_feat()
        self.labeled_feat_files = self.get_labeled_feat()

        # feat augmentation
        self.discard_rate = discard_rate
        self.mixup_num = mixup_num

        # joint learning
        self.fixed_feat_dir = fixed_feat_dir

    def get_labels(self):
        return self.labels

    def get_feat(self):
        feat_files = {}
        for slide_id in self.slide_ids:
            feat_files[slide_id] = [os.path.join(self.feat_dir, slide_id + '.pt')]
        return feat_files

    def get_labeled_feat(self):
        labeled_feat_files = {}
        for k in self.labels:
            feat_files = []
            for i in range(len(self.slide_ids)):
                label = self.labels[i]
                if label == k:
                    slide_id = self.slide_ids[i]
                    feat_files.append(os.path.join(self.feat_dir, slide_id + '.pt'))
            labeled_feat_files.update({k: feat_files})
        return labeled_feat_files

    def __getitem__(self, idx):
        slide_name = self.slide_ids[idx]
        target = self.labels[idx]
        feat_files = self.feat_files[slide_name]
        feats = torch.Tensor()
        for feat_file in feat_files:
            feat = torch.load(feat_file, map_location='cpu')
            feats = torch.cat((feats, feat), dim=0)

        # feature augmentation
        if target == 0:  # non-tumor
            if self.discard_rate > 0:
                index = random.sample(range(len(feats)), int((1 - self.discard_rate) * len(feats)))
                feats = feats[index, :]
        else:  # tumor
            if self.mixup_num > 0:
                non_tumor_feat_files = random.sample(self.labeled_feat_files.get(0), self.mixup_num)
                for feat_file in non_tumor_feat_files:
                    feat = torch.load(feat_file, map_location='cpu')
                    feats = torch.cat((feats, feat), dim=0)

        sample = {'slide_id': slide_name, 'feat': feats, 'target': target}

        # joint learning
        if self.fixed_feat_dir is not None:
            fixed_feats_dir = os.path.join(self.fixed_feat_dir, slide_name + '.pt')
            fixed_feats = torch.load(fixed_feats_dir, map_location='cpu')
            sample.update({'fixed_feat': fixed_feats})
        return sample

    def __len__(self):
        return len(self.slide_ids)


# M1's dataset
class M1Dataset(Dataset):
    def __init__(self, split, patch_dir, transform=None, img_format='png'):
        self.slide_name = list(split.keys())
        self.coords, self.labels, self.slide_ID = self.get_data(split)
        self.patch_dir = patch_dir
        self.transform = transform
        self.img_format = img_format

    def get_labels(self):
        return self.labels

    def get_data(self, split):
        coords = []
        labels = []
        slide_ID = []
        for i, data in enumerate(split.values()):
            coord = data['coords']
            label = data['labels']
            coords.extend(coord)
            labels.extend(label)
            slide_id = [i] * len(label)
            slide_ID.extend(slide_id)
        return coords, labels, slide_ID

    def __getitem__(self, index):
        coord = np.squeeze(np.array(self.coords[index]))
        img_name = '{}_{}.{}'.format(int(coord[0]), int(coord[1]), self.img_format)
        img_dir = os.path.join(self.patch_dir, self.slide_name[self.slide_ID[index]])
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)


# feature extraction dataset
class Extract_Feat_Dataset(Dataset):
    def __init__(self, slide_name, transform=None, img_format='png'):
        self.image_dir = slide_name
        self.image_paths = glob.glob(os.path.join(self.image_dir, '*.{}'.format(img_format)))
        self.image_paths.sort()
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img_name = self.image_paths[index].split('/')[-1]
        coord_ = img_name.split('.')[0]
        coord = coord_.split('_')
        coord_x = int(coord[0])
        coord_y = int(coord[1])
        if self.transform is not None:
            img = self.transform(img)
        return {'image': img, 'coords': np.array([coord_x, coord_y])}

    def __len__(self):
        return len(self.image_paths)
