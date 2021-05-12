import math
import pickle
import operator
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import pandas as pd
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from scipy.io import loadmat
import json as jsonmod
import random
#from test import Test_index


# coco, nus, flickr三大简称
def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomSizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Scale(256), transforms.CenterCrop(224)]
        # t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Scale(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


class PrecompCOCODataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, trainratio=1.0):
        self.vocab = vocab
        loc = '../data/coco_precomp/'

        # Captions
        self.captions = []
        with open(loc + '%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                    self.captions.append(line.strip())
        # 长度566435

        # add
        self.cocoids = []
        with open(loc + '%s_ids.txt' % data_split, 'r') as f:
            for line in f:
                self.cocoids.append(line.strip())
        self.labels_tx = np.load(loc + "%s_label.npy" % data_split,mmap_mode='r+')
        self.labels_im = np.load(loc + "%s_label.npy" % data_split,mmap_mode='r+')
        self.pd_tx = pd.DataFrame(self.labels_tx)
        self.pd_im = pd.DataFrame(self.labels_im)
        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split,mmap_mode='r+')

        self.length = len(self.captions) #566435
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length: #113287 566435
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        self.num_class = len(self.labels_im[0])#80
        self.trainratio = trainratio
        self.data_split = data_split
        print('data_split:',data_split)
        print("image:", self.images.shape)
        print("text:", self.length)
        print("ids:", len(self.cocoids))

        if data_split == "train":
            self.image_clslist = np.load(loc + '%s_cls.npy' % data_split,mmap_mode='r+')
            self.image_clslist = np.argmax(self.image_clslist, axis=-1)


    def __getitem__(self, index):
        # labels与img_id ：captions = 1 : 5
        img_id = index // self.im_div
        img_index = img_id

        image = torch.Tensor(self.images[img_index])

        cap_index = index
        caption = self.captions[index]
        vocab = self.vocab

        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        caption, masklist_tx = make_caption(vocab, tokens, maskNeed=(self.data_split == "train"))

        # SCANimagefeature,句子，句子特征，类别，图片路径，图片单一特征
        if self.data_split == "train":
            imcls = self.image_clslist[img_id]
            masklist_im = mask_image(imcls)

        caption = torch.Tensor(caption)
        label_im = torch.Tensor(self.labels_im[img_index])
        if self.data_split == "train":
            label_tx = torch.Tensor(self.labels_tx[cap_index // 5])
        else:
            label_tx = torch.Tensor(self.labels_tx[cap_index])



        if self.data_split == "train":
            return image, caption, index, label_im, label_tx, masklist_im, masklist_tx
        else:
            return image, caption, index, label_im, label_tx, None, None

    def __len__(self):
        return self.length


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: flickr_precomp, nus_precomp
    """
    def __init__(self, data_path, dset_image, dset_text, data_split, vocab, trainratio=1.0):
        # root 根目录
        dset_image = dset_image.split("_")[0]
        dset_text = dset_text.split("_")[0]
        print("class PrecompDataset: ", dset_image)
        print("class PrecompDataset: ", dset_text)

        if data_split == "dev":
            data_split = "test"
        dset_im_path = os.path.join(data_path, dset_image) + '/'
        dset_tx_path = os.path.join(data_path, dset_text) + '/'
        self.box = np.load(dset_tx_path + '%s_box.npy' % data_split)
        self.hw = np.load(dset_tx_path + '%s_hw.npy' % data_split)

        # Captions
        self.captions = loadmat(dset_tx_path + '%s_tag.mat' % data_split)['feat']
        print('data_m 188:   ', len(self.captions),self.captions.shape)
        self.labels_tx = loadmat(dset_tx_path + '%s_label.mat' % data_split)['feat']
        print('data_m 191:   ', self.labels_tx.shape)
        self.pd_tx = pd.DataFrame(loadmat(dset_tx_path + '%s_label.mat' % data_split)['feat'])

        self.image_feature = np.load(dset_im_path + '%s_preim.npy' % data_split, allow_pickle=True,mmap_mode='r+')
        print('data_m 197:   ', self.image_feature.shape)
        self.labels_im = loadmat(dset_im_path + '%s_label.mat' % data_split)['feat']
        self.pd_im = pd.DataFrame(loadmat(dset_im_path + '%s_label.mat' % data_split)['feat'])



        if data_split == "train":
            self.image_clslist = np.load(dset_im_path + '%s_cls.npy' % data_split, allow_pickle=True,mmap_mode='r+')
            print('data_m 206:  ',self.image_clslist.shape)
            self.image_clslist = np.argmax(self.image_clslist, axis=-1)

        # tag string
        self.class2str = []
        if "flickr" in dset_text:
            with open(dset_tx_path + "common_tags.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    self.class2str.append(str(line.strip().split()[0]).lower())
            self.semantics = np.load(dset_tx_path + '%s_semantic.npy' % "flickr")
        elif "nus" in dset_text:
            with open(dset_tx_path + "TagList1k.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    self.class2str.append(str(line.strip().split()[0]).lower())
            self.semantics = np.load(dset_tx_path + '%s_semantic.npy' % "nus")

        self.length = len(self.captions)
        self.im_div = 1

        self.vocab = vocab
        self.num_class = self.labels_tx.shape[-1]
        self.data_split = data_split
        self.trainratio = trainratio
        print("image:", self.image_feature.shape)
        print("text:", self.length)
        print("text:", self.captions.shape)

    def __getitem__(self, index):
        #print('data_m.py1:  image.shape,image_index', self.data_split)
        if self.data_split == "test" or self.data_split == "dev" or self.data_split == "train":
            cap_index = index

        image = torch.Tensor(self.image_feature[index])
        loaction = torch.Tensor(self.box[0])
        hw = torch.Tensor(self.hw[0])
        box = np.zeros((loaction.shape[0], 5), dtype=np.float32)
        box[:, :4] = loaction
        box[:, 4] = (
                (box[:, 3] - box[:, 1])
                * (box[:, 2] - box[:, 0])
                / (float(hw[0]) * float(hw[1]))
        )
        box[:, 0] = box[:, 0] / float(hw[0])
        box[:, 1] = box[:, 1] / float(hw[1])
        box[:, 2] = box[:, 2] / float(hw[0])
        box[:, 3] = box[:, 3] / float(hw[1])
        box = torch.Tensor(box)
        label_im = torch.Tensor(self.labels_im[index])
        caption = self.captions[cap_index]
        vocab = self.vocab
        tokens = [self.class2str[i].lower() for i in np.where(caption == 1)[0]]
        caption, masklist_tx = make_caption(vocab, tokens, maskNeed=(self.data_split == "train"))

        if self.data_split == "train":
            imcls = self.image_clslist[index]
            masklist_im = mask_image(imcls)

        caption = torch.Tensor(caption)
        label_tx = torch.Tensor(self.labels_tx[cap_index])
        if self.data_split == "train":
            return image, caption, index, label_im, label_tx, masklist_im, masklist_tx, box
        else:
            return image, caption, index, label_im, label_tx, None, None, box

    def __len__(self):
        if self.data_split == "train":
            return int(self.length * min(self.trainratio, 1.0))
        else:
            return self.length


def make_caption(vocab, tokens, maskNeed=False):
    """
    for add mask
    :param vocab:
    :param tokens: list of words
    :return:
        - caption: list of indexes of word
        - masklist_tx: list of masked words, include(index in sentences, index in vocab).
            if there is no word masked, return [].
    """
    caption = []
    caption.append(vocab('<start>'))
    masklist_tx = []
    # caption mask
    for token in tokens:
        probb = random.random()
        if probb >= 0 and probb < 0.15 and maskNeed:
            caption.append(vocab('<mask>'))
            masklist_tx.append([len(caption) - 1, vocab(token)])
        else:
            caption.append(vocab(token))
    caption.append(vocab('<end>'))
    #print('data_m  305:  ', caption)

    return caption, masklist_tx


def mask_image(cls):
    # cls: 36* 1000
    clsInds = cls.tolist()
    masklist_im = []
    # image mask
    for token in range(len(clsInds)):
        probb = random.random()
        if probb >= 0 and probb < 0.15:
            masklist_im.append([token, clsInds[token]])

    return masklist_im


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, label_im, label_tx, masklist_im, masklist_tx, box = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    box = torch.stack(box, 0)
    labels_im = torch.stack(label_im, 0)
    labels_tx = torch.stack(label_tx, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    # ids为index是文本的行索引, img_ids没什么用，只用来读取图片和标签

    if masklist_im is None or (len(masklist_im) >= 1 and masklist_im[0] is None):
        return images, targets, lengths, ids, labels_im, labels_tx, box
    else:
        return images, targets, lengths, ids, labels_im, labels_tx, masklist_im, masklist_tx, box


def get_precomp_loader(data_path, dset_image, dset_text, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """
        :return 使用precomp的数据集和读取器
    """
    data_name = opt.data_name
    if "coco" in data_name:
        dset = PrecompCOCODataset(data_path, data_split, vocab, trainratio=opt.trainratio)
    else:
        dset = PrecompDataset(data_path, dset_image, dset_text, data_split, vocab, trainratio=opt.trainratio)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


# def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
def get_loaders(dset_image, dset_text, dset_val, vocab, batch_size, workers, opt):
    dpath = opt.data_path
    # if '_precomp' in dset_image and '_precomp' in dset_text:
    train_loader = get_precomp_loader(dpath, dset_image, dset_text, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, dset_val, dset_val, 'dev', vocab, opt,
                                    batch_size, False, workers)

    return train_loader, val_loader


# def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = opt.data_path

    # if '_precomp' in data_name:
    test_loader = get_precomp_loader(dpath, data_name, data_name, split_name, vocab, opt,
                                     batch_size, False, workers)

    return test_loader
