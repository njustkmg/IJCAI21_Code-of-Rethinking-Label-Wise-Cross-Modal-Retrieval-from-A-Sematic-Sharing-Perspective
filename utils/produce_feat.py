import datetime
import pickle

import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
import time
import torch.optim as optim
import logging
import argparse
import os
import shutil

import joblib
from vocab import Vocabulary, deserialize_vocab
import data
import numpy as np
from sklearn.decomposition import PCA
import torch.nn as nn
import torchvision.models as Models
from resnets import resnet101
from scipy.io import savemat

######################################################################
# Start running

device = torch.device("cuda:12" if torch.cuda.is_available() else "cpu")


# class ResNet(nn.Module):
#     def __init__(self):
#         super(ResNet, self).__init__()
#         # self.feature = Models.resnet18(pretrained=True)
#         self.feature = Models.resnet50(pretrained=True).to(device)
#         for param in self.feature.parameters():
#             param.requires_grad = False
#         self.feature = nn.Sequential(*list(self.feature.children())[:-1])
#
#     def forward(self, x):
#         N = x.size()[0]
#         x = self.feature(x.view(N, 3, 256, 256))
#         x = x.view(N, 512)
#         return x


def main(parser=None):
    # Hyper Parameters
    if parser is None:
        parser = argparse.ArgumentParser()
        # parser.add_argument('--data_path', default='/data/yangy/xuyc/',
        parser.add_argument('--data_path', default='/home/administrator/PycharmProjects/workspace/RMDCRM/data/yangy/xuyc/',
                            help='path to datasets')
        parser.add_argument('--data_name', default='flickr30k',
                            help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k|FLICKR')
        parser.add_argument('--vocab_path', default='./vocab/',
                            help='Path to saved vocabulary pickle files.')
        parser.add_argument('--save_dir', default='./',
        # parser.add_argument('--save_dir', default='/data/yangy/xuyc/image_feats/',
                            help='Path to saved vocabulary pickle files.')
        parser.add_argument('--crop_size', default=256, type=int,  # resnet输入
                            help='Size of an image crop as the CNN input.')
        parser.add_argument('--workers', default=0, type=int,
                            help='Number of data loader workers.')
        parser.add_argument('--log_step', default=10, type=int,
                            help='Number of steps to print and record the log.')
        parser.add_argument('--val_step', default=500, type=int,
                            help='Number of steps to run validation.')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--use_restval', default=True,
                            help='Use the restval data for training on MSCOCO.')
        parser.add_argument('--measure', default='cosine',
                            help='Similarity measure used (cosine|order)')
        parser.add_argument('--cca_iter', default=2000)
        parser.add_argument('--n_comp', default=500)
        parser.add_argument('--train_num', default=150000)
        parser.add_argument('--pca_size', default=512)

    opt = parser.parse_args()
    print(opt)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    ###################hyper parameters setting##########################################3
    textnet_cfg = {
        'flickr': [1386, 24],  # tag维度，class数
        'nus': [1000, 10]
    }
    n_comp = opt.n_comp

    if opt.data_name == "coco" or "flickr30k":
        print('opt.vocab_path:  '+opt.vocab_path+'  '+opt.data_name)
        vocab = deserialize_vocab(
            os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))  # coco_precomp
        # vocab = pickle.load(open(os.path.join(
        #     opt.vocab_path, '%s_vocab.pkl' % opt.data_name.lower()), 'rb'))
        opt.vocab_size = len(vocab)
    else:
        vocab = None
    print('...Data loading is beginning...')

    # 读取Image Path 和 Text Onehot
    train_dataset = data.get_traintest_datasets(opt.data_name, opt, vocab=vocab, train=True)
    test_dataset = data.get_traintest_datasets(opt.data_name, opt, vocab=vocab, train=False)

    # train_num = max(opt.train_num, len(train_dataset))
    train_num = len(train_dataset)
    test_num = len(test_dataset)

    # resnet = resnet101(pretrained=True, root = "/data/yangy/xuyc/models/", ndfc=False).to(device)
    # resnet.eval()
    # for param in resnet.parameters():
    #     param.requires_grad = False
    image_train = []
    text_train = []
    for index in range(train_num):
        image, text, label = train_dataset.raw_im_te_lb(index)
        # image_train.append(resnet(image.unsqueeze(0).to(device)).cpu().numpy().reshape(-1))
        # image_train.append(np.array(image.resize((128, 128))).flatten())
        if index % 5 == 0:
            # raw_im_te_lb得到的都是图像和文本的特征
            image_train.append(image)
        text_train.append(text)

        if index % 4000 == 0:
            print(index, train_num)
    image_train = np.stack(image_train)
    text_train = np.stack(text_train)

    # savemat(opt.save_dir + "_tr_whole_feat.mat", {'image': image_train, 'text': text_train, 'label': label_train})

    print('...Data loading is completed...')
    print("image_train:", image_train.shape)
    # print("text_train:", text_train.shape)
    ###################### Model #################################
    pca_image = PCA(opt.pca_size)
    pca_text = PCA(opt.pca_size)
    image_train = pca_image.fit_transform(image_train)
    logging.info('pca image')
    text_train = pca_text.fit_transform(text_train)
    logging.info('pca text')
    savemat(opt.save_dir + opt.data_name + "_tr_feat.mat", {'image': image_train, 'text': text_train})

    #### test ####
    image_test = []
    text_test = []
    for index in range(test_num):
        image, text, label = test_dataset.raw_im_te_lb(index)
        # image_test.append(resnet(image.unsqueeze(0).to(device)).cpu().numpy().reshape(-1))
        if index % 5 == 0:
            image_test.append(image)
        text_test.append(text)

        if index % 4000 == 0:
            print(index, train_num)
    image_test = np.stack(image_test)
    text_test = np.stack(text_test)
    # savemat(opt.save_dir + opt.data_name + "_te_whole_feat.mat", {'image': image_test, 'text': text_test})
    img_embs = pca_image.transform(image_test)
    logging.info('pca image')
    cap_embs = pca_text.transform(text_test)
    logging.info('pca text')
    savemat(opt.save_dir + opt.data_name + "_te_feat.mat", {'image': img_embs, 'text': cap_embs})


def savepca(pca_text, data_name):
    print("Save CCA model")
    joblib.dump(pca_text, "model/%s_pca_text.m" % data_name)


def loadpca(data_name):
    pca_text = joblib.load("model/%s_pca_text.m" % data_name)


if __name__ == '__main__':
    main()
    # load_checkpoint_eval(prefix='./runs/cca/flickr/')
