# -----------------------------------------------------------
# Non-parallel Cross-modal Retrieval
#
# Writen by Yichu-Xu, 2020
# ---------------------------------------------------------------
import logging
import random
import time

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from model.transformer import make_model
#from test import Test
from sklearn.metrics import recall_score,precision_score,f1_score
import math


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
        check
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False, finetune=False, cnn_type='vgg19', use_abs=False, train=True):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if data_name not in ['flickr', 'nus'] and 'pre' not in data_name:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm, train=train)
    else:
        if precomp_enc_type == 'basic':
            img_enc = EncoderImagePrecomp(
                img_dim, embed_size, no_imgnorm)
            print("here")
        else:
            raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False, train=True):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        # 若为测试，不需要加载模型
        self.cnn = self.get_cnn(cnn_type, pretrained=train, local=False)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained, local=False):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            if not local:
                model = models.__dict__[arch](pretrained=True)
            else:
                if arch == 'resnet101':
                    path = "/data/yangy/xuyc/models/resnet101-5d3b4d8f.pth"
                elif arch == 'vgg19':
                    path = "/data/yangy/xuyc/models/vgg19-dcbb9e9d.pth"
                print("Load From local:", path)
                model = models.__dict__[arch]()
                pretrain_dict = torch.load(path)
                model.load_state_dict(pretrain_dict)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=1)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)
        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.relu = nn.ReLU()
        self.w = 0.0001
        self.pos_dim = 5
        self.box_fc = nn.Linear(self.pos_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images,box):
        """Extract image feature vectors."""

        features = self.fc(images)
        position_embedding = self.box_fc(box)
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)
            position_embedding = l2norm(position_embedding, dim=-1)
        features = (features+self.w*position_embedding)/2
        # features = self.relu(features)
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.position_embeddings = nn.Embedding(512, embed_size, padding_idx=0)
        self.w = 0.00001

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.position_embeddings.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2
        position_ids = torch.tensor(np.zeros([cap_emb.size(0), cap_emb.size(1)])).long().to(cap_emb.device)
        position_embeddings = self.position_embeddings(position_ids)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
            position_embeddings = l2norm(position_embeddings, dim=-1)
        cap_emb = (cap_emb + self.w * position_embeddings) / 2
        return cap_emb, cap_len


class BackNet(nn.Module):
    """
        After transformer encoder.
        include predict layer, MOC_predict layer, MLM_predict_layer
    """

    def __init__(self, input_size, output_size, vocab_size, object_size, no_txtnorm=False):
        super(BackNet, self).__init__()
        self.no_txtnorm = no_txtnorm

        self.input_size = input_size
        self.output_size = output_size
        # word embedding
        self.fc_MLM = nn.Linear(input_size, vocab_size)
        self.fc_MOC = nn.Linear(input_size, object_size)
        self.fc_predict = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, embedd, masklist, modal="text", train=False):
        """Handles variable size captions
        """

        batch_num = embedd.size(0)
        region_num = embedd.size(1)
        result = embedd.contiguous().view(batch_num * region_num, self.input_size)
        result = self.fc_predict(result)
        # TODO softmax is helpful to label-specific?
        result = nn.Sigmoid()(result)
        result = result.view(batch_num, region_num, -1)
        result = l2norm(result, dim=-1)
        if not train:
            return result, None, None
        else:
            mask_tensor = []
            cls_list = []
            for f in range(np.min([len(masklist),embedd.size(1)])):
                for ff in masklist[f]:
                    ff_ind = ff[0]
                    ff_cls = ff[1]
                    if ff_ind >= embedd.size(1):
                        continue
                    mask_tensor.append(embedd[f, ff_ind])
                    cls_list.append(ff_cls)
            mask_tensor = torch.stack(mask_tensor, dim=0)

            if modal == "text":
                ret_task = nn.Sigmoid()(self.fc_MLM(mask_tensor))
            else:
                ret_task = nn.Sigmoid()(self.fc_MOC(mask_tensor))

            return result, ret_task, cls_list


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class MultiLabelLoss(nn.Module):
    """
    Compute Multi-label loss
    """

    def __init__(self, opt, margin=0):
        super(MultiLabelLoss, self).__init__()
        self.opt = opt
        self.losstype = opt.data_name.split("_")[-1]
        self.beta_mask = 0.0005  # for mask task
        self.beta_match = 1  # for matching task
        self.margin = 1

        if self.losstype == '13':
            self.beta_mask = 1
        # self.beta_rank = 0.45 # for loss ranking
        self.threshold = 5 # 运行过3,4,5，5最优

        print("Params({}): beta_mask: {}, beta_match: {}".format(
            self.losstype,self.beta_mask, self.beta_match))

    def forward(self, im_pred, tx_pred, labels_im, labels_tx,
                img_mask_emb, img_mask_label, cap_mask_emb, cap_mask_label):
        """
            classify loss
            loss_cls:0.3852,
            loss_match:0.9996,
            loss_mv 6.90 loss_mw:7.28
        :param y_pred: N * R * class_num
        :param y_true: N * class_num
        :return: 
        """
        assert len(im_pred.size()) == 3, "预测为N * region * class_num"

        # BATCH * REGION * CLASS
        if len(im_pred.size()) == 3:
            im_pred = im_pred.mean(dim=1)
        if len(tx_pred.size()) == 3:
            tx_pred = tx_pred.mean(dim=1)

        y_pred = torch.cat([im_pred, tx_pred], dim=0)
        y_true = torch.cat([labels_im, labels_tx], dim=0)

        loss_sum = torch.zeros([1]).cuda()

        if "1" in self.losstype:
            # # 自然对数
            loss_cls = torch.log(1 + torch.exp(- torch.mul(y_pred, y_true.float()).sum(dim=-1))).sum() / y_pred.size(0)
            loss_sum += loss_cls

        if "2" in self.losstype:
            img_mask_label = torch.Tensor(img_mask_label).long().to(im_pred.get_device())
            loss_mv = nn.CrossEntropyLoss()(img_mask_emb, img_mask_label)
            cap_mask_label = torch.Tensor(cap_mask_label).long().to(im_pred.get_device())
            loss_mw = nn.CrossEntropyLoss()(cap_mask_emb, cap_mask_label)

            loss_sum += self.beta_mask * (loss_mv + loss_mw)

        return loss_sum



class Model(object):
    """
        Non-Parallel Cross-Modal Retrieval Considering Reusable Model
    """

    def __init__(self, opt, class_num, vocab_num, object_num):
        # Build Models
        self.grad_clip = opt.grad_clip

        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=True)
        if torch.cuda.is_available():
            self.img_enc.cuda()  # 启用cuda
            print('cuda is')
        else:
            print('cuda is not')
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=True)
        # self.transformer = make_model(N=6, d_model=opt.embed_size, d_ff=1024, h=8, out_size=class_num)
        self.transformer = make_model(N=6, d_model=opt.embed_size, d_ff=2048, h=8, out_size=class_num)

        # Loss and Optimizer
        self.criterion = MultiLabelLoss(opt=opt, margin=opt.margin)

        self.backnet = BackNet(opt.embed_size, class_num, vocab_num, object_num)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.transformer.cuda()
            self.backnet.cuda()
            self.criterion.cuda()
            cudnn.benchmark = True

        self.params = list(self.txt_enc.parameters())
        self.params += list(self.img_enc.fc.parameters())
        self.params += list(self.transformer.parameters())
        self.params += list(self.backnet.parameters())

        self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        # self.optimizer_share = None

        self.Eiters = 0


    def change_transformer(self, mode=False):
        for param in self.transformer.parameters():
            param.requires_grad = mode

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.transformer.state_dict(),
                      self.backnet.state_dict()
                      ]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.transformer.load_state_dict(state_dict[2])
        self.backnet.load_state_dict(state_dict[3])
        self.img_enc.cuda()
        self.txt_enc.cuda()
        self.transformer.cuda()
        self.backnet.cuda()

    def train_start(self):
        """switch to train mode
        """
        self.txt_enc.train()
        self.img_enc.train()
        self.transformer.train()
        self.backnet.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.transformer.eval()
        self.backnet.eval()

    def forward_emb_text(self, captions, lengths):
        """Compute the caption embeddings
        """
        # Set mini-batch dataset
        # captions = Variable(captions, volatile=volatile)

        # Forward
        # cap_emb (tensor), cap_lens (list)
        if torch.cuda.is_available():
            captions = captions.cuda()
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return cap_emb, cap_lens

    def forward_emb_image(self, images,box):
        """Compute the image embeddings
        """
        # Set mini-batch dataset
        # images = Variable(images, volatile=volatile)

        # Forward
        if torch.cuda.is_available():
            images = images.cuda()
            box = box.cuda()
        img_emb = self.img_enc(images,box)
        return img_emb

    def forward_loss(self, im_pred, tx_pred, labels_im, labels_tx,
                     img_mask_emb, img_mask_label, cap_mask_emb, cap_mask_label,  **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(im_pred, tx_pred, labels_im, labels_tx,
                              img_mask_emb, img_mask_label, cap_mask_emb, cap_mask_label)
        self.logger.update('Le', loss.item(), im_pred.size(0))
        return loss

    # 联合预测
    def forward_emb(self, images, captions, lengths, box, masks_im=None, masks_tx=None):
        """Compute the image and caption embeddings
        """

        # Forward
        # cap_emb (tensor), cap_lens (list)
        if torch.cuda.is_available():
            images = images.cuda()
            box = box.cuda()
            captions = captions.cuda()
        cap_emb, cap_lens = self.forward_emb_text(captions, lengths)
        img_emb = self.forward_emb_image(images,box)
        cap_emb = self.transformer(cap_emb)
        img_emb = self.transformer(img_emb)
        img_emb, img_mask_emb, img_mask_label = self.backnet(img_emb, masks_im, "image", False)
        cap_emb, cap_mask_emb, cap_mask_label = self.backnet(cap_emb, masks_tx, "text", False)

        return img_emb, cap_emb, cap_lens

    def train_emb(self, images, captions, lengths, ids=None, labels_im=None, labels_tx=None,
                  masks_im=None, masks_tx=None, box=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        cap_emb, cap_lens = self.forward_emb_text(captions, lengths)
        img_emb = self.forward_emb_image(images,box)
        cap_emb = self.transformer(cap_emb)
        img_emb = self.transformer(img_emb)
        img_emb, img_mask_emb, img_mask_label = self.backnet(img_emb, masks_im, "image", True)
        cap_emb, cap_mask_emb, cap_mask_label = self.backnet(cap_emb, masks_tx, "text", True)
        if torch.cuda.is_available():
            labels_im = labels_im.cuda()
            labels_tx = labels_tx.cuda()

        self.optimizer.zero_grad()
        # if self.optimizer_share is not None:
        #     self.optimizer_share.zero_grad()

        loss = self.forward_loss(img_emb, cap_emb, labels_im, labels_tx,
                                 img_mask_emb, img_mask_label, cap_mask_emb, cap_mask_label)
        #loss.requires_grad_(True)
        # compute gradient and do SGD step
        loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm_(self.txt_enc.parameters(), self.grad_clip)
        self.optimizer.step()
        # if self.optimizer_share is not None:
        #     self.optimizer_share.step()
        return loss.item()
