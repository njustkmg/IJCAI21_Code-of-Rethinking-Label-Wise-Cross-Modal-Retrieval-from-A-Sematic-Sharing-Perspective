# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Training script"""

import os
import pickle
import time
import shutil

import torch
import numpy as np

import utils.data_m as data  # 原始
from utils.vocab import Vocabulary, deserialize_vocab
from model.Model import Model
from utils.evaluation import i2t, t2i, AverageMeter, LogCollector, i2t_one
from utils.evaluation import validate
from torch.autograd import Variable
from utils.tools import BestStorer
import logging
import tensorboard_logger as tb_logger
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score

import argparse


def main(parser=None):
    if parser is None:
        # Hyper Parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--ds', default="",
                            help='描述')
        parser.add_argument('--gpu', default='0', type=str)
        parser.add_argument('--num', default=0, type=int)
        parser.add_argument('--data_path', default='./data/',
                            help='path to datasets')
        parser.add_argument('--data_name', default='flickr_precomp_12',
                            help='{coco,f30k,nus}_precomp')
        parser.add_argument('--vocab_path', default='./vocab',
                            help='Path to saved vocabulary json files.')
        parser.add_argument('--model_name', default='./data/runs/',
                            help='Path to save the model.')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--margin', default=0.2, type=float,
                            help='Rank loss margin.')
        parser.add_argument('--num_epochs', default=35, type=int,
                            help='Number of training epochs.')
        #parser.add_argument('--batch_size', default=128, type=int,
        parser.add_argument('--batch_size', default=64, type=int,
                            help='Size of a training mini-batch.')
        parser.add_argument('--word_dim', default=300, type=int,
                            help='Dimensionality of the word embedding.')
        parser.add_argument('--embed_size', default=1024, type=int,  # 128
                            help='Dimensionality of the joint embedding.')
        parser.add_argument('--grad_clip', default=2., type=float,
                            help='Gradient clipping threshold.')
        parser.add_argument('--num_layers', default=1, type=int,
                            help='Number of GRU layers.')
        # parser.add_argument('--learning_rate', default=.0002, type=float,
        parser.add_argument('--learning_rate', default=.0002, type=float,
                            help='Initial learning rate.')
        parser.add_argument('--lr_update', default=20, type=int,
                            help='Number of epochs to update the learning rate.')
        parser.add_argument('--workers', default=10, type=int,
                            help='Number of data loader workers.')
        parser.add_argument('--log_step', default=5, type=int,
                            help='Number of steps to print and record the log.')
        parser.add_argument('--val_step', default=10, type=int,
                            help='Number of steps to run validation.')
        # parser.add_argument('--max_violation', action='store_true',
        parser.add_argument('--img_dim', default=2048, type=int,
                            help='Dimensionality of the image embedding.')
        parser.add_argument('--no_imgnorm', action='store_true',
                            help='Do not normalize the image embeddings.')
        parser.add_argument('--no_txtnorm', action='store_true',
                            help='Do not normalize the text embeddings.')
        parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                            help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
        parser.add_argument('--agg_func', default="LogSumExp",
                            help='LogSumExp|Mean|Max|Sum')
        parser.add_argument('--cross_attn', default="t2i",
                            help='t2i|i2t')
        parser.add_argument('--precomp_enc_type', default="basic",
                            help='basic|weight_norm')
        parser.add_argument('--bi_gru', default=True,
                            help='Use bidirectional GRU.')
        parser.add_argument('--lambda_lse', default=6., type=float,
                            help='LogSumExp temp.')
        parser.add_argument('--lambda_softmax', default=9., type=float,
                            help='Attention softmax temperature.')
        parser.add_argument('--use_restval', default=True,
                            help='Use the restval data for training on MSCOCO.')
        parser.add_argument('--crop_size', default=224, type=int,
                            help='Size of an image crop as the CNN input.')
        parser.add_argument('--measure', default='cosine',
                            help='Similarity measure used (cosine|order)')
        parser.add_argument('--warm_epoch', default=5)
        parser.add_argument('--trainratio', default=1.0, type=float,
                            help='control the ratio of train set')

    opt = parser.parse_args()
    opt.model_name = opt.model_name + opt.data_name
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    print('cuda: ',os.environ["CUDA_VISIBLE_DEVICES"])
    print(opt)
    assert opt.data_name == opt.model_name.split('/')[-1], "%s 和 %s 需要对应" % \
                                                           (opt.data_name, opt.model_name)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    print('opt.model_name:   ',opt.model_name)
    tb_logger.configure(opt.model_name, flush_secs=5)

    vocab_name = "coco_precomp" if "coco" in  opt.data_name else opt.data_name.split("_precomp")[0]
    print('vocab_name:   ', vocab_name)
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, '%s_vocab.json' % vocab_name))  # coco_precomp
    opt.vocab_size = len(vocab)

    data_name = opt.data_name.split("_precomp")[0]
    print('data_name:   ', data_name)


    train_loader, val_loader = data.get_loaders(
        data_name, data_name, data_name, vocab, opt.batch_size, opt.workers, opt)
    test_loader = data.get_test_loader('test', opt.data_name, vocab,
                                       opt.batch_size, opt.workers, opt)
    class_num = train_loader.dataset.num_class

    model = Model(opt, class_num, opt.vocab_size, 1000)
    beststorer = BestStorer(opt)
    # optionally resume from a checkpoint
    beststorer.load_checkpoint(opt, model)
    start_epoch = beststorer.epoch



    # Train the Model
    for epoch in range(start_epoch, start_epoch + opt.num_epochs):
        # adjust_learning_rate(opt, model.optimizer, model.optimizer_share, opt.lr_update)
        # train for one epoch
        print("eopch : ",epoch)
        train(opt, train_loader, model, epoch, val_loader,test_loader, beststorer)
        rsum,ndcg= encodewithvalidate(opt, val_loader, model)

def train(opt, train_loader, model, epoch, val_loader,test_loader, beststorer):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()

    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        loss = model.train_emb(*train_data)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))
            # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            beststorer.compare(loss, model, epoch)


def encodewithvalidate(opt, val_loader, model, fold5=True):
    coco = (opt.data_name == "coco")
    model.val_start()
    # compute the encoding for all the validation images and captions
    img_embs, img_labels, cap_embs, cap_lens, cap_labels, cap_inputs = encode_data(
        model, val_loader, opt.log_step, logging.info, coco=coco)

    if len(img_embs.shape) == 3:
        img_embs = np.mean(img_embs, axis=1)
    if len(cap_embs.shape) == 3:
        cap_embs = np.mean(cap_embs, axis=1)
    rsum, ndcg = validate(img_embs, cap_embs, cap_labels, cap_inputs, opt, fold5=fold5)
    return rsum,ndcg


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
        check
    """
    # norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    # X = torch.div(X, norm)
    norm = np.sqrt(np.power(X, 2).sum(axis=dim, keepdims=True)) + eps
    X = np.divide(X, norm)

    return X


def adjust_learning_rate(opt, optimizer, optimizer_share, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.9 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if optimizer_share is not None:
        for param_group in optimizer_share.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def encode_data(model, data_loader, log_step=10, logging=print, coco=True):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None
    cap_inputs = None
    img_labels = None
    cap_labels = None

    max_n_word = 0
    for i, (images, captions, lengths, ids, label_im, label_tx, box) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, ids, label_im, label_tx, box) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths, box)
        if img_embs is None:
            if img_emb.dim() == 3:  # 包含36个feature的情况
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:  # 仅包含一个feature的情况
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            img_labels = np.zeros((len(data_loader.dataset), label_im.size(1)))
            cap_labels = np.zeros((len(data_loader.dataset), label_tx.size(1)))
            cap_lens = [0] * len(data_loader.dataset)
            if coco is False:
                # 手工设置记录caption,flickr最大57, nus最大131
                cap_inputs = np.zeros((len(data_loader.dataset), 150))

        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        img_labels[ids] = label_im.data.cpu().numpy().copy()
        cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()
        cap_labels[ids] = label_tx.data.cpu().numpy().copy()
        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]
            if cap_lens == 0:
                print(cap_lens)
        if coco is False:
            max_len = captions.size(1)
            assert max_len <= 150, "Cap_inputs is %d more than 150" % max_len
            cap_inputs[ids, : max_len] = captions.data.cpu().numpy().copy()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        del images, captions

    return img_embs, img_labels, cap_embs, cap_lens, cap_labels, cap_inputs


def evalrank(model_path, data_path=None, split='dev', fold5=True):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print(opt)

    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab_name = "coco_precomp" if "coco" in opt.data_name else opt.data_name.split("_precomp")[0]
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, '%s_vocab.json' % vocab_name))  # coco_precomp
    opt.vocab_size = len(vocab)


    print('Loading dataset')
    print(vocab_name,split, opt.data_name, vocab,opt.workers)
    data_loader = data.get_test_loader(split, opt.data_name, vocab,
                                       opt.batch_size, opt.workers, opt)

    class_num = data_loader.dataset.num_class
    # Construct the model
    model = Model(opt, class_num, opt.vocab_size, 1000)
    print(checkpoint['opt'])
    print("Best epoch:", checkpoint['epoch'],'\n',len(checkpoint['model'][1]))

    # load model state
    model.load_state_dict(checkpoint['model'])

    encodewithvalidate(opt, data_loader, model, fold5=fold5)


if __name__ == '__main__':
    main()
    data_path = './data/'
    evalrank('./data/runs/flickr_precomp_12/model_best.pth.tar', data_path=data_path,split="test", fold5=True)
