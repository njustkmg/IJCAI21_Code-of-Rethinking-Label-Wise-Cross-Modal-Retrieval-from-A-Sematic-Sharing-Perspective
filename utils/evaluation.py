from __future__ import print_function
import os
import sys

sys.path.append(r'../')
import pickle

from utils import data_m
import time
import numpy as np
from utils.vocab import Vocabulary, deserialize_vocab  # NOQA
import torch
# from model import
from collections import OrderedDict
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = data.get_test_loader(split, opt.data_name, vocab,
                                       opt.batch_size, opt.workers, opt)

    print('Computing results...')
    # img_embs, cap_embs = encode_data(model, data_loader)
    coco = ("coco" in opt.data_name)
    img_embs, cap_embs, cap_labels, cap_inputs = encode_data(model, data_loader, coco=coco)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if coco:
        if not fold5:
            # no cross-validation, full evaluation
            relmatrix = np.dot(cap_labels[0::5], cap_labels.T)  # ndcg relevance
            r, rt, ndcgi2t = i2t(img_embs, cap_embs, relmatrix, measure=opt.measure, return_ranks=True)
            relmatrix = relmatrix.T
            ri, rti, ndcgt2i = t2i(img_embs, cap_embs, relmatrix,
                                   measure=opt.measure, return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.2f" % rsum)
            print("Average i2t Recall: %.2f" % ar)
            print("Image to text: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % r)
            print("Average t2i Recall: %.2f" % ari)
            print("Text to image: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % ri)
            print("Average ndcg: %.2f ndcgi2t: %.2f ndcgt2i: %.2f" % (
                (ndcgi2t + ndcgt2i) / 2, ndcgi2t, ndcgt2i))

        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            ndcg_all = []
            for i in range(5):
                cap_labels_shared = cap_labels[i * 5000:(i + 1) * 5000]
                relmatrix = np.dot(cap_labels_shared[0::5], cap_labels_shared.T)
                r, rt0, ndcgi2t = i2t(img_embs[i * 5000:(i + 1) * 5000],
                                      cap_embs[i * 5000:(i + 1) * 5000],
                                      relmatrix, measure=opt.measure, return_ranks=True)
                print("Image to text: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % r, "ndcg: %.2f" % ndcgi2t)
                relmatrix = relmatrix.T
                ri, rti0, ndcgt2i = t2i(img_embs[i * 5000:(i + 1) * 5000],
                                        cap_embs[i * 5000:(i + 1) * 5000],
                                        relmatrix, measure=opt.measure, return_ranks=True)
                print("Text to image: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % ri, "ndcg: %.2f" % ndcgt2i)
                if i == 0:
                    rt, rti = rt0, rti0
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.2f ar: %.2f ari: %.2f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]
                ndcg_all.append([ndcgi2t, ndcgt2i])

            print("-----------------------------------")
            print("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            mean_ndcgs = tuple(np.array(ndcg_all).mean(axis=0).flatten())

            print("Image to text: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" %
                  mean_metrics[:8])
            print("Text to image: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" %
                  mean_metrics[8:16])
            print("Average ndcg: %.2f ndcgi2t: %.2f ndcgt2i: %.2f" % (
                (mean_ndcgs[0] + mean_ndcgs[1]) / 2, mean_ndcgs[0], mean_ndcgs[1]))

    else:
        _, unique_index, recon_index = np.unique(
            cap_inputs, return_index=True, return_inverse=True, axis=0)
        cap_embs = cap_embs[unique_index]
        img_labels = cap_labels.copy()
        cap_labels = cap_labels[unique_index]
        print('Images: %d, Captions: %d' %
              (img_embs.shape[0], cap_embs.shape[0]))

        relmatrix = np.dot(img_labels, cap_labels.T)
        r, rt, ndcgi2t = i2t_one(img_embs, cap_embs, relmatrix, unique_index, recon_index, measure=opt.measure,
                                 return_ranks=True, i2t=True)
        relmatrix = relmatrix.T
        ri, rti, ndcgt2i = i2t_one(cap_embs, img_embs, relmatrix, unique_index, recon_index, measure=opt.measure,
                                   return_ranks=True, i2t=False)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.2f" % rsum)
        print("Average i2t Recall: %.2f" % ar)
        print("Image to text: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % r)
        print("Average t2i Recall: %.2f" % ari)
        print("Text to image: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % ri)
        print("Average ndcg: %.2f ndcgi2t: %.2f ndcgt2i: %.2f" % (
            (ndcgi2t + ndcgt2i) / 2, ndcgi2t, ndcgt2i))

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t_one(images, captions, relmatrix, unique_index, recon_index, npts=None, measure='cosine', return_ranks=False,
            threshold=500, i2t=True):
    """
    Images->Text (Image Annotation)
    Images: (N, K) matrix of images
    Captions: (M, K) matrix of captions

    [[1,8,3,3,4],
     [1,8,3,3,4],
     [1,8,9,7,4],
     [1,8,9,9,4],]

    np.unique(data, return_index=True, return_inverse=True, axis=0)
    (array([[1, 8, 3, 3, 4],
            [1, 8, 9, 7, 4],
            [1, 8, 9, 9, 4]]),
    unique_index: array([0, 2, 3], dtype=int64), 原-》新
    recon_index: array([0, 0, 1, 2], dtype=int64)) 新-》原
    i2t_one(img_embs, cap_embs, relmatrix, unique_index, recon_index, measure=opt.measure,
                                 return_ranks=True, i2t=True)
    """
    if npts is None:
        npts = images.shape[0]  # captions size // 5
    #print('evalution.py  292', images.shape, captions.shape, type(npts), npts)
    index_list = []
    # if threshold > captions.shape[0]:
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    ndcgs = np.zeros(npts)
    random_ndcg = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            # d = np.dot(im, captions.T).flatten() #TODO 改为cos NDCG改为2000
            d = (np.dot(im, captions.T) / (np.linalg.norm(im, axis=1) * np.linalg.norm(captions, axis=1))
                 ).flatten()
        # get sims: d
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])
        # For NUS and FLICKR
        if i2t:
            rag = [recon_index[index]]
        else:
            # Text to image，会出现多个对应的image； unique_index[index] 给的是在原列表中的位置
            rag = np.where(recon_index == recon_index[unique_index[index]])[0].tolist()
        # Score
        rank = 1e20
        for i in rag:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        # tmp = numpy.where(inds == unique_index[index])[0][0]
        # if tmp < rank:
        #     rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        # compute NDCG
        inds_500 = inds[0:threshold]  # 取前500个结果的索引
        rel_500 = relmatrix[index][inds_500]  # 取前500个结果的相关度
        rel_order_500 = np.sort(relmatrix[index])[::-1][0:threshold]  # 所有结果排序后的相关度前500个
        relmatrix_pp = relmatrix[index].copy()
        np.random.shuffle(relmatrix_pp)
        rel_shuffle_500 = relmatrix_pp[0:threshold]  # 随机的NDCG为0.38，LCFS检索的NDCG为0.58, flickr来说
        # NDCG@K,K越大，值越高，K=2000随机的NDCG为0.82，LCFS检索的NDCG为0.88
        dcg = 0.0
        idcg = 0.0
        shuffledcg = 0.0
        for ind_t in range(threshold):
            dcg += rel_500[ind_t] / np.log2(ind_t + 2)
            idcg += rel_order_500[ind_t] / np.log2(ind_t + 2)
            shuffledcg += rel_shuffle_500[ind_t] / np.log2(ind_t + 2)
        if dcg > 0:
            ndcgs[index] = dcg / idcg
            random_ndcg[index] = shuffledcg / idcg
        # print(ndcgs[index])
        # print("-------------------------------------")
    '''print(ndcgs.min(), ndcgs.max(), ndcgs.mean(), np.median(ndcgs))
    print(random_ndcg.min(), random_ndcg.max(), random_ndcg.mean(), np.median(random_ndcg))'''
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    ndcgs = ndcgs.mean() * 100
    if return_ranks:
        # return (r1, r5, r10, medr, meanr), (ranks, top1)
        return (r1, r5, r10, r20, r50, r100, medr, meanr), (ranks, top1), ndcgs
    else:
        # return (r1, r5, r10, medr, meanr)
        return (r1, r5, r10, r20, r50, r100, medr, meanr), ndcgs


def i2t(images, captions, relmatrix, npts=None, measure='cosine', return_ranks=False, threshold=500):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    images (5000,80) captions (5000,80)
    """
    if npts is None:
        npts = images.shape[0] // 5  # captions size // 5
    index_list = []
    #print('evalution.py  386', images.shape, captions.shape,type(npts),npts)

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    ndcgs = np.zeros(npts)
    #print('evalution.py  392:' ,npts,images.shape[0])
    for index in range(npts):
        # Get query image
        #print(index)
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
            '''d = []
            for i in range(len(captions)):
                d.append((torch.matmul(torch.from_numpy(im), torch.transpose(torch.from_numpy(captions[i]), 0, 1))).max(dim=-1)[0].sum())
            d = np.array(d)'''
        # get sims: d
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        '''if index == 500:
            print("evalution 421 : ", index, rank)
            print(d[inds[0]])
            print(d[inds[rank]])
            print(d[inds[:300]])'''
        top1[index] = inds[0]
        #print("evalution.py  419:  ",index,npts)

        # compute NDCG
        inds_500 = inds[0:threshold]  # 取前500个结果的索引
        if not np.all(relmatrix == 0):
            rel_500 = relmatrix[index][inds_500]  # 取前500个结果的相关度
            rel_order_500 = np.sort(relmatrix[index])[::-1][0:threshold]  # 所有结果排序后的相关度前500个
            dcg = 0.0
            idcg = 0.0
            for ind_t in range(threshold):
                dcg += rel_500[ind_t] / np.log2(ind_t + 2)
                idcg += rel_order_500[ind_t] / np.log2(ind_t + 2)
            if dcg > 0:
                ndcgs[index] = dcg / idcg
        # else:
        #     print("Warning",dcg, idcg)
    # Compute metrics
    print("evalution.py  435:  ", len(np.where(ranks < 1)[0]), len(ranks),npts)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    ndcgs = ndcgs.mean() * 100
    if return_ranks:
        # return (r1, r5, r10, medr, meanr), (ranks, top1)
        return (r1, r5, r10, r20, r50, r100, medr, meanr), (ranks, top1), ndcgs
    else:
        # return (r1, r5, r10, medr, meanr)
        return (r1, r5, r10, r20, r50, r100, medr, meanr), ndcgs


def t2i(images, captions, relmatrix, npts=None, measure='cosine', return_ranks=False, threshold=500):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = np.array([images[i] for i in range(0, len(images), 5)])
    #print('evalution.py  465', images.shape, captions.shape, type(npts), npts)

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    ndcgs = np.zeros(5 * npts)

    for index in range(npts):  # npts 里层为text遍历

        # Get query captions
        queries = captions[5 * index:5 * index + 5]
        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)  # 5text * N images
        inds = np.zeros(d.shape, dtype=np.int)
        #print('evalution 489; ',d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]
            '''if index == 500:
                print("evalution 502 : ", index, np.where(inds[i] == index)[0][0])
                print(d[i][inds[i][0]])
                print(d[i][np.where(inds[i] == index)[0][0]])
                print(inds[i][:500])'''

            # compute NDCG
            inds_500 = inds[i][0:threshold]  # 取前500个结果的索引
            if not np.all(relmatrix == 0):
                rel_500 = relmatrix[5 * index + i][inds_500]  # 取前500个结果的相关度
                rel_order_500 = np.sort(relmatrix[5 * index + i])[::-1][0:threshold]  # 所有结果排序后的相关度前500个
                dcg = 0.0
                idcg = 0.0
                for ind_t in range(threshold):
                    #print('evalution 502:',ind_t,len(rel_500),threshold,len(inds_500),inds.shape)
                    dcg += rel_500[ind_t] / np.log2(5 * ind_t + i + 2)
                    idcg += rel_order_500[ind_t] / np.log2(5 * ind_t + i + 2)
                if dcg > 0:
                    ndcgs[5 * index + i] = dcg / idcg
            # else:
        #print(index, inds[0][0], inds[0][1], inds[0][2], inds[0][3], inds[0][4])

            #     print("Warning", dcg, idcg)

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    ndcgs = ndcgs.mean() * 100
    if return_ranks:
        # return (r1, r5, r10, medr, meanr), (ranks, top1)
        return (r1, r5, r10, r20, r50, r100, medr, meanr), (ranks, top1), ndcgs
    else:
        # return (r1, r5, r10, medr, meanr)
        return (r1, r5, r10, r20, r50, r100, medr, meanr), ndcgs


def validate(img_embs, cap_embs, cap_labels, cap_inputs, opt, fold5=False):
    coco = ("coco"if "coco" in opt.data_name or opt.data_name == "flickr30k" else False)
    print(coco,fold5)
    if coco :
        if fold5:
        #if not fold5:
            # no cross-validation, full evaluation
            relmatrix = np.dot(cap_labels[0::5], cap_labels.T)  # ndcg relevance
            print("evalution.py  534:  ")
            r, rt, ndcgi2t = i2t(img_embs, cap_embs, relmatrix, measure=opt.measure, return_ranks=True)
            relmatrix = relmatrix.T
            ri, rti, ndcgt2i = t2i(img_embs, cap_embs, relmatrix,
                                   measure=opt.measure, return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.2f" % rsum)
            print("Average i2t Recall: %.2f" % ar)
            print("Image to text: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % r)
            print("Average t2i Recall: %.2f" % ari)
            print("Text to image: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % ri)
            print("Average ndcg: %.2f ndcgi2t: %.2f ndcgt2i: %.2f" % (
                (ndcgi2t + ndcgt2i) / 2, ndcgi2t, ndcgt2i))

            currscore = rsum
        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            ndcg_all = []
            print("eva 606 : ",img_embs.shape,cap_embs.shape)
            for i in range(5):
                cap_labels_shared = cap_labels[i * 5000:(i + 1) * 5000]
                relmatrix = np.dot(cap_labels_shared[0::5], cap_labels_shared.T)
                r, rt0, ndcgi2t = i2t(img_embs[i * 5000:(i + 1) * 5000],
                                      cap_embs[i * 5000:(i + 1) * 5000],
                                      relmatrix, measure=opt.measure, return_ranks=True)
                print("Image to text: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % r, "ndcg: %.2f" % ndcgi2t)
                relmatrix = relmatrix.T
                ri, rti0, ndcgt2i = t2i(img_embs[i * 5000:(i + 1) * 5000],
                                        cap_embs[i * 5000:(i + 1) * 5000],
                                        relmatrix, measure=opt.measure, return_ranks=True)
                print("Text to image: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % ri, "ndcg: %.2f" % ndcgt2i)
                if i == 0:
                    rt, rti = rt0, rti0
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.2f ar: %.2f ari: %.2f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]
                ndcg_all.append([ndcgi2t, ndcgt2i])

            print("-----------------------------------")
            print("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            mean_ndcgs = tuple(np.array(ndcg_all).mean(axis=0).flatten())

            print("Image to text: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" %
                  mean_metrics[:8])
            print("Text to image: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" %
                  mean_metrics[8:16])
            print("Average ndcg: %.2f ndcgi2t: %.2f ndcgt2i: %.2f" % (
                (mean_ndcgs[0] + mean_ndcgs[1]) / 2, mean_ndcgs[0], mean_ndcgs[1]))

            currscore = mean_metrics[-1]

    else:
        _, unique_index, recon_index = np.unique(
            cap_inputs, return_index=True, return_inverse=True, axis=0)
        cap_embs = cap_embs[unique_index]
        img_labels = cap_labels.copy()
        cap_labels = cap_labels[unique_index]
        print('Images: %d, Captions: %d' %(img_embs.shape[0], cap_embs.shape[0]))

        relmatrix = np.dot(img_labels, cap_labels.T)
        #print('evaluation.py    ',img_embs, cap_embs, relmatrix, unique_index, recon_index, opt.measure )
        r, rt, ndcgi2t = i2t_one(img_embs, cap_embs, relmatrix, unique_index, recon_index, measure=opt.measure,
                                 return_ranks=True, i2t=True)
        relmatrix = relmatrix.T
        ri, rti, ndcgt2i = i2t_one(cap_embs, img_embs, relmatrix, unique_index, recon_index, measure=opt.measure,
                                   return_ranks=True, i2t=False)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.2f" % rsum)
        print("Average i2t Recall: %.2f" % ar)
        print("Image to text: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % r)
        print("Average t2i Recall: %.2f" % ari)
        print("Text to image: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % ri)
        print("Average ndcg: %.2f ndcgi2t: %.2f ndcgt2i: %.2f" % (
            (ndcgi2t + ndcgt2i) / 2, ndcgi2t, ndcgt2i))

        currscore = rsum

    return currscore, (ndcgi2t,ndcgt2i)
