# -*- coding:UTF-8 -*-
import os
import torch
import shutil

# author:Yc.Xu
# datetime:2020/7/15 23:39
# software: PyCharm

"""
Doc:

"""


# remember best R@ sum and save checkpoint

class BestStorer(object):

    def __init__(self, opt):
        self.model_dict = None
        self.loss = 10e6
        self.epoch = 0
        self.opt = opt
        self.Eiters = 0
        if not os.path.exists(opt.model_name):
            print(opt.model_name)
            os.mkdir(opt.model_name)

    def compare(self, loss, model, epoch):
        is_best = loss < self.loss
        if is_best:
            self.loss = loss
        self.model_dict = model.state_dict()
        self.epoch = epoch

        self.Eiters = model.Eiters
        print("保存当前模型 epoch:{}, best:{}, loss:{}, Eiters:{}".format(
            epoch, self.loss, loss, model.Eiters))
        self.save_checkpoint(is_best, loss)

    def save_checkpoint(self, is_best, loss):
        self.__save_checkpoint({
            'epoch': self.epoch,
            'model': self.model_dict,
            'best_loss': loss,
            'opt': self.opt,
            'Eiters': self.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(self.epoch), prefix=self.opt.model_name + '/')

    def __save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', prefix=''):
        tries = 15
        error = None

        # deal with unstable I/O. Usually not necessary.
        while tries:
            try:
                torch.save(state, prefix + filename)
                if is_best:
                    shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
            except IOError as e:
                error = e
                tries -= 1
            else:
                break
            print('model save {} failed, remaining {} trials'.format(filename, tries))
            if not tries:
                raise error

    def load_checkpoint(self, opt, model):
        if opt.resume:
            if os.path.isfile(opt.resume):
                print("=> loading checkpoint '{}'".format(opt.resume))
                checkpoint = torch.load(opt.resume)
                self.epoch = checkpoint['epoch'] + 1
                self.best_loss = checkpoint['best_loss']
                model.load_state_dict(checkpoint['model'])

                # Eiters is used to show logs as the continuation of another
                # training
                model.Eiters = checkpoint['Eiters']
                print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
                      .format(opt.resume, self.epoch, self.best_loss))
            else:
                print("=> no checkpoint found at '{}'".format(opt.resume))
