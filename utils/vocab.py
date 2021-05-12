# -*- coding:UTF-8 -*-

import nltk
from collections import Counter
from pycocotools.coco import COCO
import argparse
import os
import json

annotations = {
    'coco_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'coco': ['annotations/captions_train2014.json',
             'annotations/captions_val2014.json'],
    'flickr': ['common_tags.txt'],
    'nus': ['TagList1k.txt'],
    'flickr30k': ['dataset_flickr30k.json']
}


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def from_coco_json(path):
    coco = COCO(path)
    ids = coco.anns.keys()
    captions = []
    for i, idx in enumerate(ids):
        captions.append(str(coco.anns[idx]['caption']))

    return captions

def from_flickr_json(path):
    dataset = json.load(open(path, 'r'))['images']
    captions = []
    for i, d in enumerate(dataset):
        captions += [str(x['raw']) for x in d['sentences']]

    return captions

def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            captions.append(line.strip().split()[0])
    return captions

def serialize_vocab(vocab, dest):
    d = {}
    d['word2idx'] = vocab.word2idx
    d['idx2word'] = vocab.idx2word
    d['idx'] = vocab.idx

    with open(dest, "w") as f:
        json.dump(d, f)


def deserialize_vocab(src):
    """
    将序列化转为vocab对象
    :param src:存储的字典
    :return: vocab对象
    """
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    #print("d: ",d)
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab

def build_vocab(data_path, data_name, caption_file, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for path in caption_file[data_name]:
        full_path = os.path.join(os.path.join(data_path, data_name), path)
        if data_name == 'coco':
            captions = from_coco_json(full_path)
        elif data_name == 'flickr30k':
            captions = from_flickr_json(full_path)
        else:
            captions = from_txt(full_path)
        for i, caption in enumerate(captions):
            if 'coco' in data_name or 'flickr30k' in data_name:
                tokens = nltk.tokenize.word_tokenize(
                    caption.lower())
            else:
                tokens = nltk.tokenize.word_tokenize(
                    caption.lower().decode('utf-8'))
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    if data_name in ['flickr', 'nus']:
        words = [word for word, cnt in counter.items()]
    else:
        words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(data_path, data_name):
    # data_path: 文本
    # data_name: 名字
    if not os.path.isdir("./vocab"):
        os.mkdir("vocab")
    vocab = build_vocab(data_path, data_name, caption_file=annotations, threshold=4)
    serialize_vocab(vocab, '../vocab/%s_vocab.json' % data_name)
    print("Saved vocabulary file to ", '../vocab/%s_vocab.json' % data_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='data')
    parser.add_argument('--data_path', default='/data/yangy/xuyc/')
    parser.add_argument('--data_name', default='nus',
                        help='coco_precomp|coco|nus|flickr|flickr30k')
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name)
