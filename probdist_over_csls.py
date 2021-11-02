###############################################################################
#
# Edited from Mikel Artetxe's eval_translation.py and in VecMap:
#   https://github.com/artetxem/vecmap/blob/b82246f6c249633039f67fa6156e51d852bd73a3/eval_translation.py
#   -Kelly Marchisio, March 2021.
#
###############################################################################

# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from third_party.vecmap_fork import embeddings 
import argparse
from collections import defaultdict
import cupy as cp 
from scipy.special import softmax
import sys
import torch
from utils import utils


BATCH_SIZE = 500
TEMP=0.1


###############################################################################


def get_topk_translations(x, z, k, vocab_inds, vocab_list, neighborhood=10):
    # Get topk best translations per source word in vocab_inds
    translation = defaultdict(list)
    knn_sim_bwd = cp.zeros(z.shape[0])
    for i in range(0, z.shape[0], BATCH_SIZE):
        j = min(i + BATCH_SIZE, z.shape[0])
        knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=neighborhood, inplace=True)
    for i in range(0, len(vocab_inds), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(vocab_inds))
        # From Kelly: you can say this is equivalent b/c you're going to
        # finding the top values, not the raw CSLS score. So you'd be
        # subtracting the same thing from each column in the row anyway per
        # word, so you don't need to do it since you'll find argmax across row. 
        similarities = 2*x[vocab_list[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
        nn_vals, nn = torch.topk(torch.tensor(similarities), k, dim=1)
        for k in range(j-i):
            translation[vocab_list[i+k]] = nn[k]
    return translation


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    n = m.shape[0]
    ans = cp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = cp.array(m)
    ind0 = cp.arange(n)
    ind1 = cp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('--sents', metavar='PATH', help='Training sentences')
    parser.add_argument('--sep', default=None, help='default separator.')
    parser.add_argument('-i', default=sys.stdin.fileno(), help='words to translate') 
    parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
    parser.add_argument('-topk', default=100, type=int, 
        help='how many potential translations to consider')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    args = parser.parse_args()
    dtype='float32'

    # Read input embeddings
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, 200000, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, 200000, dtype=dtype)

    cp.cuda.Device(0).use()
    x = cp.asarray(x)
    z = cp.asarray(z)
    cp.random.seed(args.seed)

    embeddings.length_normalize(x)
    embeddings.length_normalize(z)

    # Build word to index map
    # From vecmap eval_translation.py (very similar to)
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    src_ind2word = {i: word for i, word in enumerate(src_words)}
    trg_ind2word = {i: word for i, word in enumerate(trg_words)}

    # Read dictionary and compute coverage
    f = open(args.i, encoding=args.encoding, errors='surrogateescape')
    oov = set()
    vocab = set()
    vocab_inds = set()
    for line in f:
        try:
            src_wd = line.strip()
            src_ind = src_word2ind[src_wd]
            vocab.add(src_wd)
            vocab_inds.add(src_ind)
        except KeyError:
            oov.add(src_wd)
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    vocab_ind_list = list(vocab_inds)

    # Find translations
    # Get topk best translations per source word
    # translation = get_topk_translations(x, z, 5, vocab_inds, vocab_ind_list)
    (word_counts, src2trg_pairs, trg2src_pairs, src2trg_pairs_inds,
            trg2src_pairs_inds) = utils.possible_word_pairs(
            open(args.sents, 'r', encoding='utf-8', errors='surrogateescape'),
            src_word2ind, trg_word2ind, args.sep)

    src2trg_pairs_inds_list = [item for item in src2trg_pairs_inds.items()]

    knn_sim_bwd = cp.zeros(z.shape[0])
    for i in range(0, z.shape[0], BATCH_SIZE):
        j = min(i + BATCH_SIZE, z.shape[0])
        knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)

    # For each word in the dictionary, 
    translation = defaultdict(int)
    for vocab_ind in vocab_inds: 
        possible_trg_words = list(src2trg_pairs_inds[vocab_ind])
        if possible_trg_words:
            similarities = 2*x[vocab_ind].dot(z.T) - knn_sim_bwd  # From Mikel: Equivalent to the real CSLS scores for NN
            similarities = cp.expand_dims(similarities, 0)
            similarities = cp.take(similarities[0], possible_trg_words)
            similarities = cp.expand_dims(similarities, 0)
            nn_vals, nn = torch.topk(torch.tensor(similarities), 
                    min(args.topk, len(possible_trg_words)), dim=1)
            predictions = [possible_trg_words[i] for i in nn[0]]
            probs = softmax(nn_vals/TEMP).tolist()[0]
            translation[vocab_ind] = list(zip(predictions,probs))
        else:
            translation[vocab_ind] = []

    # Print Output
    print('The following words are OOV:', file=sys.stderr)
    for i in oov:
        print(i, file=sys.stderr)

    for src_ind in translation:
        for trg_pair in translation[src_ind]:
            print(src_ind2word[src_ind], trg_ind2word[trg_pair[0]],
                    trg_pair[1], sep='\t')


if __name__ == '__main__':
    main()
