from collections import defaultdict
import sys

def count_words(corpus, sep='||_||'):
    # Processes list of paired sentences separated by sep. 
    # Returns sentences and vocabulary for both languages.
    word_counts = {'src': defaultdict(int), 'trg': defaultdict(int)}
    for line in corpus:
        l1_sent, l2_sent = line.split(sep)
        l1_words = l1_sent.strip().split()
        l2_words = l2_sent.strip().split()
        for word in l1_words:
            word_counts['src'][word] += 1
        for word in l2_words:
            word_counts['trg'][word] += 1
    return word_counts
    

def possible_word_pairs(corpus, src_word2ind, trg_word2ind, sep='||_||'): 
    src2trg_pairs = defaultdict(set)
    trg2src_pairs = defaultdict(set)
    src2trg_pairs_inds = defaultdict(set)
    trg2src_pairs_inds = defaultdict(set)
    word_counts = {'src': defaultdict(int), 'trg': defaultdict(int)}
    for line in corpus:
        l1_sent, l2_sent = line.split(sep)
        l1_words = l1_sent.strip().split()
        l2_words = l2_sent.strip().split()
        word_counts, src2trg_pairs, src2trg_pairs_inds, _, _ = (
                _update_wordpairs_and_counts(l1_words, l2_words, word_counts, 
                    'src', src2trg_pairs, src2trg_pairs_inds, src_word2ind, trg_word2ind))
        word_counts, trg2src_pairs, trg2src_pairs_inds, _, _= (
                _update_wordpairs_and_counts(l2_words, l1_words, word_counts, 
                    'trg', trg2src_pairs, trg2src_pairs_inds, trg_word2ind, src_word2ind)) 
    return (word_counts, src2trg_pairs, trg2src_pairs, src2trg_pairs_inds, 
            trg2src_pairs_inds) 


def _update_wordpairs_and_counts(src_words, trg_words, word_counts, key, pairs,
        pairs_inds, src_word2ind, trg_word2ind):
    # Note - will only do updates for words that are in relevant source and
    # target vocabularies. 
    oov_src = set()
    oov_trg = set()
    for src_word in src_words:
        pairs[src_word].update(trg_words)
        word_counts[key][src_word] += 1
        if src_word2ind.get(src_word) is None:
            oov_src.add(src_word)
            continue
        trg_inds = set()
        for trg_word in trg_words:
            try:
                trg_inds.add(trg_word2ind[trg_word]) 
                pairs_inds[src_word2ind[src_word]].update(trg_inds)
            except KeyError:
                oov_trg.add(trg_word)
    return word_counts, pairs, pairs_inds, oov_src, oov_trg 


def write_probs_outfile(probs, outfile, word_counts, mincount=0, 
        topk=None, maxcount=sys.maxsize, sep='||_||'): 
    # Write probs outfile. If topk=-1, return all predictions.
    with open(outfile, 'w', encoding='utf-8', errors='surrogateescape') as outfile:
        for srcwd in probs:
            if (word_counts['src'][srcwd] >= mincount and 
                    word_counts['src'][srcwd] < maxcount):
                trgwd_probs = [(k,v) for k,v in probs[srcwd].items()]
                sorted_trgwds = sorted(
                        trgwd_probs, key=lambda tup: tup[1], reverse=True) 
                if topk is not None:
                    sorted_trgwds = sorted_trgwds[:topk]
                for trgwd, prob in sorted_trgwds:
                    if (word_counts['trg'][trgwd] >= mincount and
                            word_counts['trg'][trgwd] < maxcount):
                        outfile.write("%s %s %s %s %0.4f\n" % (srcwd, sep, trgwd, sep, prob))


def dict_from_probsfile(probsfile, sep='||_||'):
    probs = defaultdict(dict)
    for line in probsfile:
        srcwd, trgwd, prob = line.split(sep)
        probs[srcwd.strip()][trgwd.strip()] = float(prob) 
    return probs


def merge_probsfiles(pf1, pf2):
    # Merge two probs files. pf2 takes preference.
    pdict1 = dict_from_probsfile(pf1)
    pdict2 = dict_from_probsfile(pf2)
    for srcwd in pdict2:
        if pdict1.get(srcwd): 
            pdict1[srcwd].update(pdict2[srcwd])
        else:
            pdict1[srcwd] = pdict2[srcwd]
    return pdict1


def merge_probsfiles_and_write(pf1, pf2, outfile, sep='||_||'):
    pfile1 = open(pf1, 'r', encoding='utf-8', errors='surrogateescape')
    pfile2 = open(pf2, 'r', encoding='utf-8', errors='surrogateescape')
    outfile = open(outfile, 'w', encoding='utf-8', errors='surrogateescape')
    probs = merge_probsfiles(pfile1, pfile2)
    for srcwd in probs:
        for trgwd, prob in probs[srcwd].items():
            outfile.write("%s %s %s %s %0.4f\n" % (srcwd, sep, trgwd, sep, prob))
