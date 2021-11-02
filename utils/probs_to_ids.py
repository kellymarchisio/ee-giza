import sys

probs_file=open(sys.argv[1], 'r', encoding='utf-8', errors='surrogateescape')
src_vocab_file=open(sys.argv[2], 'r', encoding='utf-8',
        errors='surrogateescape')
trg_vocab_file=open(sys.argv[3], 'r', encoding='utf-8',
        errors='surrogateescape')

src_vocab={}
trg_vocab={}
src_counts={}
trg_counts={}

for line in src_vocab_file:
    id, wd, count = line.split()
    src_vocab[wd] = id
    src_counts[wd] = count

for line in trg_vocab_file:
    id, wd, count = line.split()
    trg_vocab[wd] = id
    trg_counts[wd] = count

for line in probs_file:
    src, trg, prob = line.split()
    print(src_vocab[src], trg_vocab[trg], src_counts[src], trg_counts[trg], prob)
