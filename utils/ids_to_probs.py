import sys

probs_file=open(sys.argv[1], 'r')
src_vocab_file=open(sys.argv[2], 'r')
trg_vocab_file=open(sys.argv[3], 'r')

src_vocab={}
trg_vocab={}

for line in src_vocab_file:
    id, wd, _ = line.split()
    src_vocab[wd] = id

for line in trg_vocab_file:
    id, wd, _ = line.split()
    trg_vocab[wd] = id

for line in probs_file:
    src, trg, prob = line.split()
    print(src_vocab[src], trg_vocab[trg], prob)
