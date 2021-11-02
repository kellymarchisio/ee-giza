import sys

probs_file=open(sys.argv[1], 'r')
src_vocab_file=open(sys.argv[2], 'r')
trg_vocab_file=open(sys.argv[3], 'r')

src_vocab={}
src_vocab['0'] = 'NULL'
trg_vocab={}
trg_vocab['0'] = 'NULL'
for line in src_vocab_file:
    id, wd, _ = line.split()
    src_vocab[id] = wd 

for line in trg_vocab_file:
    id, wd, _ = line.split()
    trg_vocab[id] = wd

for line in probs_file:
    src, trg, prob = line.split()
    print(src_vocab[src], trg_vocab[trg], prob)
