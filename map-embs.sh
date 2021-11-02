#!/bin/bash

. ./local-settings.sh

SRC=$1
TRG=$2

SRC_EMBS=`pwd`/embs/wiki.$SRC.vec
TRG_EMBS=`pwd`/embs/wiki.$TRG.vec
OUTDIR=`pwd`/embs/mapped/$SRC-$TRG
mkdir -p $OUTDIR
echo Outdir: $OUTDIR

python3 $VECMAP/map_embeddings.py --cuda --unsupervised --max_embs 200000 \
	$SRC_EMBS $TRG_EMBS $OUTDIR/$SRC.mapped $OUTDIR/$TRG.mapped
