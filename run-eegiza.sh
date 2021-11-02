#!/bin/bash 

###############################################################################
#
# Embedding-Enhanced Giza++ run script.
#
# Written by Kelly Marchisio for the CLSP Grid, March 2021.
#	Based off of: https://github.com/lilt/alignment-scripts
#
###############################################################################

. ./local-settings.sh

SRC=$1
TRG=$2
N=$3
OUTDIR=`pwd`/exps/$SRC-$TRG/$N
mkdir -p $OUTDIR

if [ -z $SRC ] || [ -z $TRG ] || [ -z $N ]; then
	echo "You must provide src/trg language and lines of bitext. Exiting." 
	exit
fi


#################################################
# Set up data and directories.

INPUT_SENTS=data/train/$SRC$TRG.lc.plustest
CORPUS=$OUTDIR/corpus.$SRC-$TRG
tail -n $N $INPUT_SENTS.src > $CORPUS.$SRC.$N
tail -n $N $INPUT_SENTS.tgt > $CORPUS.$TRG.$N

$LILT/scripts/create_fast_align_corpus.sh \
	$CORPUS.$SRC.$N $CORPUS.$TRG.$N $CORPUS	
$LILT/scripts/create_fast_align_corpus.sh \
	$CORPUS.$TRG.$N $CORPUS.$SRC.$N $CORPUS.rev	

SOURCE_PATH=$CORPUS.$SRC.$N
TARGET_PATH=$CORPUS.$TRG.$N
SOURCE_NAME=`basename $SOURCE_PATH`
TARGET_NAME=`basename $TARGET_PATH`
TEST_TALP=data/test/$SRC$TRG.talp

SRC_EMBS=embs/mapped/$SRC-$TRG/$SRC.mapped
TRG_EMBS=embs/mapped/$SRC-$TRG/$TRG.mapped


#################################################
# Make GIZA++ files.

eegizapp/build/bin/plain2snt ${SOURCE_PATH} ${TARGET_PATH}
eegizapp/build/bin/mkcls -n10 -p${SOURCE_PATH} -V${SOURCE_PATH}.class &
eegizapp/build/bin/mkcls -n10 -p${TARGET_PATH} -V${TARGET_PATH}.class &
wait

eegizapp/build/bin/snt2cooc \
	${SOURCE_PATH}_${TARGET_NAME}.cooc ${SOURCE_PATH}.vcb \
	${TARGET_PATH}.vcb ${SOURCE_PATH}_${TARGET_NAME}.snt &
eegizapp/build/bin/snt2cooc \
	${TARGET_PATH}_${SOURCE_NAME}.cooc ${TARGET_PATH}.vcb \
	${TARGET_PATH}.vcb ${TARGET_PATH}_${SOURCE_NAME}.snt &
wait

#################################################
# Make GIZA++ and CSLS Probability files.

FREQS_DIR=$OUTDIR/freqs
PROB_TABLES_DIR=$OUTDIR/prob-tables
mkdir -p $FREQS_DIR $PROB_TABLES_DIR

cut -d' ' -f2 ${SOURCE_PATH}.vcb > $FREQS_DIR/freqall.srcwds.$N.$SRC
cut -d' ' -f2 ${TARGET_PATH}.vcb > $FREQS_DIR/freqall.srcwds.$N.$TRG

LOW_FREQ_WORDS_FWD=$PROB_TABLES_DIR/low_freq_words_all.$SRC.probs
LOW_FREQ_WORDS_REV=$PROB_TABLES_DIR/low_freq_words_all.$TRG.probs
# Get probs of low-freq words according to VecMap, in both directions. 
python3 probdist_over_csls.py $SRC_EMBS $TRG_EMBS \
	-i $FREQS_DIR/freqall.srcwds.$N.$SRC --sents $CORPUS \
	--sep "|||" > $LOW_FREQ_WORDS_FWD
python3 probdist_over_csls.py $TRG_EMBS $SRC_EMBS \
	-i $FREQS_DIR/freqall.srcwds.$N.$TRG --sents $CORPUS.rev \
	--sep "|||" > $LOW_FREQ_WORDS_REV
python `pwd`/utils/probs_to_ids.py $LOW_FREQ_WORDS_FWD ${SOURCE_PATH}.vcb \
	${TARGET_PATH}.vcb > $PROB_TABLES_DIR/`basename $LOW_FREQ_WORDS_FWD`.ids
python `pwd`/utils/probs_to_ids.py $LOW_FREQ_WORDS_REV ${TARGET_PATH}.vcb \
	${SOURCE_PATH}.vcb > $PROB_TABLES_DIR/`basename $LOW_FREQ_WORDS_REV`.ids

#################################################
# GIZA++

run_giza () {
	results_dir=$1

	for direction in "Forward" "Backward"; do
		out_folder=$results_dir/$direction
		eegizapp/build/bin/mgiza $out_folder/config/config.txt \
			1> $out_folder/stdout 2> $out_folder/stderr
		cat $out_folder/*A3.final* > $out_folder/allA3.final.txt
	done

	$LILT/scripts/a3ToTalp.py < $results_dir/Forward/allA3.final.txt \
		> $results_dir/$SRC$TRG.talp
	$LILT/scripts/a3ToTalp.py < $results_dir/Backward/allA3.final.txt \
		> $results_dir/$SRC$TRG.reverse.talp
	$LILT/scripts/combine.sh $results_dir/$SRC$TRG.talp \
		$results_dir/$SRC$TRG.reverse.talp $TEST_TALP $results_dir \
		> $results_dir/$SRC$TRG-results.txt
}

cp base-fwd-config.txt $OUTDIR/fwd-config.txt
cp base-rev-config.txt $OUTDIR/rev-config.txt
sed -i "s/SRC/$SRC/g" $OUTDIR/*-config.txt
sed -i "s/TRG/$TRG/g" $OUTDIR/*-config.txt
sed -i "s/SIZE/$N/g" $OUTDIR/*-config.txt

#################################################
# Run GIZA++ with interpolation.
RESULTS_DIR=$OUTDIR/results
mkdir -p $RESULTS_DIR/Forward/config $RESULTS_DIR/Backward/config
cp $OUTDIR/fwd-config.txt $RESULTS_DIR/Forward/config/config.txt
cp $OUTDIR/rev-config.txt $RESULTS_DIR/Backward/config/config.txt
sed -i "s/RESULTS_DIR/results/g" $RESULTS_DIR/*/config/config.txt 

run_giza $RESULTS_DIR 

#################################################
# Run Vanilla GIZA++. 

RESULTS_DIR=$OUTDIR/results-vanilla
mkdir -p $RESULTS_DIR/Forward/config $RESULTS_DIR/Backward/config
cp $OUTDIR/fwd-config.txt $RESULTS_DIR/Forward/config/config.txt
cp $OUTDIR/rev-config.txt $RESULTS_DIR/Backward/config/config.txt
sed -i "s/RESULTS_DIR/results-vanilla/g" $RESULTS_DIR/*/config/config.txt 
sed -i "s/interpolateprobsfromfilem1 1/interpolateprobsfromfilem1 0/g" \
	$RESULTS_DIR/*/config/config.txt
sed -i "s/interpolateprobsfromfilehmm 1/interpolateprobsfromfilehmm 0/g" \
	$RESULTS_DIR/*/config/config.txt

run_giza $RESULTS_DIR 

