################################################################################
# 
# This script is a small edit to Lilt's combine.sh here:
#	https://github.com/lilt/alignment-scripts/blob/master/scripts/combine.sh 
#
# It was edited to allow specifying output directory by Kelly Marchisio, 2021.
#
################################################################################

#!/bin/bash

set -e

SCRIPT_DIR=${0%/combine.sh}

# check parameter count and write usage instruction
if (( $# != 4 )); then
  echo "Usage: $0 alignment reverse_alignment reference_path outdir"
  exit 1
fi

reference_path=$3
reference_lines=`cat $3 | wc -l`

outdir=$4

alignment_path=$1
alignment_reverse_path=$2
alignment_name=${1##*/}
alignment_reverse_name=${2##*/}
alignment_prefix=${alignment_name%.*}


# only use test data
tail -n $reference_lines $alignment_path > $outdir/test.${alignment_name}
tail -n $reference_lines $alignment_reverse_path > $outdir/test.${alignment_reverse_name}

# Do not reverse the alignment there
for method in "grow-diagonal-final" "grow-diagonal" "intersection" "union"; do
  ${SCRIPT_DIR}/combine_bidirectional_alignments.py \
	  $outdir/test.${alignment_name} \
	  $outdir/test.${alignment_reverse_name} --method $method $5 \
	  > $outdir/test.${alignment_prefix}.${method}.talp
done

for file_path in $outdir/test.${alignment_prefix}*.talp $outdir/test.${alignment_reverse_name}; do
  reverseRef=""
  if [[ ${file_path} == $outdir/test.${alignment_reverse_name} ]]; then
    reverseRef="--reverseRef"
  fi
  ${SCRIPT_DIR}/aer.py ${reference_path} ${file_path} --oneRef $reverseRef --fAlpha 0.5
done

