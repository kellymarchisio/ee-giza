. ./local-settings.sh

cd $DIR/third_party/alignment-scripts-fork 
./preprocess/run.sh

# This issue may have been fixed in later versions of alignment-scripts (Lilt) 
cp_clean() {
	input=$1	
	output=$2
	# Clean out any non-breaking spaces.
	cat $input | sed 's/\xC2\xA0/ /g' | sed 's/\xE2\x80\x89//g' \
		 | sed 's/\xE3\x80\x80/ /g' > $output
}

cd $DIR
for lang in deen roen enfr; do
	for drt in src tgt; do
		cp_clean third_party/alignment-scripts-fork/train/*plustest* data/train
		cp third_party/alignment-scripts-fork/test/$lang.lc.$drt data/test
	done	
	cp third_party/alignment-scripts-fork/test/$lang.talp data/test
done

