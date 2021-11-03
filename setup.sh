. ./local-settings.sh

cd $DIR/third_party/alignment-scripts-fork 
./preprocess/run.sh
cd $DIR

mkdir -p data/train data/test
for lang in deen roen enfr; do
	for drt in src tgt; do
		cp third_party/alignment-scripts-fork/train/*plustest* data/train
		cp third_party/alignment-scripts-fork/test/$lang.lc.$drt data/test
	done	
	cp third_party/alignment-scripts-fork/test/$lang.talp data/test
done

