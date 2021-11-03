. ./local-settings.sh

cd $DIR/third_party/alignment-scripts-fork 
./preprocess/run.sh
cd $DIR

mkdir -p data/train data/test
cp third_party/alignment-scripts-fork/train/*plustest* data/train
for lang in deen roen enfr; do
	cp third_party/alignment-scripts-fork/test/$lang.talp data/test
done

