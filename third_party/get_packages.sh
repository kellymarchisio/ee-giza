# Set up our fork of vecmap
git clone https://github.com/artetxem/vecmap.git 
cp ../scripts/map_embeddings.py vecmap/
# Please pardon the hackiness... but it worked...
echo >> vecmap/embeddings.py
echo >> vecmap/embeddings.py
cat vecmap/cupy_utils.py >> vecmap/embeddings.py
sed -i 's/from cupy_utils import */## from cupy_utils import */g' vecmap/embeddings.py
mv vecmap vecmap-fork

git clone https://github.com/lilt/alignment-scripts.git
cp ../scripts/combine-with-outdir.sh alignment-scripts/scripts/combine.sh
mv alignment-scripts alignment-scripts-fork

