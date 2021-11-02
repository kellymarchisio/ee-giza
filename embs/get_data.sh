################################################################################
# Get Wikipedia embeddings.
################################################################################
for lang in en ro de fr 
do
	wget -c https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$lang.vec 
done
