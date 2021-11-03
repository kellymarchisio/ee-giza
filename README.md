Embedding-Enhanced GIZA++: Improving Word Alignment in Very Low-Resource Settings 
======================

This is an implementation of the alignment system presented in:
- Kelly Marchisio, Conghao Xiong, and Philipp Koehn. 2021. **[Embedding-Enhanced 
GIZA++: Improving Word Alignment in Very Low-Resource Settings Using
Embeddings](https://arxiv.org/abs/2104.08721)**. Preprint.


If you use this software for academic research, please cite the paper above.

Requirements
--------
- python3
- cupy
- scipy
- pytorch
- boost
--------

Setup
--------
To download pretrained word embeddings, run `sh get_data.sh` from the embs/ folder.
To download the necessary software, run `sh get_packages.sh` from the third\_party/ folder.
Manually download the En-De data. Instructions: https://github.com/lilt/alignment-scripts 
Run `sh setup.sh`

To build ee-giza++:
```
cd eegizapp 
mkdir build
cd build
cmake ..
make
```


Usage
--------

Map embeddings for the language pair you want to align. For example, to map
German and English pretrained embedding spaces with VecMap, run: 

	sh map-embs.sh de en

Then run EE-Giza++ with 1000 sentences:

	sh run-eegiza.sh de en 1000 

Results for eegiza++ will be found in exps/de-en/1000/results/deen-results.txt

Results for vanilla giza++ will be found in exps/de-en/1000/results-vanilla/deen-results.txt


Troubleshooting
--------
You'll need Boost to compile eegiza++. Here's what my env variables look like:

```
export LIBRARY_PATH="/PATH/TO/BOOST/boost_1_65_1/install/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/PATH/TO/BOOST/boost_1_65_1/install/lib:$LD_LIBRARY_PATH"
export INCLUDE_PATH="/PATH/TO/BOOST/boost_1_65_1/install/include:$INCLUDE_PATH"
export CMAKE_LIBRARY_PATH="/PATH/TO/BOOST/boost_1_65_1/install/lib:$CMAKE_LIBRARY_PATH"
export CMAKE_INCLUDE_PATH="/PATH/TO/BOOST/boost_1_65_1/install/include:$CMAKE_INCLUDE_PATH"
export BOOST_ROOT="/PATH/TO/BOOST/boost_1_65_1/install"
export BOOST_INCLUDE="/PATH/TO/BOOST/boost_1_65_1/install/include"
export BOOST_LIBDIR="/PATH/TO/BOOST/boost_1_65_1/install/lib"
```
