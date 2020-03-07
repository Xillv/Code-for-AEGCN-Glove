# Code-for-AEGCN-Glove
Code for [applied sciences](https://www.mdpi.com/) paper "[Targeted Sentiment Classification Based on Attentional Encoding and Graph Convolutional Networks](https://www.mdpi.com/2076-3417/10/3/957)"
#Model

# Requirements
* Python >= 3.6
* pytorch >= 1.0
* SpaCy 2.0.18
* numpy 1.15.4
# Usage
* First install SpaCy package and language models with
```
pip install spacy
```
and
```
python -m spacy download en
```
* Download pretrained GloVe embeddings [here](http://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.840B.300d.zip) and put ```glove.840B.300d.txt``` into ```glove/```.
* Train with [train.py]
# Citation
If you find this repository is helpful to you, please kindly cite our paper and star this repo
```
@article{xiao2020targeted,
    title={Targeted Sentiment Classification Based on Attentional Encoding and Graph Convolutional Networks},
    author={Xiao, Luwei and Hu, Xiaohui and Chen, Yinong and Xue, Yun and Gu, Donghong and Chen, Bingliang and Tao, Zhang},
    journal={Applied Sciences},
    volume={10},
    number={3},
    pages={957},
    year={2020},
    publisher={Multidisciplinary Digital Publishing Institute}
}
```
# Acknowledgement
This work is mainly based on the repositories of [ASGCN](https://github.com/GeneZC/ASGCN) and [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch).Thanks a million to anyone who assists with this work and their valuable devotion!
