# MSJE: Learning TFIDF Enhanced Joint Embedding for Recipe-Image Cross-Modal Retrieval Service

This repository contains the code to train and evaluate models from the paper:  
_Learning TFIDF Enhanced Joint Embedding for Recipe-Image Cross-Modal Retrieval Service_

[//]: # ( If you find this code useful, please consider citing:)


 

## Contents
1. [Instroduction](#introduction)
1. [Installation](#installation)
2. [Recipe1M Dataset](#recipe1m-dataset)
3. [Vision models](#vision-models)
4. [Out-of-the-box training](#out-of-the-box-training)
5. [Training](#training)
6. [Testing](#testing)
7. [Contact](#contact)

## Introduction

### Recipe-to-Image retrieval task

<p align="center">
    <img src="assets/retrieval_exp.png" width="800"/>
</p>

Given a recipe query which contains the recipe title, a list of ingredients and a sequence of cooking instructions, the goal is to train a statistical model to retrieve the associated image. For the recipe query, we list the top 5 images retrieved by JESR, ACME and our MSJE model.

## Installation

We use the environment with Python 3.7.6 and Pytorch 1.4.0. Run ```pip install --upgrade cython``` and then install the dependencies with ```pip install -r requirements.txt```. Our work is an extension of [im2recipe](https://github.com/torralba-lab/im2recipe-Pytorch).

## Recipe1M Dataset

The Recipe1M dataset is available for download [here](http://im2recipe.csail.mit.edu/dataset/download), where you can find some code used to construct the dataset and get the structured recipe text, food images, pre-trained instruction featuers and so on. 

## Vision models

This current version of the code uses a pre-trained ResNet-50.

## Out-of-the-box training

To train the model, you will need to create following files:
* `data/train_lmdb`: LMDB (training) containing skip-instructions vectors, ingredient ids and categories.
* `data/train_keys`: pickle (training) file containing skip-instructions vectors, ingredient ids and categories.
* `data/val_lmdb`: LMDB (validation) containing skip-instructions vectors, ingredient ids and categories.
* `data/val_keys`: pickle (validation) file containing skip-instructions vectors, ingredient ids and categories.
* `data/test_lmdb`: LMDB (testing) containing skip-instructions vectors, ingredient ids and categories.
* `data/test_keys`: pickle (testing) file containing skip-instructions vectors, ingredient ids and categories.
* `data/text/vocab.txt`: file containing all the vocabulary found within the recipes.

Recipe1M LMDBs and pickle files can be found in train.tar, val.tar and test.tar. [here](http://im2recipe.csail.mit.edu/dataset/download)

It is worth mentioning that the code is expecting images to be located in a four-level folder structure, e.g. image named `0fa8309c13.jpg` can be found in `./data/images/0/f/a/8/0fa8309c13.jpg`. Each one of the Tar files contains the first folder level, 16 in total. 

The pre-trained TFIDF vectors for each recipe, image category feature for each image and the optimized category label for each image-recipe pair can be found in [id2tfidf_vec.pkl](https://drive.google.com/file/d/1TF6iuUDHqfqIT-LT5WQ-zX98mNyTzuOS/view?usp=sharing), [id2img_101_cls_vec.pkl](https://drive.google.com/file/d/1RGkl7ghjoX25hhgq2837Ou5iQWboiPbi/view?usp=sharing) and [id2class_1005.pkl](https://drive.google.com/file/d/1J75v63kgMTNGl9UKldq9kGQhSMRom1NX/view?usp=sharing) respectively.
### Word2Vec

Training word2vec with recipe data:

- Download and compile [word2vec](https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip)
- Train with:

```
./word2vec -hs 1 -negative 0 -window 10 -cbow 0 -iter 10 -size 300 -binary 1 -min-count 10 -threads 20 -train tokenized_instructions_train.txt -output vocab.bin
```

- Run ```python get_vocab.py vocab.bin``` to extract dictionary entries from the w2v binary file. This script will save ```vocab.txt```, which will be used to create the dataset later.
- Move ```vocab.bin``` and ```vocab.txt``` to ```./data/text/```.



## Training

- Train the model with: 
```
CUDA_VISIBLE_DEVICES=0 python train.py 
```
We did the experiments with batch size 100, which takes about 11 GB memory.



## Testing
- Test the trained model with
```
CUDA_VISIBLE_DEVICES=0 python test.py
```
- The results will be saved in ```results```, which include the MedR result and recall scores for the recipe-to-image retrieval and image-to-recipe retrieval.
- Our best model trained with Recipe1M (TSC paper) can be downloaded [here](https://drive.google.com/drive/folders/1q4MpqSXr_ZCy2QiBn1XV-B6fFlQFjwSV?usp=sharing).


## Contact

For any questions or suggestions you can use the issues section or reach us at zhongweixie@gatech.edu.
