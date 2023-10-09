# Multi-Modal Transformer with Global-Local Alignment for Composed Query Image Retrieval.

## Introduction
<div align=center><img src="figs/figure_0.png" width="400"></div>
In this paper, we study the composed query image retrieval, which aims at retrieving the target image similar to the composed query, i.e., a reference image and the desired modification text. 

## ComqueryFormer Architecture 
<div align=center><img src="figs/figure_2.png" width="600"></div>

We introduce the first unified multi-modal transformer named ComqueryFormer for composed query image retrieval, which performs the feature encoding and composition by a stack of transformer layers. In addition, we propose an effective global-local mechanism to align the composed query and target image in a complementary manner. Especially, we propose to implicitly detect discriminative visual regions through learnable region masks for local alignment. The ComqueryFormer outperforms previous state-of-the-art results on three public datasets that are FashionIQ, CIRR, and Fashion200K.

## Description of the Code [(From TIRG)](https://github.com/google/tirg/edit/master/README.md)
The code is based on TIRG code. 
`datasets.py` and `test_retrieval.py` have been modified to add Fashion IQ dataset.
- `main.py`: driver script to run training/testing
- `datasets.py`: Dataset classes for loading images & generate training retrieval queries
- `text_model.py`: LSTM model to extract text features
- `img_text_composition_models.py`: various image text compostion models 
- `torch_function.py`: contains soft triplet loss function and feature normalization function
- `test_retrieval.py`: functions to perform retrieval test and compute recall performance


## Getting Started
### Prerequisites
```
pip install -r requirement.txt
```

### Datasets

**FashionIQ**: Download FashionIQ dataset images from [here](https://github.com/hongwang600/fashion-iq-metadata). 

**Fashion200K**: Download the Fashion200K dataset images from [webpage](https://github.com/xthan/fashion-200k) and the generated test_queries.txt by [TIRG](https://github.com/google/tirg/edit/master/README.md) from [here](https://storage.googleapis.com/image_retrieval_css/test_queries.txt). 

**CIRR**: Please check out the [CIRR repo](https://github.com/Cuberick-Orion/CIRR#download-cirr-dataset) for instructions. 

## Running the Code

For training and testing new models, pass the appropriate arguments. 

For instance, for training ComqueryFormer model on FashionIQ dataset run the following command:

```
python -W ignore main.py --dataset=fashionIQ --dataset_path=datasets/fashion_iq --model=comqueryformer --loss=batch_based_classification --use_complete_text_query False  --epochs=100  --log_dir=logs/fashioniq_compare/ --epochs=100 --loader_num_workers=2 --batch_size=24 --comment=_comqueryformer
```










