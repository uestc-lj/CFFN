# CFFN
Code for the MMAsia 2023 paper [Cross-modal Consistency Learning with Fine-grained Fusion Network for Multimodal Fake News Detection]

# Dataset
The datasets used in our paper are [Twitter](http://www.multimediaeval.org/mediaeval2016/verifyingmultimediause/index.html) and [Weibo](https://forms.gle/Hqzcv8DCy15JbeZW6).
The code of dataset preprocessing is listed in ``resource\dataset`` folder:
* ``Weibo`` folder for Weibo dataset
* ``Mediaeval`` folder for Twitter dataset

## pretrained model

For Weibo dataset, [bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main) is needed and moving it to the folder ``resource/bert``.

For Twitter dataset, [bert_base](https://huggingface.co/bert-base-uncased/tree/main) is needed and moving it to the folder ``resource/bert``.

# Dependencies
  ```
    python 3.8.1
    pytorch-pretrained-bert==0.6.2
    timm==0.4.12
    numpy==1.23.5
    tensorboard==2.12.0
    torch 1.9.0 + cu11.1
    torchvision==0.10.0
    pandas==1.5.3
    scikit-learn==0.24.1
    gensim==4.3.1
    jieba==0.42.1
    tqdm==4.64.1
    python-json-logger==2.0.7
    transformers==3.3.1
 ```
## Running the Code

For Weibo dataset：
 ```
 cd ./weibo
 bash train.sh
 ```

For Twitter dataset：
 ```
 cd ./Mediaeval
 bash train.sh
 ```


