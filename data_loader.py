import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms

class Rumor_Dataset(Dataset):
    def __init__(self, args, data_path, image_path):

        self.dataset = args.dataset
        self.max_words = args.max_words  
        self.image_path = image_path
        
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        
        if self.dataset == 'twitter':
            self.trans = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.454, 0.440, 0.423], [0.282, 0.278, 0.278])
            ])

        elif self.dataset == 'weibo':
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            raise 'Dataset Error'

        texts, imgs, ids, labels = self.read_data(data_path)

        self.texts = texts
        self.imgs = imgs
        self.ids = ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        text = self.texts[index]
              
        text_content = self.tokenizer.encode_plus(text, add_special_tokens = True, padding = 'max_length', truncation = True, max_length = self.max_words, return_tensors = 'pt')
        
        label_tensor = torch.tensor(self.labels[index])
        if self.labels[index] == 0:
            path_tensor = torch.tensor([1.0, 0.0])
        elif self.labels[index] == 1:
            path_tensor = torch.tensor([0.0, 1.0])
            
        return text_content["input_ids"].flatten().clone().detach().type(torch.LongTensor), text_content["attention_mask"].flatten().clone().detach().type(torch.LongTensor), text_content["token_type_ids"].flatten().clone().detach().type(torch.LongTensor), self.process_image(self.imgs[index]), label_tensor, path_tensor
            
    def read_data(self, data_path):
        data = pd.read_csv(data_path)
    
        texts = list(data['text'])
        imgs = list(data['imgs'])
        ids = list(data['id'])
        labels = list(data['label'])
        
        count_label(data['label'])
        return texts, imgs, ids, labels
    
    def process_image(self, img_str):
        img_file_path = os.path.join(self.image_path, img_str)
        if os.path.exists(img_file_path):
            im = Image.open(img_file_path).convert('RGB')
            return self.trans(im) 

def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: x[1].sum(), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def count_label(reserved_label):
    t = [1 for l in reserved_label if l == 1]
    f = [1 for l in reserved_label if l == 0]

    print('real [{}], fake [{}]'.format(len(t), len(f)))
    
def get_Dataloader(args):
    print("loading train set", )
    traindata = Rumor_Dataset(args, args.train_path, args.image_path)
    print("train no:", len(traindata))
    
    print("loading test set", )
    testdata = Rumor_Dataset(args, args.test_path, args.image_path)
    print("test no:", len(testdata))

    train_loader = DataLoader(traindata, batch_size = args.batchsize, shuffle = True, num_workers=1)
    test_loader = DataLoader(testdata, batch_size = args.batchsize, shuffle = False, num_workers=4)

    return train_loader, test_loader

def get_Testloader(args):    
    print("loading test set", )
    testdata = Rumor_Dataset(args, args.test_path, args.image_path)
    print("test no:", len(testdata))

    test_loader = DataLoader(testdata, batch_size = args.batchsize, shuffle = False, num_workers=4)

    return test_loader