import pandas as pd
import copy
import numpy as np
from PIL import Image
import os
import re

def create_txt(file_path, label, img_path):   
    new_text = []
    greater_1 = []
    no_img = []
    
    with open(file_path) as f:
        for index, line in enumerate(f):
            line = line.strip()
            
            if index % 3 == 0:
                tmp = line.split('|')
                id = tmp[0]
            elif index % 3 == 1:
                imgs = [tmp for tmp in line.split('|') if tmp != 'null']
            elif index % 3 == 2:
                content = line

                clean_postText = re.sub(r"(http|https)((\W+)(\w+)(\W+)(\w*)(\W+)(\w*)|(\W+)(\w+)(\W+)|(\W+))","",content)
                img_list = pick_image(imgs, img_path)
                
                if len(img_list) > 1:
                    greater_1.append(id)
                
                if len(img_list) == 0:
                    no_img.append(id)
                
                if clean_postText == "" and len(img_list) == 0:
                    new_text.append((id, ','.join(img_list), content, clean_postText, label, "no_image and no_text"))
                elif clean_postText == "":
                    new_text.append((id, ','.join(img_list), content, clean_postText, label, "no_text"))
                elif len(img_list) == 0:
                    new_text.append((id, ','.join(img_list), content, clean_postText, label, "no_image"))
                else:
                    new_text.append((id, ','.join(img_list), content, clean_postText, label, "full"))
    
    print('greater 1 [{}]'.format(len(greater_1)))
    print(greater_1)
    print('no_img [{}]'.format(len(no_img)))
    print(no_img)
    
    return new_text

def pick_image(imgs, img_path):
    img_list =[]
    for img in imgs:
        img_name = img.split('/')[-1]         
        if os.path.exists(os.path.join(img_path, img_name)):
            img_list.append(img_name)    

    return img_list
    
def process(input_path):
    data = pd.read_csv(input_path)
    data = data[data['note']=='full']

    texts = list(data['clean_text'])
    imgs = list(data['imgs'])
    ids = list(data['id'])
    labels = list(data['label'])
    
    reserved_id, reserved_text, reserved_img, reserved_label = [], [], [], []
    for id, text, img, label in zip(ids, texts, imgs, labels):
        label = 0 if label == 'fake' else 1
        
        picked_img = pick_visual(img, img_path)
        if picked_img == -1:
            continue
            
        reserved_id.append(id)
        reserved_text.append(text)
        reserved_img.append(picked_img)
        reserved_label.append(label)
        
    return reserved_id, reserved_text, reserved_img, reserved_label
    
def pick_visual(imgs, img_path):
    img_list = imgs.split(',')
    for img in img_list:
        img_file_path = os.path.join(img_path, img)
        if os.path.exists(img_file_path):
            im = Image.open(img_file_path).convert('RGB')
            if im.width < 64 and im.height < 64:
                continue
            else:
                return img

    return -1

def count_label(reserved_label):
    t = [1 for l in reserved_label if l == 1]
    f = [1 for l in reserved_label if l == 0]

    print('real [{}], fake [{}]'.format(len(t), len(f)))


rumor_img_path = './MM17-WeiboRumorSet/rumor_images'
nonrumor_img_path = './MM17-WeiboRumorSet/nonrumor_images'
    
train_rumor_path = './MM17-WeiboRumorSet/tweets/train_rumor.txt'
train_nonrumor_path = './MM17-WeiboRumorSet/tweets/train_nonrumor.txt'

test_rumor_path = './MM17-WeiboRumorSet/tweets/test_rumor.txt'
test_nonrumor_path = './MM17-WeiboRumorSet/tweets/test_nonrumor.txt'


train_rumor_frame = create_txt(train_rumor_path, 'fake', rumor_img_path)
train_nonrumor_frame = create_txt(train_nonrumor_path, 'real', nonrumor_img_path)

test_rumor_frame = create_txt(test_rumor_path, 'fake', rumor_img_path)
test_nonrumor_frame = create_txt(test_nonrumor_path, 'real', nonrumor_img_path)

train_data = []
train_data.extend(train_rumor_frame)
train_data.extend(train_nonrumor_frame)

train_frame = pd.DataFrame(train_data, columns=['id', 'imgs', 'text', 'clean_text', 'label', 'note']) 
train_frame.to_csv("test_ori.csv")

test_data = []
test_data.extend(test_rumor_frame)
test_data.extend(test_nonrumor_frame)

test_frame = pd.DataFrame(test_data, columns=['id', 'imgs', 'text', 'clean_text', 'label', 'note']) 
test_frame.to_csv("test_ori.csv")


train_file = './train_ori.csv'
test_file = './test_ori.csv'
img_path = './MM17-WeiboRumorSet/origin_images'

print('---------------------------------process train---------------------------------------')    
reserved_id, reserved_text, reserved_img, reserved_label = process(train_file)
train = zip(reserved_id, reserved_text, reserved_img, reserved_label)

train_df = pd.DataFrame(train, columns=['id', 'text', 'imgs', 'label'])
count_label(reserved_label)
print('total num')
print(len(train_df))

print('---------------------------------process test---------------------------------------')    
reserved_id, reserved_text, reserved_img, reserved_label = process(test_file)
test = zip(reserved_id, reserved_text, reserved_img, reserved_label)

test_df = pd.DataFrame(test, columns=['id', 'text', 'imgs', 'label'])
count_label(reserved_label)
print('total num')
print(len(test_df))

train_df.to_csv("train.csv")
test_df.to_csv("test.csv")



