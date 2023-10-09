import os 
import re
import pandas as pd

def process_sent(text):
    new_text = []
    for t in text.split(" "):
        t = '@' if t.startswith('@') and len(t) > 1 else t
        new_text.append(t)

    text = " ".join(new_text)
    text = text.replace('http: ', 'http:')    
    text = text.replace(r"\n", " ")
    text = text.replace("\\", "")
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", text)
    text = ' '.join(text.split())

    return text.strip()

def select_train(train_path1, train_path2, img_path):
    reserved_ids, reserved_content, reserved_clean_content, reserved_imgs, reserved_label = [], [], [], [], []
    greater_1 = []
    
    data = pd.read_csv(train_path1, sep='\t')

    ids = data['tweetId']
    content = data['tweetText']
    label = data['label']
    print('start [{}]'.format(len(label)))
    
    for i, imgs in enumerate(data['imageId(s)']):
        if label[i] != 'fake' and label[i] != 'real':
            continue
            
        image_name_list = []       
        for img in imgs.split(','):
            for postfix in ['.jpg', '.png', '.jpeg', '.gif']:
                img_name = img + postfix
                
                if os.path.exists(os.path.join(img_path, img_name)):
                    image_name_list.append(img_name)
        
        clean_content = process_sent(content[i])
        if len(image_name_list) != 0 and clean_content !="":
            reserved_ids.append(ids[i])
            reserved_content.append(clean_content)
            reserved_imgs.append(image_name_list[0])
            reserved_label.append(0 if label[i] == 'fake' else 1)

        if len(image_name_list) > 1:
            greater_1.append(ids[i])
    
    data = pd.read_csv(train_path2, sep='\t')

    ids = data['post_id']
    content = data['post_text']
    label = data['label']
    print('start [{}]'.format(len(label)))
    
    for i, imgs in enumerate(data['image_id(s)']):
        if label[i] != 'fake' and label[i] != 'real':
            continue
            
        image_name_list = []       
        for img in imgs.split(','):
            for postfix in ['.jpg', '.png', '.jpeg', '.gif']:
                img_name = img + postfix
                
                if os.path.exists(os.path.join(img_path, img_name)):
                    image_name_list.append(img_name)
                    
        clean_content = process_sent(content[i])
        if len(image_name_list) != 0 and clean_content !="":
            reserved_ids.append(ids[i])
            reserved_content.append(clean_content)
            reserved_imgs.append(image_name_list[0])
            reserved_label.append(0 if label[i] == 'fake' else 1)

        if len(image_name_list) > 1:
            greater_1.append(ids[i])

    new_text = zip(reserved_ids, reserved_content, reserved_imgs, reserved_label)
    print('reserved [{}]'.format(len(reserved_ids)))
    print(greater_1)
    print(len(greater_1))
    count_label(reserved_label)
    return pd.DataFrame(new_text, columns=['id', 'text', 'imgs', 'label']) 

def select_test(test_path1, test_path2, img_path):
    reserved_ids, reserved_content, reserved_clean_content, reserved_imgs, reserved_label = [], [], [], [], []
    greater_1 = []
    
    data = pd.read_csv(test_path1, sep='\t')
    ids = data['tweetId']
    content = data['tweetText']
    label = data['label']
    print('start [{}]'.format(len(label)))
    
    for i, imgs in enumerate(data['imageId(s)']):
        if label[i] != 'fake' and label[i] != 'real':
            continue
        
        image_name_list = []
        for img in imgs.split(','):
            for postfix in ['.jpg', '.png', '.jpeg', '.gif']:
                img_name = img + postfix
                
                if os.path.exists(os.path.join(img_path, img_name)):
                    image_name_list.append(img_name)
                    
        clean_content = process_sent(content[i])
        if len(image_name_list) != 0 and clean_content !="":
            reserved_ids.append(ids[i])
            reserved_content.append(clean_content)
            reserved_imgs.append(image_name_list[0])
            reserved_label.append(0 if label[i] == 'fake' else 1)

        if len(image_name_list) > 1:
            greater_1.append(ids[i])
    
    data = pd.read_csv(test_path2, sep='\t')
    ids = data['post_id']
    content = data['post_text']
    label = data['label']
    print('start [{}]'.format(len(label)))
    
    for i, imgs in enumerate(data['image_id']):
        if label[i] != 'fake' and label[i] != 'real':
            continue
            
        image_name_list = []        
        for img in imgs.split(','):
            for postfix in ['.jpg', '.png', '.jpeg', '.gif']:
                img_name = img + postfix
                
                if os.path.exists(os.path.join(img_path, img_name)):
                    image_name_list.append(img_name)
                
        clean_content = process_sent(content[i])
        if len(image_name_list) != 0 and clean_content !="":
            reserved_ids.append(ids[i])
            reserved_content.append(clean_content)
            reserved_imgs.append(image_name_list[0])
            reserved_label.append(0 if label[i] == 'fake' else 1)

        if len(image_name_list) > 1:
            greater_1.append(ids[i])

    new_text = zip(reserved_ids, reserved_content, reserved_imgs, reserved_label)
    print('reserved [{}]'.format(len(reserved_ids)))
    print(greater_1)
    print(len(greater_1))
    count_label(reserved_label)
    # return pd.DataFrame(new_text, columns=['id', 'text', 'clean_text', 'imgs', 'label']) 
    df = pd.DataFrame(new_text, columns=['id', 'text', 'imgs', 'label']) 
    # df = df.drop_duplicates(subset=['clean_text', 'imgs', 'label'],keep=False)
    count_label(df['label'])
    return df

def count_label(reserved_label):
    t = [1 for l in reserved_label if l == 1]
    f = [1 for l in reserved_label if l == 0]

    print('real [{}], fake [{}]'.format(len(t), len(f)))

train_path1 = './MediaEval/train/tweets.txt'
train_path2 = './MediaEval/train/posts.txt'

test_path1 = './MediaEval/test/tweets.txt'
test_path2 = './MediaEval/test/posts_groundtruth.txt'

img_path = './MediaEval/images'

train_frame = select_train(train_path1, train_path2, img_path)
train_frame.to_csv("train.csv")

test_frame = select_test(test_path1, test_path2, img_path)
test_frame.to_csv("test.csv")






