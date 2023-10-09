import random, os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable

from model import DetectionModule
from data_loader import get_Dataloader
import logging
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
    
def save_rumor_model(rumor_accuracy, rumor_f, best_rumor_accuracy, best_rumor_f1, type, epoch, save_path):
    logger.info('{} rumor acc: {}, f_macro: {}'.format(type, rumor_accuracy, rumor_f))
                                        
    if rumor_accuracy > best_rumor_accuracy:
        best_rumor_accuracy = rumor_accuracy

        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'best_accuracy': best_rumor_accuracy}, save_path + ".rumorbest_{}.pt".format(type))
        logger.info("Saved {} best epoch {}, best rumor accuracy {}".format(type, epoch, best_rumor_accuracy))       
    
    if rumor_f > best_rumor_f1:
        best_rumor_f1 = rumor_f

        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'best_f1': best_rumor_f1}, save_path + ".rumorf1best_{}.pt".format(type))
        logger.info("Saved {} best epoch {}, best rumor f_macro {}".format(type, epoch, best_rumor_f1))

    return best_rumor_accuracy, best_rumor_f1

def training(model, args, train_loader, test_loader, writer):
    save_path = args.outdir + '/Weibo_model'
    best_rumor_accuracy, best_rumor_f1 = 0.0, 0.0
    running_loss = 0.0

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = args.learning_rate,
                                 weight_decay = args.weight_decay)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_path_detection = torch.nn.MSELoss()
    
    for epoch in range(int(args.num_train_epochs)):
        running_loss, loss_1, loss_2 = 0.0, 0.0, 0.0
        for input_ids, attention_mask, token_type_ids, img_emb, label, path_tensor in train_loader:
            # print('id {}'.format(id))
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            img_emb = img_emb.cuda()
            label = label.cuda()
            path_tensor = path_tensor.cuda()
                        
            model.train()
            
            optimizer.zero_grad()
            pre_detection, path_prob = model(input_ids, attention_mask, token_type_ids, img_emb)
            
            loss_detection = loss_func_detection(pre_detection, label)            
            loss_path = loss_path_detection(path_prob, path_tensor)
            
            loss = loss_detection + args.tradeoff * loss_path
            loss.backward()
            
            running_loss += loss.item()
            loss_1 += loss_detection.item()
            loss_2 += loss_path.item()
            
            optimizer.step()
                                
        logger.info('Epoch: {0}, Loss: {1}, loss_detection: {2}, loss_path: {3}'.format(epoch, running_loss, loss_1, loss_2))                
                
        with torch.no_grad():
            rumor_accuracy, rumor_f, test_loss = eval_model(model, test_loader)   
            
            best_rumor_accuracy, best_rumor_f1 = save_rumor_model(rumor_accuracy, rumor_f, best_rumor_accuracy, best_rumor_f1, 'test', epoch, save_path)
        
        writer.add_scalar('train_loss',
                            running_loss,
                            epoch)

        writer.add_scalar('test_loss',
                            test_loss,
                            epoch)
        
        
def eval_model(model, testset_reader):
    model.eval()
    
    loss_func_detection = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    pre_rumor, act_rumor = [], []
    for input_ids, attention_mask, token_type_ids, img_emb, label, path_tensor in testset_reader:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()
        img_emb = img_emb.cuda()
        label = label.cuda()

        pre_detection, _ = model(input_ids, attention_mask, token_type_ids, img_emb)
        
        pre_rumor.extend(pre_detection.cpu().max(1)[1].numpy())
        act_rumor.extend(label.cpu().numpy())
        
        loss_detection = loss_func_detection(pre_detection, label)
        test_loss += loss_detection.item()
       
    pre_rumor = np.array(pre_rumor)
    act_rumor = np.array(act_rumor)
    
    acc, pre, recall, f_score, f_0, f_1 = cal_scores(pre_rumor, act_rumor)
    logger.info('f_0: %.3f f_1: %.3f ' % (f_0, f_1))
    
    return acc / len(act_rumor), f_score, test_loss

def cal_scores(pred_label, true_label):
    acc = (pred_label == true_label).sum()
    pre, recall, f_score, _  = precision_recall_fscore_support(true_label, pred_label, average = 'macro')

    _, _, f_0, _ = precision_recall_fscore_support(true_label, pred_label, average = 'macro', labels = [0])
    _, _, f_1, _ = precision_recall_fscore_support(true_label, pred_label, average = 'macro', labels = [1])

    return acc, pre, recall, f_score, f_0, f_1
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #cuda
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    #optimizer
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="The initial weight decay for Adam.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for Adam.") 
    #training setting
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--bert_path', required=True)
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--dataset', required=True)
    
    parser.add_argument("--batchsize", default=128, type=int, help="Total batch size for training.")
    parser.add_argument("--max_words", default=200, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--num_train_epochs", default=50, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--image_size", default=224, type=int, help='Number of image_size.')
    parser.add_argument("--patch_size", default=32, type=int, help='Number of patch_size.')
    #model setting
    parser.add_argument('--embed_size', type=int, default=256, help='Dropout.')
    #textEncoder
    parser.add_argument('--bert_hid', type=int, default=768, help='Dropout.')
    parser.add_argument('--num_layers', type=int, default=1, help='Dropout.')
    parser.add_argument('--freeze_textencoder', action='store_true', default=True, help='Dropout.')
    parser.add_argument('--txtnorm', action='store_true', default=True, help='Dropout.')
    #imageEncoder
    parser.add_argument('--vit_hid', type=int, default=768, help='Dropout.')
    parser.add_argument('--freeze_imgencoder', action='store_true', default=True, help='Dropout.')
    parser.add_argument('--imgnorm', action='store_true', default=True, help='Dropout.')
    #attention
    parser.add_argument('--att_hid', type=int, default=2048, help='Dropout.')
    parser.add_argument('--n_head', type=int, default=16, help='Dropout.')
    parser.add_argument('--thres', type=float, default=0.0, help='Dropout.')
    
    parser.add_argument('--h_dim', type=int, default=64, help='Dropout.')
    parser.add_argument('--expert_hid', type=int, default=128, help='Dropout.')
    parser.add_argument('--lambda_softmax', type=int, default=11, help='Dropout.')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Dropout.')
    parser.add_argument('--tradeoff', type=float, default=1.0, help='Dropout.')
    
    parser.add_argument('--text_aggregation', default='cnn', help='Dropout.')
    parser.add_argument('--image_aggregation', default='fc', help='Dropout.')
    
    args = parser.parse_args()
    config = vars(args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
   
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    
    writer = SummaryWriter(os.path.join(args.outdir, 'runs'), flush_secs=1)
    
    train_loader, test_loader = get_Dataloader(args)

    model = DetectionModule(args, config)
        
    model = model.cuda()
    training(model, args, train_loader, test_loader, writer)
