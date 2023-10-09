import torch 
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()        
        # 1D-CNN
        Ks = [1, 2, 3]
        in_channel = 1
        out_channel = args.embed_size
        self.fc = nn.Linear(args.bert_hid, args.embed_size)
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, args.bert_hid)) for K in Ks])
        self.mapping = nn.Linear(len(Ks)*out_channel, args.embed_size)       

    def forward(self, input):
        x = input.unsqueeze(1)  
        x_emb = self.fc(input)
        x1 = F.relu(self.convs1[0](x)).squeeze(3)  
        x2 = F.relu(self.convs1[1](F.pad(x, (0, 0, 0, 1)))).squeeze(3)
        x3 = F.relu(self.convs1[2](F.pad(x, (0, 0, 1, 1)))).squeeze(3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.transpose(1, 2)  
        word_emb = self.mapping(x)
        word_emb = word_emb + x_emb
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in [x1, x2, x3]]
        x = torch.cat(x, 1)

        txt_emb = self.mapping(x)
        txt_emb = txt_emb + x_emb.mean(1)
        
        return word_emb, txt_emb
        