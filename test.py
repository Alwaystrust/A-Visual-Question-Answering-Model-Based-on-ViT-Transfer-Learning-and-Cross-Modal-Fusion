import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from position_emb import prepare_graph_variables
import torch.nn.functional as F



def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


@torch.no_grad()
def test(model, dataloader, model_method):
    score = 0
    upper_bound = 0
    num_data = 0
    
    start = 0
    for v, q, a in iter(dataloader):
        start += 1
        v, q = v.cuda(non_blocking=True),q.cuda(non_blocking=True)
        v = Variable(v)
        q = Variable(q)
        
        if model_method in ['zero','ban','exchange']:
            pred = model(v, q, None)
            
        if model_method in ['reattention']:
            pred, _, _, _ = model(v, q, a)
        
        if model_method in ['Self_Supervised']:
            pred, _ = model(q, v, False)

        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score

        num_data += pred.size(0)
  
    print(num_data)
    score = score / len(dataloader.dataset)

    return score
