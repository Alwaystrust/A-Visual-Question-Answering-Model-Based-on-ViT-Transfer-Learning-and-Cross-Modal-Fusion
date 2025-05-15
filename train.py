import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from position_emb import prepare_graph_variables
import torch.nn.functional as F


import torch
from torch import nn
from torch.nn import functional as F

def convert_sigmoid_logits_to_binary_logprobs(logits):
    """Computes log(sigmoid(logits)), log(1-sigmoid(logits))."""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob
def cross_entropy_loss(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels / 10
    if 'miu' in kwargs:
        loss = loss * smooth(kwargs['miu'], kwargs['mask'])
    return loss.sum(dim=-1).mean()
def elementwise_logsumexp(a, b):
    """computes log(exp(x) + exp(b))"""
    return torch.max(a, b) + torch.log1p(torch.exp(-torch.abs(a - b)))
def renormalize_binary_logits(a, b):
    """Normalize so exp(a) + exp(b) == 1"""
    norm = elementwise_logsumexp(a, b)
    return a - norm, b - norm
def smooth(miu, mask):
    miu_valid = miu * mask
    miu_invalid = miu * (1.0 - mask) # most 1.0
    return miu_invalid + torch.clamp(F.softplus(miu_valid), max=100.0)


class Plain(nn.Module):
    def forward(self, logits, labels, **kwargs):
        if config.loss_type == 'ce':
            loss = cross_entropy_loss(logits, labels, **kwargs)


class LearnedMixin(nn.Module):
    def __init__(self, hid_size=1024, smooth=True, smooth_init=-1, constant_smooth=0.0):
        """
        w: Weight of the entropy penalty
        smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        smooth_init: How to initialize `a`
        constant_smooth: Constant to add to the bias to smooth it
        """
        super(LearnedMixin, self).__init__()
        self.smooth_init = smooth_init
        self.constant_smooth = constant_smooth
        self.bias_lin = torch.nn.Linear(hid_size, 1)
        self.smooth = smooth
        if self.smooth:
            self.smooth_param = torch.nn.Parameter(
                smooth_init * torch.ones((1,), dtype=torch.float32))
        else:
            self.smooth_param = None

    def bias_convert(self, **kwargs):
        factor = self.bias_lin.forward(kwargs['hidden'])  # [batch, 1]
        factor = F.softplus(factor)

        bias = torch.stack([kwargs['bias'], 1 - kwargs['bias']], 2)  # [batch, n_answers, 2]

        # Smooth
        bias += self.constant_smooth
        if self.smooth:
            soften_factor = torch.sigmoid(self.smooth_param)
            bias = bias + soften_factor.unsqueeze(1)

        bias = torch.log(bias)  # Convert to logspace

        # Scale by the factor
        # [batch, n_answers, 2] * [batch, 1, 1] -> [batch, n_answers, 2]
        bias = bias * factor.unsqueeze(1)
        return bias

    def loss_compute(self, logits, labels, bias_converted, **kwargs):
        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
        log_probs = torch.stack([log_prob, log_one_minus_prob], 2)

        # Add the bias in
        logits = bias_converted + log_probs

        # Renormalize to get log probabilities
        log_prob, log_one_minus_prob = renormalize_binary_logits(logits[:, :, 0], logits[:, :, 1])

        # Compute loss
        loss_single = -(log_prob * labels + (1 - labels) * log_one_minus_prob)
        if 'miu' in kwargs:
            loss_single = loss_single * smooth(kwargs['miu'], kwargs['mask'])
        loss = loss_single.sum(1).mean(0)
        return loss

    def forward(self, logits, labels, **kwargs):
        bias_converted = self.bias_convert(**kwargs)
        loss = self.loss_compute(logits, labels, bias_converted, **kwargs)
        return loss


class LearnedMixinH(LearnedMixin):
    def __init__(self, hid_size=1024, smooth=True, smooth_init=-1, constant_smooth=0.0, w=0.36):
        """
        w: Weight of the entropy penalty
        smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        smooth_init: How to initialize `a`
        constant_smooth: Constant to add to the bias to smooth it
        """
        super(LearnedMixinH, self).__init__(hid_size, smooth, smooth_init, constant_smooth)
        self.w = w

    def forward(self, logits, labels, **kwargs):
        bias_converted = self.bias_convert(**kwargs)
        loss = self.loss_compute(logits, labels, bias_converted, **kwargs)

        # Re-normalized version of the bias
        bias_norm = elementwise_logsumexp(bias_converted[:, :, 0], bias_converted[:, :, 1])
        bias_logprob = bias_converted - bias_norm.unsqueeze(2)

        # Compute and add the entropy penalty
        entropy = -(torch.exp(bias_logprob) * bias_logprob).sum(2).mean()
        return loss + self.w * entropy

def FocalLoss(logits, labels):
    alpha = 0.25
    gamma=2.0
    epsilon=1e-9
    
    logits = F.softmax(logits, dim=-1)
    ce_loss = - (labels * torch.log(logits)).sum(dim=-1)
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt)**gamma * ce_loss
    loss = loss.max()
    return loss
    
def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


## 双注意力的损失需要的补充函数
def classification_loss(logits, labels):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss
def attention_consistency_loss(attention_weights, reattention_weights):
    return torch.sum(torch.pow(attention_weights - reattention_weights,2))
def calculate_loss(logits,labels,was_reattended,attention_weights,reattention_weights,reattention_factor=0.8):
    return classification_loss(logits, labels) + (reattention_factor * attention_consistency_loss(attention_weights, reattention_weights) if was_reattended else 0), (attention_consistency_loss(attention_weights, reattention_weights) if was_reattended else 0)


## 自监督的损失需要的补充函数
def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits_neg,dim=-1), 1, top_ans_ind).sum(1)
    
    qice_loss = neg_top_k.mean()
    return qice_loss



def train(model, train_loader, eval_loader, num_epochs, output, model_method):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()         

        for i, (v, q, a) in enumerate(train_loader):
            v, q, a = v.cuda(non_blocking=True),q.cuda(non_blocking=True),a.cuda(non_blocking=True)
            #v2 = v2.cuda(non_blocking=True)
            v = Variable(v)
            #v2 = Variable(v2)
            q = Variable(q)
            a = Variable(a)
            #v = torch.cat((v,v2),1)
            if model_method in ['zero','ban','exchange']:
                pred = model(v,q,a)
                loss = instance_bce_with_logits(pred, a)
                #loss = FocalLoss(pred,a)
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()
                batch_score = compute_score_with_logits(pred, a.data).sum()
                total_loss += loss.item() * v.size(0)
                train_score += batch_score

            if model_method in ['reattention']:
                #with torch.cuda.amp.autocast():
                pred, v_att, v_re_att, _ = model(v, q, a)
                loss, att_loss = calculate_loss(pred,a,True,v_att,v_re_att)
                #scaler.scale(loss).backward()
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 0.25)
                #scaler.step(optim)
                optim.step()
                optim.zero_grad()
                #scaler.update()
                batch_score = compute_score_with_logits(pred, a.data).sum()
                total_loss += loss.item() * v.size(0)
                train_score += batch_score

            if model_method in ['Self_Supervised']:
                            # for the labeled samples
                pretrain_epoches = 12
                if epoch < pretrain_epoches:
                    #with torch.cuda.amp.autocast():
                    logits_pos, _= model(q, v, False)
                    bce_loss_pos = instance_bce_with_logits(logits_pos, a)
                    #bce_loss_pos = instance_bce(logits_pos, a)
                    loss = bce_loss_pos
                else:
                    #with torch.cuda.amp.autocast():
                    logits_pos, logits_neg, _, _ = model(q, v, True)

                    bce_loss_pos = instance_bce_with_logits(logits_pos, a)
                    #bce_loss_pos = instance_bce(logits_pos, a)

                    self_loss = compute_self_loss(logits_neg, a)

                    loss = bce_loss_pos + 3 * self_loss
                    #3是权重

                #scaler.scale(loss).backward()
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 0.25)
                #scaler.step(optim)
                optim.step()
                optim.zero_grad()
                #scaler.update()
                batch_score = compute_score_with_logits(logits_pos, a.data).sum()
                train_score += batch_score.item()
                total_loss += loss.item() * v.size(0)

                if epoch < pretrain_epoches: #pretrain
                    total_self_loss = 0
                    train_score_neg = 0
                else: #fintune
                    score_neg = compute_score_with_logits(logits_neg, a.data).sum()
                    total_self_loss += self_loss.item() * v.size(0)
                    train_score_neg += score_neg.item()

            if model_method in ['Graph']:
                pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
                'implicit', bb, sem_adj_matrix, spa_adj_matrix, num_objects,
                20, 64, 11,
                15, device)
                pred, att = model(v, b, q, pos_emb, sem_adj_matrix,
                              spa_adj_matrix, a)

                loss = instance_bce_with_logits(pred, a)
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()
                batch_score = compute_score_with_logits(pred, a.data).sum()
                total_loss += loss.item() * v.size(0)
                train_score += batch_score  

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader, model_method)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
            
            
            
            
@torch.no_grad()
def evaluate(model, dataloader, model_method):
    score = 0
    upper_bound = 0
    num_data = 0
    for (v,q, a) in iter(dataloader):
        v, q = v.cuda(non_blocking=True),q.cuda(non_blocking=True)
        #v2 = v2.cuda(non_blocking=True)
        v = Variable(v)
        #v2 = Variable(v2)
        q = Variable(q)
        #v = torch.cat((v,v2),1)
        if model_method in ['zero','ban','exchange']:
            pred = model(v,q, None)

        if model_method in ['reattention']:
            pred, _, _, _ = model(v, q, a)
        
        if model_method in ['Self_Supervised']:
            pred, _ = model(q, v, False)

        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound

