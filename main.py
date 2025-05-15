import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,ConcatDataset
import numpy as np
import random
from dataset import Dictionary, VQAFeatureDataset, VisualGenomeFeatureDataset
#from dataset_mini import Dictionary, VQAFeatureDataset
#from dataset_person import Dictionary, VQAFeatureDataset
import base_model
from train import train
from test import test
import utils
import gc
import sys





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    #不能设置2048 又慢又差
    parser.add_argument('--model', type=str, default='zero')
    parser.add_argument('--output', type=str, default='saved_models/fasterrcnn_1024')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument('--emb', default='glove', type=str,help='embbeding method including bert glove elmo')
    parser.add_argument('--enc', default='rnn', type=str,help='encoding method including cnn rnn transformer')
    parser.add_argument('--data', default='fasterrcnn', type=str,help='data preprocessing method including fasterrcnn maskrcnn revisit solo_sig solo_no_sig')
    parser.add_argument('--gamma', type=int, default=3, help='glimpse')
    parser.add_argument('--fm', type=str, default='zero', help='fusion method')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    #args.output = args.output.split('/')[0] + '/' + args.data
    print(args.output)
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    #torch.backends.cudnn.benchmark = True
    
    '''
    In this part we we will build the dataset we need in the following task. Using the dataset.py-class VQAFeatureDataset to get train-and-val dataset.
    we need to feed in dataset-type(train/val) and dataset_source(fasterrcnn/maskrcnn/revisit) and Dictionary(this is in the dataset.py,too. This dictionary includes the information of the index of all words and the list of all words， so that we need feed this into VQADataset-class to convenience the processing of the sentence-tokenizing)
    Finally we will get an class (it' s name is train_dset and eval_dset, is't not a big dataset), then if we need to get some data， the VQAFeatureDataset has an function， naming __getitem__,this function will return part data we need.
    '''
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, args.data, args.emb)
    eval_dset = VQAFeatureDataset('val', dictionary, args.data, args.emb)
    batch_size = args.batch_size
    train_loader = DataLoader(train_dset, batch_size, num_workers=4, shuffle=False, pin_memory=True, persistent_workers = True)
    eval_loader =  DataLoader(eval_dset, batch_size, num_workers=4, shuffle=False, pin_memory=True, persistent_workers = True)
    

    
    #train_loader_list = [train_loader]
    #eval_loader_list = [eval_loader]
    
    #if args.data != 'fasterrcnn':
    #    train_dset2 = VQAFeatureDataset('train', dictionary, 'solo_faster_background_12', args.emb)
    #    eval_dset2 = VQAFeatureDataset('val', dictionary, 'solo_faster_background_12', args.emb)
        
    #    train_loader2 = DataLoader(train_dset2, batch_size, num_workers=4, shuffle=False, pin_memory=True, persistent_workers = True)
    #    eval_loader2 =  DataLoader(eval_dset2, batch_size, num_workers=4, shuffle=False, pin_memory=True, persistent_workers = True)
    
    #    train_loader_list.append(train_loader2)
    #    eval_loader_list.append(eval_loader2)
    # constructor is a choose of backbone of this model，although i don't find any difference between build_baseline0 and build_baseline0_newatt.
    #"getattr" is an advanced method. it' s means that (train_dset, args.num_hid, args.emb, args.enc) as input to constructor. then the output of constructor as input to base_model.
    constructor = 'build_%s' % args.model
    
    model = getattr(base_model, constructor)(train_dset, args.num_hid, args.emb, args.enc,args.fm, args.gamma).cuda()
    '''
    这一步提前把初始化的东西赋值进这个网络里面，对于原始的方法，我们输入的是索引，即每一个句子已经变成了[1,231,4341,7435,42,546,...]这种，然后w_emb网络的作用是把每个句子 1*14的索引向量变成 300*14的数值向量，而w_emb简单来说可以看作是一个查询矩阵，就是会生成一个单词数*300的矩阵（初始化通过下面指定的文件glove_init_300d.npy），换言之，如果设置nn.embbding为frozen，那么实际上w_emb就是一个查表，把索引改向量，而nn.embedding可以接入训练过程进行微调。nn.embedding的输入只能是编号.
    其实就是glove做初始化的权重参数，然后在后续的网络中对词向量进行fine-tune。
    需要再次核查的是，这里的索引和单词的关系用的是data/dictionary.pkl文件里的对应关系。是否其他模型也是这样需要核查。
    另：当我把下面这句注释掉之后，模型也能跑，只不过此刻的词向量就是随机初始化了。效果也确实很差。
    '''
    model.w_emb.init_embedding('data/glove/glove_300d.npy')
    model.model_net.init_glove('data/glove/glove_300d.npy',train_dset.dictionary.ntoken)
    #model.model_net.init_bert('data/bert',False)
    model = nn.DataParallel(model).cuda()
    
    #这里可以加载之前训练到一半的一些权重。
    #ckpt = torch.load('saved_models/revisit_reattention_test/model.pth', map_location='cuda:0')
    #model.load_state_dict(ckpt)

    #vg_train = [VisualGenomeFeatureDataset('train', train_dset.features, train_dset.spatials, dictionary, args.emb, 'data/' + args.data)]
    #vg_eval = [VisualGenomeFeatureDataset('val', eval_dset.features, eval_dset.spatials, dictionary, args.emb, 'data/' + args.data)]

    #他们有的为了扩充数据 就是把训练集和验证集直接全部合并成训练集训练。
    #train_dset = ConcatDataset([train_dset]+vg_train)
    #eval_dset = ConcatDataset([eval_dset]+vg_eval)
    



    #for i, (v, q, a) in enumerate(train_loader):
    #    torch.save(v,'saved_vectors/solo_no_sig_' + str(i) + '.pt')
    #    if i == 10:
    #        break
    
    train(model, train_loader, eval_loader, args.epochs, args.output,args.model)

    
    #ckpt = torch.load(args.resume, map_location=device)
    #last_epoch = ckpt['epoch']
    #model.load_state_dict(ckpt['state_dict'])
    #test(model, eval_loader, args.model)
