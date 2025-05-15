# -*- coding: utf-8 -*-

from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

from encoder import Encoder


class Model(nn.Module):
    def __init__(self, emb_method, enc_method, word_dim, hidden_size, out_size, method = 'zero'):

        super(Model, self).__init__()

        self.emb_method = emb_method
        self.enc_method = enc_method
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.word_dim = word_dim
        self.method = method
        #这边定义的dropout经常会导致问题出现，感觉是在init_xx函数里定义的dropout会和 __init__函数里定义的有冲突，顺序我也不太清楚。
        
        #if self.emb_method == 'elmo':
        #    self.init_elmo()
        #elif self.emb_method == 'glove':
        #   self.init_glove()
        #elif self.emb_method == 'bert':
        #    self.init_bert()

        self.encoder = Encoder(self.enc_method, self.word_dim, self.hidden_size, self.out_size, self.method)

    def forward(self, x):
        if self.emb_method == 'elmo':
            word_embs = self.get_elmo(x)
        elif self.emb_method == 'glove':
            word_embs = self.get_glove(x)
        elif self.emb_method == 'bert':
            word_embs = self.get_bert(x)
        x = self.encoder(word_embs)
        
        #x = self.dropout(x)
        return x

    def init_bert(self,bert_file,train_flag):
        '''
        initilize the Bert model
        '''
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_file)
        self.bert = AutoModel.from_pretrained(bert_file)
        for param in self.bert.parameters():
            #可训练带来的结果很灾难 还是冻结比较好
            param.requires_grad = train_flag
        
    def get_bert(self, sentence_lists):
        '''
        get the bert word embedding vectors for a sentences
        '''
        embeddings = self.bert(sentence_lists)
        return embeddings[0]  
   
    
    def init_glove(self,np_file,ntoken):
        '''
        load the GloVe model
        '''
        self.glove = nn.Embedding(ntoken+1, 300, padding_idx=ntoken)
        self.ntoken = ntoken

        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, 300)
        self.glove.weight.data[:self.ntoken] = weight_init

    def get_glove(self,x):
        '''
        get the glove word embedding vectors for a sentences
        '''
        emb = self.glove(x)
        drop = nn.Dropout(0.0)
        emb = drop(emb)

        return emb

    
    
    
    #待完善
    def init_elmo(self):
        #self.elmo = Elmo(self.opt.elmo_options_file, self.opt.elmo_weight_file, 1)
        self.elmo = Elmo("./data/elmo/elmo_2x2048_256_2048cnn_1xhighway_options.json", "./data/elmo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5", 1)
        for param in self.elmo.parameters():
            param.requires_grad = False
        self.word_dim = 512
    def get_elmo(self, sentence_lists):
        #get the ELMo word embedding vectors for a sentences
        character_ids = batch_to_ids(sentence_lists)
        character_ids = character_ids.to("cuda")
        embeddings = self.elmo(character_ids)
        return embeddings['elmo_representations'][0]