from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import tools.compute_softscore


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()
    

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        #现在出现了这样一个问题，我想用上数据扩充这一步，即把visual_genome用上，里面其实就是一些新的问题和回答，可以作为VQA数据集的一个扩充（额外的数据集）。然后但是在最初的create_dictionary文件里，并没有把vg数据集里的question_answers.json用上，这就导致在向量化或者tokenize的时候，vg里的有些词在已经保存在本地的dictionary里是没有的。
        else:
            for w in words:
                #所以是否需要在这里补一个，如果没有，那么就continue，也就是咱也不去搞修改词典，我们就是tokenize的时候如果词典里没有这个词，就pass，就找有的。
                try:
                    tokens.append(self.word2idx[w])
                except:
                    continue
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val,label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        answer_my = [label2ans[i] for i in answer['labels']]
        flag = 1
        for i in answer_my:
            if i not in ['1', '2', '3', '5', '4', '0', '6', '24', '10', '7', '13', '50', '11', '8', '30', '31', '55', '14', '12', '66', '46', '15', '2000', '9', '25', '68', '70', '45', '44', '35', '18', '20', '16', '200', '36', '300', '40', '2008', '34', '29', '33', '2012', '17', '100', '19', '23', '37', '60', '1990', '53', '22', '21', '1950', '2015', '75', '500', '2010', '2016', '61', '27', '32', '88', '1000', '28', '193', '42', '48', '700', '56', '150', '2013', '2007', '51', '38', '120', '52', '39', '600', '90', '350', '2009', '2011', '80', '43', '26', '106', '41', '400', '47', '870', '101', '72', '64', '65', '54', '1980', '59', '99', '49', 'little', 'double', 'several', 'many', 'lot', 'lots', 'hundreds', 'thousands', 'millions']:
                #上面的数量标签获取是从 vqa-cp_v2.ipynb里获取的。
                flag = 0
                break
        if len(answer_my) > 0 and flag == 1:
            entries.append(_create_entry(img_id2val[img_id], question, answer))
                
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, data_name, method, dataroot='data'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        
        
        '''
        获取指定数据集（fasterrcnn方法、maskrcnn方法、revisit方法）里的图像特征和图像索引。
        图像特征放在hdf5文件里，hf.get('image_features')可以获取。图像id和序号的对应关系存储在pkl文件里。
        图像id指的是其在coco数据集里的实际的编号（图片名coco_train_000000123这种），序号指的是从处理图像开始按照处理的顺序编号1，2，3......
        '''
        assert data_name in ['maskrcnn','revisit','fasterrcnn','solo_sig','solo_no_sig','solo_part','solo_faster','solo_faster_2','fasterrcnn_1024','fasterrcnn_1024_2']
        print('loading %s features from h5 file'%(data_name))
        if data_name == 'maskrcnn':
            path = os.path.join(dataroot,'maskrcnn')
        if data_name == 'revisit':
            path = os.path.join(dataroot,'revisit')
        if data_name == 'fasterrcnn':
            path = os.path.join(dataroot,'fasterrcnn')
        if data_name == 'solo_sig':
            path = os.path.join(dataroot,'solo_sig')
        if data_name == 'solo_no_sig':
            path = os.path.join(dataroot,'solo_no_sig')    
        if data_name == 'solo_part':
            path = os.path.join(dataroot,'solo_part')
        if data_name == 'fasterrcnn_1024':
            path = os.path.join(dataroot,'fasterrcnn_1024')
        if data_name == 'fasterrcnn_1024_2':
            path = os.path.join(dataroot,'fasterrcnn_1024_2')
        if data_name == 'solo_faster':
            path = os.path.join(dataroot,'solo_faster')
        if data_name == 'solo_faster_2':
            path = os.path.join(dataroot,'solo_faster_2')            
        with open(os.path.join(path, '%s36_imgid2idx.pkl' % name), 'rb') as f:
            self.img_id2idx = pickle.load(f)
            
        h5_path = os.path.join(path, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            #spatials咱用不上呀 先注释掉
            #self.spatials = np.array(hf.get('spatial_features'))

        '''
        这一步是获取 问题和答案对，同时这俩都有imageid属性，根据这个校准防止错位，最终得到一个 imageid-image-question-answer的entries
        调用_load_dataset和_create_entry函数。
        
        输入地址和trian/val和图片id序号对应索引，输出如下结构的entry
        [{'question_id': 9000,
        'image_id': 9,
        'image': 52181,
        'question': 'How many cookies can be seen?',
        'answer': {'labels': [17], 'scores': [1]}} .....]
        image_id是图片真实的名称，image是每个数据集都不同的按照预处理顺序生成的0，1，2，3...的序号。
        '''
        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans)
        
        
        #这一步仅仅针对entries里的question做处理，entry['question']是诸如[”What color are the dishes?“]这种，我们需要根据字典里的东西进行编码，得到[1,324,5423,32,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1].然后赋值到entries['q_token']里。
        self.tokenize(method)
        
        
        #这一步是将所有的数据转换成pytorch的数据结构，因为从hdf5还是从pkl等地方读取到的数据都是编码状态的，我们需要通过numpy的方式解码，然后得到numpy的矩阵，再在此基础上将其转换为pytorch的数据结构。
        self.tensorize()
        
        
        
        self.v_dim = self.features.size(2)
        #self.s_dim = self.spatials.size(2)

    def tokenize(self, method = 'glove', max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        assert method in ['glove','bert','elmo']
        if method == 'glove':
            for entry in self.entries:
                tokens = self.dictionary.tokenize(entry['question'], False)
                tokens = tokens[:max_length]
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                    tokens = padding + tokens
                utils.assert_eq(len(tokens), max_length)
                entry['q_token'] = tokens
        elif method == 'bert':
            #特别提醒：bert的句向量是28个
            
            
            #对于bert也是有相应的tokenize的方式，只是不同于golve，glove是纯粹的排号，反正给每一个词一个号码牌，而bert的话会给每个句子开头结尾都加上一个token，总之就是tokenize的方式不同，我们集成到dataset.py函数里来实现。
            #tips貌似bert_tokenizer适合并行计算，所以我们先循环取出全部的句子，然后统一处理，然后再循环放回去。
            bert_tokenizer = AutoTokenizer.from_pretrained('./data/bert/')
            bert = AutoModel.from_pretrained('./data/bert/')
            temp = []
            for entry in self.entries:
                temp.append(entry['question'])
            
            ids = bert_tokenizer(temp, padding=True, return_tensors="pt")
            inputs_list = ids['input_ids']
            
            for entry,inputs in zip(self.entries,inputs_list):                
                entry['q_token'] = inputs[:24] + inputs[-1:]

        else:
            #elmo的待完成，貌似他的token和模型是和在一起的，我看都是直接输入句子，然后输出就是句向量。
            pass
            

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        #self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        #spatials = self.spatials[entry['image']]
        #question = entry['question']
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, question, target
        #return features, spatials, question, target
        #return features, question, target

    def __len__(self):
        return len(self.entries)

    

    
COUNTING_ONLY = False
# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False
    
    
    

def _load_visualgenome(dataroot, name, img_id2val, label2ans, adaptive=False):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(dataroot, 'question_answers.json')
    image_data_path = os.path.join(dataroot, 'image_data.json')
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    cache_path = os.path.join(dataroot, 'cache', 'vg_%s%s_target.pkl' % (name, '_adaptive' if adaptive else ''))

    if os.path.isfile(cache_path):
        entries = pickle.load(open(cache_path, 'rb'))
    else:
        entries = []
        ans2label = pickle.load(open(ans2label_path, 'rb'))
        vgq = json.load(open(question_path, 'r'))
        _vgv = json.load(open(image_data_path, 'r')) #108,077
        vgv = {}
        for _v in _vgv: 
            if None != _v['coco_id']:
                vgv[_v['image_id']] = _v['coco_id']
        counts = [0, 0, 0, 0] # used image, used question, total question, out-of-split
        for vg in vgq:
            coco_id = vgv.get(vg['id'], None)
            if None != coco_id:
                counts[0] += 1
                img_idx = img_id2val.get(coco_id, None)
                if None == img_idx:
                    counts[3] += 1
                for q in vg['qas']:
                    counts[2] += 1
                    _answer = tools.compute_softscore.preprocess_answer(q['answer'])
                    label = ans2label.get(_answer, None)
                    if None != label and None != img_idx:
                        counts[1] += 1
                        answer = {
                            'labels': [label],
                            'scores': [1.]}
                        entry = {
                            'question_id' : q['qa_id'],
                            'image_id'    : coco_id,
                            'image'       : img_idx,
                            'question'    : q['question'],
                            'answer'      : answer}
                        if not COUNTING_ONLY or is_howmany(q['question'], answer, label2ans):
                            entries.append(entry)

        print('Loading VisualGenome %s' % name)
        print('\tUsed COCO images: %d/%d (%.4f)' % \
            (counts[0], len(_vgv), counts[0]/len(_vgv)))
        print('\tOut-of-split COCO images: %d/%d (%.4f)' % \
            (counts[3], counts[0], counts[3]/counts[0]))
        print('\tUsed VG questions: %d/%d (%.4f)' % \
            (counts[1], counts[2], counts[1]/counts[2]))
        with open(cache_path, 'wb') as f:
            pickle.dump(entries, open(cache_path, 'wb'))

    return entries

class VisualGenomeFeatureDataset(Dataset):
    def __init__(self, name, features, spatials, dictionary,method, dataroot='data', adaptive=False, pos_boxes=None):
        super(VisualGenomeFeatureDataset, self).__init__()
        # do not use test split images!
        assert name in ['train', 'val']
        temp_root = 'data'
        ans2label_path = os.path.join(temp_root, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(temp_root, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive
        
        self.img_id2idx = pickle.load(
                open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))

        self.features = features
        self.spatials = spatials
        if self.adaptive:
            self.pos_boxes = pos_boxes

        self.entries = _load_visualgenome(temp_root, name, self.img_id2idx, self.label2ans)
        self.tokenize(method)
        self.tensorize()
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.spatials.size(1 if self.adaptive else 2)

        
    def tokenize(self, method = 'glove', max_length=14):
        assert method in ['glove','bert','elmo']
        if method == 'glove':
            for entry in self.entries:
                tokens = self.dictionary.tokenize(entry['question'], False)
                tokens = tokens[:max_length]
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                    tokens = padding + tokens
                utils.assert_eq(len(tokens), max_length)
                entry['q_token'] = tokens
        elif method == 'bert':
            bert_tokenizer = AutoTokenizer.from_pretrained('./data/bert/')
            bert = AutoModel.from_pretrained('./data/bert/')
            temp = []
            for entry in self.entries:
                temp.append(entry['question'])
            
            ids = bert_tokenizer(temp, padding=True, return_tensors="pt")
            inputs_list = ids['input_ids']
            
            for entry,inputs in zip(self.entries,inputs_list):                
                entry['q_token'] = inputs[:24] + inputs[-1:]

        else:
            pass

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        return features, spatials, question, target

    def __len__(self):
        return len(self.entries)
    
    
    
    
    
#下面我们将上述的两个构造数据集类合并。因为我们的那个狗bert的词向量数是不固定的。
