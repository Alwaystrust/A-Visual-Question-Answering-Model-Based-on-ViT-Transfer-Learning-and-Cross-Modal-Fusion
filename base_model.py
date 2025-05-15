import torch
import torch.nn as nn
from attention import Attention,Attention2, NewAttention,BiAttention,ReAttention, SelfAttention,Att_3
from language_model import WordEmbedding, QuestionEmbedding, WordEmbedding2, QuestionEmbedding2,QuestionSelfAttention
from classifier import SimpleClassifier
from fc import FCNet
import torch.nn.functional as F
from bc import BCNet
from counting import Counter
from multi_layer_net import MultiLayerNet
from fusion import BAN, BUTD, MuTAN                               
from relation_encoder import ImplicitRelationEncoder,ExplicitRelationEncoder
import random
from conv import Exchange, ModuleParallel, BatchNorm1dParallel, Bottleneck
from model import Model
import numpy as np
import scipy.signal
import numpy as np
'''
class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, z_net,model_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.z_net = z_net
        self.model_net = model_net
        self.classifier = classifier

    def forward(self, v, q, labels):
    #def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #w_emb = self.w_emb(q)
        #q_emb = self.q_emb(w_emb) # [batch, q_dim]
        q_emb = self.model_net(q)
        
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        
        #joint_repr = self.z_net(q_repr - v_repr)
        joint_repr = q_repr * v_repr
        ##
        logits = self.classifier(joint_repr)
        return logits
        
def build_baseline0_newatt(dataset, num_hid, emb, enc):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    z_net = FCNet([num_hid, num_hid])
    if emb == 'glove':
        word_len = 300
    elif emb == 'bert':
        word_len = 768
    else:
        word_len = 512
    model_net = Model(emb, enc, word_len, num_hid, num_hid)
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, z_net,model_net, classifier)
'''

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_prj, v_prj, model_net, gamma, fm, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier
        self.glimpse = gamma
        self.q_prj = nn.ModuleList(q_prj)
        self.v_prj = nn.ModuleList(v_prj)
        self.model_net = model_net
        self.fusion_method = fm
        #self.two_prj = nn.ModuleList(two_prj)
        #self.randsort = randsort
        
    def forward(self, v, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #w_emb = self.w_emb(q)
        #q_emb = self.q_emb(w_emb) # [batch, q_dim]
        q_emb = self.model_net(q)
        
        #print(q_emb.shape)
        
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        #v_emb2 = (att * v2).sum(1) # [batch, v_dim]
        #att2 = self.v_att2(v2, q_emb)
        #v_emb3 = (att2 * v2).sum(1) # [batch, v_dim]        
        #print(v_emb3.shape)
        #att3 = self.my_att1(v1,v_emb3)
        #v_emb1 = (att3 * v1).sum(1)
        #att4 = self.my_att2(v2,v_emb2)
        #v_emb0 = (att4 * v2).sum(1)
        
        #v_emb = v_emb3
        
        joint_repr = 0
        for g in range(self.glimpse):
            q_repr = self.q_prj[g](q_emb)
            
            v_repr = self.v_prj[g](v_emb)
            #v_repr2 = self.v_prj[g](v_emb2)
            joint_repr += q_repr * v_repr
            #joint_repr += q_repr * v_repr + q_repr * v_repr2

        logits = self.classifier(joint_repr)
        return logits


#原版
def build_zero(dataset, num_hid,emb, enc,fm, gamma=8):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    #print(q_emb.num_hid)
    #print(dataset.v_dim)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    #v_att2 = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    #my_att1 = NewAttention(dataset.v_dim, dataset.v_dim, num_hid)
    #my_att2 = NewAttention(dataset.v_dim, dataset.v_dim, num_hid)
    
    #加一下随机的一个排列
    #randsort = np.arange(1024)
    #random.shuffle(randsort)
    
    q_prj = []
    v_prj = []
    #two_prj = []
    objects = 36  # minimum number of boxes
    for i in range(gamma):
        q_prj.append(FCNet([q_emb.num_hid, num_hid]))
        v_prj.append(FCNet([dataset.v_dim, num_hid]))    
        #two_prj.append(FCNet([dataset.num_ans_candidates, num_hid]))
    if emb == 'glove':
        word_len = 300
    elif emb == 'bert':
        word_len = 768
    else:
        word_len = 512
    
    model_net = Model(emb, enc, word_len, num_hid, num_hid)

    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    
    return BaseModel(w_emb, q_emb, v_att,q_prj, v_prj,model_net, gamma,fm, classifier)

    
class BanModel(nn.Module):
    def __init__(self, dataset, v_att, b_net, q_prj, c_prj, classifier,model_net, counter, glimpse):
        super(BanModel, self).__init__()
        self.dataset = dataset
        self.glimpse = glimpse
        self.model_net = model_net
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        q_emb = self.model_net(q)
        #boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h
            
            #atten, _ = logits[:,g,:,:].max(2)
            #embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            #q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifier(q_emb.sum(1))

        return logits
    
def build_ban(dataset, num_hid, emb, enc,fm, gamma=8):
    
    if emb == 'glove':
        word_len = 300
    elif emb == 'bert':
        word_len = 768
    else:
        word_len = 512   
    
    #发现这个ban从rnn编码句子的输出是整个output，而一般的zero方式输出的是output[: -1]，即最后一层。
    model_net = Model(emb, enc, word_len, num_hid, num_hid, 'ban')
    #w_emb = WordEmbedding(dataset.dictionary.ntoken, word_len, .0)
    #q_emb = QuestionEmbedding(word_len, num_hid, 1, False, .0)
    
    #这里的第二个参数非常需要注意。这里的第二个参数对应的是文本模型最终映射得到的维度，即输出维度，即1024，而不是300或者768
    #因为这是接model_net后面的。这个后面就是已经转换成了 1024维了。
    v_att = BiAttention(dataset.v_dim, num_hid, num_hid, gamma)
    
    b_net = []
    q_prj = []
    c_prj = []
    objects = 36  # minimum number of boxes
    for i in range(gamma):
        b_net.append(BCNet(dataset.v_dim, num_hid, num_hid, None, k=1))
        q_prj.append(FCNet([num_hid, num_hid], '', .2))
        
        c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, .5)
    counter = Counter(objects)
    return BanModel(dataset, v_att, b_net, q_prj, c_prj, classifier,model_net, counter, gamma)

class ReattentionModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_att, v_selfatt, q_selfatt, q_prj, v_prj, re_att, classifier, model_net):
        super(ReattentionModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb

        self.q_prj = q_prj
        self.v_prj = v_prj
        
        self.v_selfatt = v_selfatt
        self.q_selfatt = q_selfatt
        
        self.q_att = q_att
        self.v_att = v_att
        
        self.re_att = re_att
        self.classifier = classifier
        self.model_net = model_net
        
    def _get_attented_features(self, attention_input, attention_layer, feature_vector):
        attention_weights = attention_layer(attention_input)
        attended_vector = (attention_weights * feature_vector).sum(1)
        return attended_vector, attention_weights
    
    def calculate_similarity_matrix(self, v_proj, q_proj):
        q_proj = q_proj.transpose(1, 2)
        similarity_matrix = v_proj @ q_proj
        return similarity_matrix

    def forward(self, v, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #w_emb = self.w_emb(q)
        #q_emb = self.q_emb(w_emb) # [batch, q_dim]
        q_emb = self.model_net(q)
        proj_q = self.q_prj(q_emb)
        #proj_q dimention [batch * 14 * 1024]
        
        proj_v = self.v_prj(v)
        #proj_v dimention [batch * 36 * 1024]
        
        self_att_v,att_v_w = self._get_attented_features(proj_v,self.v_selfatt,proj_v)
        #self_att_v输出的维度是[batch * 1024] att_v_w 的维度是 [batch * 36 * 1]
        
        self_att_q,att_q_w = self._get_attented_features(proj_q,self.q_selfatt,proj_q)
        #self_att_q的维度是[batcj * 1024] att_v_q 的维度是 [batch * 14 * 1] 相当于自注意力的话最终的输出是把单元合起来了，成了一个向量。
        
        similarity_matrix = self.calculate_similarity_matrix(proj_v, proj_q)
        #相似性矩阵的大小是 [batch * 36 * 14],计算的是v的36个块块和q的14个单词之间的相关性。
        
        att_q,att_q_w= self._get_attented_features(similarity_matrix.transpose(1, 2),self.q_att,proj_q)
        #把得到的这个相关性矩阵作用到 proj_v和proj_q上面。然后得到新的att_q和att_v
        #att_q 维度为 [batch * 1028] att_q_w 的维度为 [batch * 14 * 1]
        #att_v 维度为 [batch * 1028]
        
        att_v,att_v_w = self._get_attented_features(similarity_matrix,self.v_att,proj_v)

        
        #hadamand融合
        joint_repr = att_q * att_v
        logits = self.classifier(joint_repr)
        
        re_att_v_w = self.re_att(joint_repr,proj_q,proj_v)
        #re_att_v_w dimention [batch * 36 * 1] 没太明白
        
        return (logits, att_v_w,re_att_v_w,att_q_w)

def build_reattention(dataset, num_hid,emb, enc,fm, gamma=1):
    if emb == 'glove':
        word_len = 300
    elif emb == 'bert':
        word_len = 768
    else:
        word_len = 512   
    
    #发现这个ban从rnn编码句子的输出是整个output，而一般的zero方式输出的是output[: -1]，即最后一层。
    model_net = Model(emb, enc, word_len, num_hid, num_hid, 'ban')
    
    w_emb = WordEmbedding2(vocabulary_size=dataset.dictionary.ntoken,embedding_dimension=300,dropout=0.25)
    q_emb = QuestionEmbedding2(input_dimension=300,number_hidden_units=num_hid,number_of_layers=1)
    q_prj = MultiLayerNet(dimensions=[num_hid, num_hid], dropout=0.5)
    #这个v_prj难道可以直接对36个框的输入用嘛 确实是可以的
    v_prj = MultiLayerNet(dimensions=[dataset.v_dim,num_hid,],dropout=0.5)
    
    v_selfatt = SelfAttention(num_hid, dropout=0.3)
    q_selfatt = SelfAttention(num_hid, dropout=0.3)
    
    #先写死了吧 后面再调
    q_att = Attention2(36, dropout=0.3)
    v_att = Attention2(24, dropout=0.3)
    
    re_att = ReAttention(num_hid,36,0.3)


    classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates, .5)
    

    return ReattentionModel(w_emb, q_emb, v_att, q_att, v_selfatt, q_selfatt, q_prj, v_prj, re_att, classifier,model_net)







class Self_Supervised_Model(nn.Module):
    def __init__(self, w_emb, q_emb, gv_att_1,gv_att_2, q_net, v_net,gamma,num_hid, classifier):
        super(Self_Supervised_Model, self).__init__()

        self.w_emb = w_emb
        self.q_emb = q_emb

        self.q_net = q_net
        self.gv_net = v_net

        self.gv_att_1 = gv_att_1
        self.gv_att_2 = gv_att_2
        self.classifier = classifier
        
        self.normal = nn.BatchNorm1d(num_hid,affine=False)

    def forward(self, q, gv_pos, self_sup=True):

        """Forward
        q: [batch_size, seq_length]
        gv_pos: [batch, K, v_dim]
        self_sup: use negative images or not
        return: logits, not probs
        """

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]
        q_repr = self.q_net(q_emb)
        
        batch_size = q.size(0)

        logits_pos, att_gv_pos = self.compute_predict(q_repr, q_emb, gv_pos)

        if self_sup:
            # construct an irrelevant Q-I pair for each instance
            index = random.sample(range(0, batch_size), batch_size)
            gv_neg = gv_pos[index]
            logits_neg, att_gv_neg = \
                self.compute_predict(q_repr, q_emb, gv_neg)
            return logits_pos, logits_neg, att_gv_pos, att_gv_neg
        else:
            return logits_pos, att_gv_pos

    def compute_predict(self, q_repr, q_emb, v):

        att_1 = self.gv_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.gv_att_2(v, q_emb)  # [batch, 1, v_dim]
        att_gv = att_1 + att_2

        gv_embs = (att_gv * v)  # [batch, v_dim]
        gv_emb = gv_embs.sum(1)
        gv_repr = self.gv_net(gv_emb)

        joint_repr = q_repr * gv_repr

        joint_repr_normal = self.normal(joint_repr)
        logits = self.classifier(joint_repr_normal)

        return logits, att_gv

    
def build_Self_Supervised(dataset, num_hid, gamma=1):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)

    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    
    
    gv_att_1 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid,dropout=0.2,norm = 'weight', act='ReLU')
    gv_att_2 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid,dropout=0.2,norm = 'weight', act='ReLU')


    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return Self_Supervised_Model(w_emb, q_emb, gv_att_1,gv_att_2, q_net, v_net,gamma,num_hid, classifier)







class Graph(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, q_att, v_relation,
                 joint_embedding, classifier, glimpse, fusion, relation_type):
        super(Graph, self).__init__()
        self.name = "ReGAT_%s_%s" % (relation_type, fusion)
        self.relation_type = relation_type
        self.fusion = fusion
        self.dataset = dataset
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att
        self.v_relation = v_relation
        self.joint_embedding = joint_embedding
        self.classifier = classifier

    def forward(self, v, b, q, implicit_pos_emb, sem_adj_matrix,
                spa_adj_matrix, labels):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]torch.Size([1, 14]) torch.int32
        labels:torch.Size([1, 3129]) torch.int64
        pos: [batch_size, num_objs, nongt_dim, emb_dim]
        sem_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels] 【1024，36，15】
        spa_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels] 【1024，36，11】

        return: logits, not probs
        """
#        q = q.type(torch.LongTensor).cuda()
        w_emb = self.w_emb(q) #torch.Size([40, 14, 600]) torch.float32 cuda 40=320/8
        q_emb_seq = self.q_emb.forward_all(w_emb)  #torch.Size([1, 14, 1024]) torch.float32 cuda [batch, q_len, q_dim]
        q_emb_self_att = self.q_att(q_emb_seq) #torch.Size([1, 1024]) torch.float32 cuda

        # [batch_size, num_rois, out_dim]
        if self.relation_type == "semantic": #如果是语义，传递语义关系。
            v_emb = self.v_relation.forward(v, sem_adj_matrix, q_emb_self_att)
        elif self.relation_type == "spatial": #如果是空间，传递空间关系
            v_emb = self.v_relation.forward(v, spa_adj_matrix, q_emb_self_att)
        else:  # implicit
            v_emb = self.v_relation.forward(v, implicit_pos_emb,
                                            q_emb_self_att) #torch.Size([1, 36, 1024]) torch.float32 cuda

        if self.fusion == "ban":
            joint_emb, att = self.joint_embedding(v_emb, q_emb_seq, b)
        elif self.fusion == "butd":
            q_emb = self.q_emb(w_emb)  # [batch, q_dim]
            joint_emb, att = self.joint_embedding(v_emb, q_emb)
        else:  # mutan
            #joint_emb:torch.Size([1, 3129]) torch.float32 cuda
            #att:torch.Size([1, 2048]) torch.float32 cuda
            joint_emb, att = self.joint_embedding(v_emb, q_emb_self_att) #
        if self.classifier:
            logits = self.classifier(joint_emb)
        else:
            logits = joint_emb
        return logits, att


    
def build_Graph(dataset, num_hid, gamma=1):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0) #调用WordEmbding词嵌入方法
    q_emb = QuestionEmbedding(300,args.num_hid, 1, False, .0)

    
    q_att = QuestionSelfAttention(num_hid, .2)

    if args.relation_type == "semantic": #如果关系类型是语义的
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.sem_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    elif args.relation_type == "spatial": #如果关系类型是空间的
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.spa_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    else: #否则是隐式关系
        v_relation = ImplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
                        num_heads=args.num_heads, num_steps=args.num_steps,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    #分类器
    classifier = SimpleClassifier(num_hid, num_hid * 2,
                                  dataset.num_ans_candidates, 0.5)
    gamma = 0
    #采用融合方法
    if args.fusion == "ban":
        joint_embedding = BAN(args.relation_dim, args.num_hid, args.ban_gamma)
        gamma = args.ban_gamma
    elif args.fusion == "butd":
        joint_embedding = BUTD(args.relation_dim, args.num_hid, args.num_hid)
    else:
        joint_embedding = MuTAN(args.relation_dim, args.num_hid,
                                dataset.num_ans_candidates, args.mutan_gamma)
        gamma = args.mutan_gamma
        classifier = None
    return Graph(dataset, w_emb, q_emb, q_att, v_relation, joint_embedding,
                 classifier, gamma, args.fusion, args.relation_type)














def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv1d(in_planes, out_planes, kernel_size=3,stride=stride, padding=1, bias=bias))

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv1d(in_planes, out_planes, kernel_size=1,stride=stride, padding=0, bias=bias))

#这些parallel就是为了对于多个不同源的输入能够做到保持同结构？ 我猜的
#感觉像是这样 像是高级写法的siamase？


class exchange(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net,q_net_2,v_net_2, gamma, classifier, block, layers, num_classes=16, bn_threshold=2e-2):
        super(exchange, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier

        self.q_net = q_net
        self.v_net = v_net
        self.q_net_2 = q_net_2
        self.v_net_2 = v_net_2
        
        self.inplanes = 1
        self.num_parallel = 2
        self.dropout = ModuleParallel(nn.Dropout(p=0.5))
        self.conv1 = ModuleParallel(nn.Conv1d(1, 1, kernel_size=7, stride=1, padding=3,  bias=False))
        self.bn1 = BatchNorm1dParallel(1, self.num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.maxpool = ModuleParallel(nn.MaxPool1d(kernel_size=3, stride=1, padding=1))
        
        #self.layer1 = self._make_layer(block, 2, layers[0], bn_threshold)
        #self.layer2 = self._make_layer(block, 4, layers[1], bn_threshold, stride=1)

        #4 * 4 上面layers[x,x]定义了四个重复的这个东西，然后每个输出的层数是4
        self.clf_conv = self.conv3x3(1, 8, bias=True)
        
        #大概是权重？？ 有几个源就初始化一个多长的一维向量。
        self.alpha = nn.Parameter(torch.ones(self.num_parallel, requires_grad=True))
        # self.alpha = nn.Parameter(torch.ones([1, num_parallel, 157, 157], requires_grad=True))
        self.register_parameter('alpha', self.alpha)
        
        
    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)      
        #alpha_soft = F.softmax(self.alpha)
        q_repr2 = self.q_net_2(q_repr)
        v_repr2 = self.v_net_2(v_repr)        
        #joint_repr = self.my_forward([q_repr.unsqueeze(1),v_repr.unsqueeze(1)])
        #joint_repr = alpha_soft[0] * q_repr + alpha_soft[1] * v_repr
        joint_repr = (q_repr2 + q_repr) * (v_repr2 + v_repr)
        logits = self.classifier(joint_repr)
        return logits

    def conv3x3(self, in_planes, out_planes, stride=1, bias=False):

        return ModuleParallel(nn.Conv1d(in_planes, out_planes, kernel_size=3,stride=stride, padding=1, bias=bias))

    def conv1x1(self, in_planes, out_planes, stride=1, bias=False):

        return ModuleParallel(nn.Conv1d(in_planes, out_planes, kernel_size=1,stride=stride, padding=0, bias=bias))
    
    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
        #个人感觉就是把bottlenet那边定义的模型解析出来了，
        
        downsample = None
        
        #这是如果在bottleneck里几次卷积之后维度变小了的话，那么下采样的模型就按照这一部分定义的东西来走。
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm1dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold))

        return nn.Sequential(*layers)

    def my_forward(self, x):
        
        x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.maxpool(x)

        #l1 = self.layer1(x)
        #l2 = self.layer2(l1)
        #out = self.clf_conv(x)
        ens = 0     
        #softmax这个权重，然后最终的结果是把这几个源的结果进行加权得到的。
        alpha_soft = F.softmax(self.alpha)
        for l in range(self.num_parallel):
            ens += alpha_soft[l] * x[l].detach()
        
        return torch.mean(ens,1)

#原版
def build_exchange(dataset, num_hid,gamma=8):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    
    q_prj = []
    v_prj = []
    objects = 36  # minimum number of boxes

    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])

    q_net_2 = FCNet([num_hid, num_hid, num_hid,num_hid])
    v_net_2 = FCNet([num_hid, num_hid, num_hid,num_hid])    
    #siamese = SiameseNetwork()
    #criterion = ContrastiveLoss() #定义损失函数
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    
    block = Bottleneck
    
    layers = [2,2]
    return exchange(w_emb, q_emb, v_att, q_net,v_net,q_net_2,v_net_2,gamma, classifier, block, layers, 16, 2e-2)
