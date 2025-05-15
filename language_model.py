import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)
        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch, x):
        # just to get the type of tensor
        device = x.device
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(torch.zeros(*hid_shape).float().to(device)),Variable(torch.zeros(*hid_shape).float().to(device)))
        else:
            return Variable(torch.zeros(*hid_shape).float().to(device)) 

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch,x)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        if self.ndirections == 1:
            return output[:, -1]
        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)
    #看来这个out如果是整个的，那么输出的就是14个，就是14*1024的，如果输出最后一层，就是1*1024，就是拿最后一层来表征整个句子。
    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch, x)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output

    
    
class WordEmbedding2(nn.Module):
    """A class for extracting feature vectors from tokens in the question.

    This class essentially behaves as a look-up table storing fixed-length
    300-dimensional word embeddings are extracted. Embeddings are first
    initialized using GloVe word embeddings,
    (https://nlp.stanford.edu/pubs/glove.pdf).
    The embeddings will be finetuned during training to so that they are more
    specific of the task at hand (https://arxiv.org/pdf/1505.07931.pdf)

    Attributes:
        embedding_lookup: Embedding table that returns vector embedding given
            an index. The table also doubles as a trainable layer.
    """

    def __init__(self,vocabulary_size: int,embedding_dimension: int = 300,dropout: float = 0.0):
        """Initializes WordEmbedding.

        Args:
            vocabulary_size: Size of the lookup table.
            pretrained_vectors_file: Path to numpy file containing pretrained
                vector embeddings of all the words in the model's vocabulary.
                See tools/create_embedding.py to generate this file.
            embedding_dimension: Dimension of the extracted vector.
        """
        super().__init__()

        # padding_idx: dataset.py pads the list of token for shorter questions
        # with a value equal to the size of the vocabulary. Embeddings at an
        # index equal to this value (=vocabulary_size) are not updated during
        # training.
        self.embedding_lookup = nn.Embedding(
            num_embeddings=vocabulary_size + 1,
            embedding_dim=embedding_dimension,
            padding_idx=vocabulary_size,
        )

        self.dropout = nn.Dropout(dropout)
        self.vocabulary_size = vocabulary_size
        # Ensures that the word vectors are fine tuned to the VQA task during
        # training.
        self.embedding_lookup.weight.requires_grad = True
        
    def init_embedding(self, np_file):
        pretrained_weights = torch.from_numpy(np.load(np_file))
        self.embedding_lookup.weight.data[:self.vocabulary_size] = pretrained_weights

    def forward(self, inp):
        """Defines the computation performed at every call."""
        return self.dropout(self.embedding_lookup(inp))


class QuestionEmbedding2(nn.Module):
    """For extracting features from word indices of a tokenized question.

    The input list of word indices are w.r.t to the WordEmbedding lookup table.
    """

    def __init__(
        self,
        input_dimension: int,
        number_hidden_units: int,
        number_of_layers: int,
    ):
        """Initializes QuestionEmbedding.

        Args:
            input_dimension: The number of expected features in the input inp.
            number_hidden_units: The number of features in the hidden state h.
            number_of_layers: Number of recurrent layers.
        """
        super().__init__()

        self.number_hidden_units = number_hidden_units
        self.number_of_layers = number_of_layers
        self.lstm = nn.LSTM(
            input_size=input_dimension,
            hidden_size=number_hidden_units,
            num_layers=number_of_layers,
            bidirectional=False,
            batch_first=True,
        )

    def init_hidden(self, batch_size):
        """Grabs parameters of the model to instantiate a tensor on same device.

        Based on
        https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L56

        Args:
            bcz: batch size.
        """
        weight = next(self.parameters())
        return (
            weight.new_zeros(
                self.number_of_layers, batch_size, self.number_hidden_units
            ),
            weight.new_zeros(
                self.number_of_layers, batch_size, self.number_hidden_units
            ),
        )

    def forward(self, inp):
        """Defines the computation performed at every call.

        Args:
            inp:
                tensor containing the features of the input sequence:
                (batch, sequence, input_dimension)
                Tensor has shape:
                (batch_size, question_sequence_length, input_size).
        """
        batch_size = inp.size(0)
        hidden = self.init_hidden(batch_size)

        # Compact weights into single contiguous chunk of memory.
        self.lstm.flatten_parameters()

        output, hidden = self.lstm(inp, hidden)

        return output


class QuestionSelfAttention(nn.Module):
    def __init__(self, num_hid, dropout):
        super(QuestionSelfAttention, self).__init__()
        self.num_hid = num_hid
        self.drop = nn.Dropout(dropout)
        self.W1_self_att_q = MultiLayerNet(
            [num_hid, num_hid], dropout=dropout, activation_fn_name=None
        )
        self.W2_self_att_q = MultiLayerNet(
            [num_hid, 1], activation_fn_name=None
        )

    def forward(self, ques_feat):
        """
        ques_feat: [batch, 14, num_hid]
        """
        batch_size = ques_feat.shape[0]
        q_len = ques_feat.shape[1]

        # (batch*14,num_hid)
        ques_feat_reshape = ques_feat.contiguous().view(-1, self.num_hid)
        # (batch, 14)
        atten_1 = self.W1_self_att_q(ques_feat_reshape)
        atten_1 = torch.tanh(atten_1)
        atten = self.W2_self_att_q(atten_1).view(batch_size, q_len)
        # (batch, 1, 14)
        weight = F.softmax(atten.t(), dim=1).view(-1, 1, q_len)
        ques_feat_self_att = torch.bmm(weight, ques_feat)
        ques_feat_self_att = ques_feat_self_att.view(-1, self.num_hid)
        # (batch, num_hid)
        ques_feat_self_att = self.drop(ques_feat_self_att)
        return ques_feat_self_att