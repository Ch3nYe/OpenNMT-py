"""Define RNN-based encoders with GCN embedding."""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

from torch_geometric.data import Data



class GCNRNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.
    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, gcn_path=None, use_bridge=False, src_vocab=None):
        super(GCNRNNEncoder, self).__init__()
        assert gcn_path is not None
        assert src_vocab is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.gcn = torch.load(gcn_path)
        self.gcn.eval()

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=list(self.gcn.modules())[-1].out_features, # gcn output size
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Find vocab data for tree builting
        # 1.2.0 read src_vocab
        vocab = torch.load(src_vocab)
        self.DELIMITER = vocab["src"].fields[0][1].vocab.stoi["<EOT>"]
        self.BLANK = vocab["src"].fields[0][1].vocab.stoi["<blank>"]
        self.idx2num = []
        for i in vocab["src"].fields[0][1].vocab.itos:
            if i.isdigit():
                self.idx2num.append(int(i))
            else:
                self.idx2num.append(-1)
        # 2.2.0 read src_vocab
        # with open(src_vocab, "r") as f:
        #     idx = 0
        #     self.DELIMITER = -1
        #     self.idx2num = []
        #     for ln in f:
        #         ln = ln.strip().split('\t')[0]
        #         if idx == 0 and ln != "<unk>":
        #             self.idx2num.append(-1)
        #         if idx == 1 and ln != "<blank>":
        #             self.idx2num.append(-1)
        #         if ln == "<EOT>":
        #             self.DELIMITER = idx
        #         if ln.isdigit():
        #             self.idx2num.append(int(ln))
        #         else:
        #             self.idx2num.append(-1)
        #         idx += 1
        # assert self.DELIMITER >= 0, \
        #     "In GCNRNNEncoder src_vocab must include <EOT> token"


        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)



    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.gcn_path,
            opt.bridge,
            opt.src_vocab)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        # emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        lengths_list = []
        self.gcn.eval()
        embs = []
        for i in range(src.shape[1]):
            g, node_num = self.generate(src[:, i])
            g.to(src.device)
            lengths_list.append(node_num)
            emb, _ = self.gcn.project(g.x, g.edge_index, g.batch) # emb.shape == number of node * node embedding size
            embs.append(emb.detach())
        embs = pad_sequence(embs)

        packed_emb = embs
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            packed_emb = pack(embs, lengths_list, enforce_sorted=False)

        memory_bank, encoder_final = self.rnn(packed_emb.to(src.device))

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout

    def generate(self, src)->Data:
        # get feature, get edge
        nodes = []
        edges = []
        feature_end_flag = False
        edges_end_flag = False
        node_num = 0
        for token in src:
            if not feature_end_flag:
                if token == self.DELIMITER:
                    feature_end_flag = True
                    continue
                nodes.append(token)
                node_num += 1
            elif not edges_end_flag:
                if token == self.DELIMITER:
                    edges_end_flag = True
                    continue
                edges.append(self.idx2num[token])
            else: # deal with lib funcs
                pass

        x = torch.tensor(nodes).float().unsqueeze(1)
        edges = [edges[i:i+2] for i in range(0,len(edges),2)]
        edges = torch.tensor(edges).long().t()
        if edges.dim() !=2: # no edge graph
            edges = torch.tensor([[],[]]).long()
        # only one batch
        batch = torch.zeros([x.shape[0]]).long()
        # new Data()
        # x:'torch.FloatTensor'[node_num, 1], edge:'torch.LongTensor'[2, edge_num], batch:''torch.LongTensor'[node_num]
        graph_data = Data(x=x,edge_index=edges,batch=batch)
        return graph_data, node_num