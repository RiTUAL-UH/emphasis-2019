import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import config

class SeqModel(nn.Module):
    def __init__(self, embeddings, num_labels, extractor_type,  hidden_dim):
        super(SeqModel, self).__init__()
        print("hidden dim: ", hidden_dim)
        self.wordEmbedding = EmbeddingLayer(embeddings)
        self.featureEncoder = FeatureEncoder(input_dim=embeddings.shape[1], extractor_type= extractor_type, hidden_dim =hidden_dim)
        if config.if_att:
            self.attention = Attention(hidden_dim)

        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, 12),
            nn.LayerNorm(12),
            nn.Linear(12, num_labels),
        )



        if torch.cuda.is_available():
            self.wordEmbedding = self.wordEmbedding.cuda()
            self.featureEncoder = self.featureEncoder.cuda()
            if config.if_att:
                self.attention = self.attention.cuda()
            self.score_layer = self.score_layer.cuda()



    def forward(self, w_tensor, mask):
        emb_sequence = self.wordEmbedding(w_tensor)  # w_tensor shape: [batch_size, max_seq_len] 
        features = self.featureEncoder(emb_sequence, mask)  # emb_sequence shape: [batch_size, max_seq_len, emb_dim] 

        if config.if_att:
            att_output, att_weights = self.attention(features, mask.float())
            scores = self.score_layer(att_output) # features shape: [batch_size, max_seq_len, hidden_dim] 
        else:
            scores = self.score_layer(features)  # features shape: [batch_size, max_seq_len, hidden_dim] 
            att_weights = None
        return scores, att_weights # score shape: [batch_size, max_seq_len, num_labels] 



class EmbeddingLayer(nn.Module):
    def __init__(self, embeddings):
        super(EmbeddingLayer, self).__init__()

        self.word_encoder = nn.Sequential(
            nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False),
            nn.Dropout(0.3)
        )

        if torch.cuda.is_available():
            self.word_encoder = self.word_encoder.cuda()

    def forward(self, w_tensor):
        return self.word_encoder(w_tensor)



class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, extractor_type, hidden_dim):
        super(FeatureEncoder, self).__init__()


        self.extractor_type = extractor_type
        self.hidden_dim = hidden_dim

        if self.extractor_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, self.hidden_dim//2, num_layers=2, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.4)
           

            if torch.cuda.is_available():
                self.lstm = self.lstm.cuda()
                self.dropout = self.dropout.cuda()


    def forward(self, sequences, mask):
        """
               :param sequences: sequence shape: [batch_size, seq_len, emb_dim] => [128, 44, 100]
               :param mask:
               :return:
        """
        if self.extractor_type == 'lstm':
            lengths = torch.sum(mask, 1) # sum up all 1 values which is equal to the lenghts of sequences
            lengths, order = lengths.sort(0, descending=True)
            recover = order.sort(0, descending=False)[1]

            sequences = sequences[order]
            packed_words = pack_padded_sequence(sequences, lengths.cpu().numpy(), batch_first=True)
            lstm_out, hidden = self.lstm(packed_words, None)

            feats, _ = pad_packed_sequence(lstm_out)
            feats = feats.permute(1, 0, 2)
            feats = feats[recover] # feat shape: [batch_size, seq_len, hidden_dim] 
        return feats




class Attention(nn.Module):
    """Attention mechanism written by Gustavo Aguilar https://github.com/gaguilar"""
    def __init__(self,  hidden_size):
        super(Attention, self).__init__()
        self.da = hidden_size
        self.dh = hidden_size

        self.W = nn.Linear(self.dh, self.da)        # (feat_dim, attn_dim)
        self.v = nn.Linear(self.da, 1)              # (attn_dim, 1)

    def forward(self, inputs, mask):
        # Raw scores
        u = self.v(torch.tanh(self.W(inputs)))      # (batch, seq, hidden) -> (batch, seq, attn) -> (batch, seq, 1)

        # Masked softmax
        u = u.exp()                                 # exp to calculate softmax
        u = mask.unsqueeze(2).float() * u           # (batch, seq, 1) * (batch, seq, 1) to zerout out-of-mask numbers
        sums = torch.sum(u, dim=1, keepdim=True)    # now we are sure only in-mask values are in sum
        a = u / sums                                # the probability distribution only goes to in-mask values now

        # Weighted vectors
        z = inputs * a

        return  z,  a.view(inputs.size(0), inputs.size(1))
