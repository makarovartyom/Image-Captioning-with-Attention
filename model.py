import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        features = self.resnet(images)
        # first, we need to resize the tensor to be 
        # (batch, size*size, feature_maps)
        batch, feature_maps, size_1, size_2 = features.size()       
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch, size_1*size_2, feature_maps)
       
        #features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    # sample_prob - probability of sampling from classifier's output
    def __init__(self, num_features, embedding_dim, hidden_dim, vocab_size, p =0.5):
        """
        Arguments:
         - embedding_dim - specified size of embeddings;
         - hidden_dim - the size of RNN layer (number of hidden states)
         - vocab_size - size of vocabulary
        """
        super(DecoderRNN, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # scale the inputs to softmax
        self.sample_temp = 0.5
        # embedding layer that turns words into a vector of a specified size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTM will have a single layer of size 512 (512 hidden units)
        # LSTM will input concatinated context vector (produced by attention) 
        # and corresponding hidden state of Decoder
        self.lstm = nn.LSTMCell(embedding_dim + num_features, hidden_dim)
        # produce the final output
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # add attention layer 
        self.attention = BahdanauAttention(num_features, hidden_dim)
        # Dropout layer
        self.drop = nn.Dropout(p=p)
        # add initialization fully-connected layers
        # initialize hidden state and cell memory using average feature vector 
        # Source: https://arxiv.org/pdf/1502.03044.pdf
        self.init_h = nn.Linear(num_features, hidden_dim)
        self.init_c = nn.Linear(num_features, hidden_dim)
        
    def forward(self, captions, features, sample_prob = 0.0):
        # create embeddings for captions
        embed = self.embeddings(captions)
        # initialize the hidden state
        h, c = self.init_hidden(features)
        # estimate the length of sequence and input sizes
        
        seq_len = captions.size(1)
        feature_size = features.size(1)
        batch_size = features.size(0)
        #vocab_size = self.vocab_size
        # this tensor will store the outputs from lstm cell
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
        # the same way we store the attention weights
        atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(device)
        for t in range(seq_len):
            # do not use scheduled sampling for the first timestep (<start> word)
            sample_prob = 0.0 if t == 0 else 0.5
            
            # check if sample prob is bigger than random
            use_sampling = np.random.random() < sample_prob
            
            if use_sampling == False:
                word_embed = embed[:,t,:]
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([word_embed, context], 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h)
            if use_sampling == True:
                scaled_output = output / self.sample_temp
                scoring = F.log_softmax(scaled_output, dim=1)
                top_idx = output.topk(1)[1]
                word_embed = self.embeddings(top_idx).squeeze(1)                                          
            outputs[:, t, :] = output
            atten_weights[:, t, :] = atten_weight    
        
        return outputs, atten_weights
    
    # initialize hidden state and cell memory using average feature vector 
    # Source: https://arxiv.org/pdf/1502.03044.pdf
    
    # "The initial memory state and hidden state of the LSTM
    # are predicted by an average of the annotation vectors fed
    # through two separate MLPs (init_c and init_h)"
    def init_hidden(self, features):
        mean_annotations = torch.mean(features, dim = 1)
        h0 = self.init_h(mean_annotations)
        c0 = self.init_c(mean_annotations)
        return h0, c0
    
    def sample(self, features, max_sentence = 20):
        sentence = []
        #input_word = data_loader.dataset.vocab('<start>')
        input_word = torch.tensor(0).unsqueeze(0).to(device)
        h, c = self.init_hidden(features)
        while True:
            embedded_word = self.embeddings(input_word)
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h) 
            scoring = F.log_softmax(output, dim=1)
            top_idx = output[0].topk(1)[1]
            sentence.append(top_idx.item())
            input_word = top_idx
            if (len(sentence) >= max_sentence or top_idx == 1):
                break
        return sentence      
                    
    
"""
- Additive Bahdanau Attention
- paper: https://arxiv.org/pdf/1409.0473.pdf
"""    
class BahdanauAttention(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim = 1):
        super(BahdanauAttention, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # fully-connected layer to learn first weight matrix Wa
        self.W_a = nn.Linear(self.num_features, self.hidden_dim)
        # fully-connected layer to learn the second weight matrix Ua
        self.U_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        # fully-connected layer to produce score (output), learning weight matrix va
        self.v_a = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, features, decoder_hidden):
        # add additional dimension to a hidden (need for summation later)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        atten_1 = self.W_a(features)
        atten_2 = self.U_a(decoder_hidden)
        # apply tangent to combined result from 2 fc layers
        atten_tan = torch.tanh(atten_1+atten_2)
        # one score corresponds to one Encoder's output
        # these values are how much each Encoder's output will be expressed in attention vector
        # Decoder will look at this vector before producing an output
        atten_score = self.v_a(atten_tan)
        atten_weight = F.softmax(atten_score, dim = 1)
        # first, we will multiply each vector by its softmax score
        # next, we will sum up this vectors, producing the attention context vector
        # the size of this vector will be equal to a size of the Encoder's single output
        context = torch.sum(atten_weight * features, 
                           dim = 1)
        atten_weight = atten_weight.squeeze(dim=2)
        return context, atten_weight
    