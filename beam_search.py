import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BeamSearch():
    """Class performs the caption generation using Beam search.
    
    Attributes:
    ----------
    - decoder - trained Decoder of captioning model
    - features - feature map outputed from Encoder
    
    Returns:
    --------
    - sentence - generated caption
    - final_scores - cummulative scores for produced sequences
    """
    def __init__(self, decoder, features, k, max_sentence):
        
        self.k = k
        self.max_sentence = max_sentence
        self.decoder = decoder
        self.features = features
        
        self.h = decoder.init_hidden(features)[0]
        self.c = decoder.init_hidden(features)[1]
        
        self.start_idx = torch.zeros(1).long()
        self.start_score = torch.FloatTensor([0.0]).repeat(k)
        
        # hidden states on the first step for a single word
        self.hiddens = [[[self.h, self.c]]*k]
        self.start_input = [[self.start_idx], self.start_score]
        self.complete_seqs = [list(), list()]
        # track the step
        self.step = 0
        
        
    def beam_search_step(self):
        """Function performs a single step of beam search, returning start input"""
        top_idx_temp = []
        top_score_temp = []
        hiddens_temp = []
        
        for i, w in enumerate(self.start_input[0][-1]):
            
            hidden_states = self.hiddens[self.step][i]
            h = hidden_states[0]
            c = hidden_states[1]
            # scoring stays with the same dimensions
            embedded_word = self.decoder.embeddings(w.view(-1))
            context, atten_weight = self.decoder.attention(self.features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            
            h, c = self.decoder.lstm(input_concat, (h, c))
            output = self.decoder.fc(h)
            scoring = F.log_softmax(output, dim=1)
            top_scores, top_idx = scoring[0].topk(self.k)
        
            top_cum_score = top_scores + self.start_input[1][i]
            # append top indices and scores
            top_idx_temp.append(top_idx.view(-1, self.k))
            top_score_temp.append(top_cum_score.view(-1, self.k))
            # append hidden states
            hiddens_temp.append([h, c])
        self.hiddens.append(hiddens_temp)
            
        # concatinate temp lists
        top_idx_temp = torch.cat(top_idx_temp, dim =0)
        top_score_temp = torch.cat(top_score_temp, dim =0)
        cum_score = top_score_temp
        
        top_cum_scores = self.get_cummulative_score(cum_score)
        ready_idx, tensor_positions = self.get_ready_idx(top_cum_scores, 
                                                         top_idx_temp,
                                                         cum_score)
        row_pos = self.get_positions(tensor_positions)
        # update the attributes
        self.update_start_input(ready_idx, row_pos, top_cum_scores)
        self.update_hiddens(row_pos)
        self.update_step()
            
        # step == 1 means we have generated the hiddens from <start> word and outputed k first words
        # we use them to generate k second words
        if self.step == 1:
            self.hiddens[self.step] = self.hiddens[self.step] * self.k
            self.start_input[0][0] = self.start_input[0][0].view(self.k,-1)
        
        return  self.start_input
    
    def get_cummulative_score(self, cum_score):
        """Getting the top scores and indices from cum_score"""
        top_cum_scores, _ = cum_score.flatten().topk(self.k)
        return top_cum_scores
        
    
    def get_ready_idx(self, top_cum_scores, top_idx_temp, cum_score):
        """Obtain a list of ready indices and their positions"""
        # got the list of top positions 
        tensor_positions = [torch.where(cum_score == top_cum_scores[i]) for i in range(self.k)]
        # it's important to sort the tensor_positions by first entries (rows)
        # because rows represent the sequences: 0, 1 or 2 sequences
        tensor_positions = sorted(tensor_positions, key = lambda x: x[0])
        # get read top k indices that will be our input indices for next iteration
        ready_idx = torch.cat([top_idx_temp[tensor_positions[ix]] for ix in range(self.k)]).view(self.k, -1)
        return ready_idx, tensor_positions
        
        
    def get_positions(self, tensor_positions):
        """Retruns the row positions for tensors"""
        row_pos = [x[0] for x in tensor_positions]
        row_pos = torch.cat(row_pos, dim =0)
        return row_pos
    
    def get_nonend_tokens(self):
        """Get tokens that are not <end>"""
        non_end_token = self.start_input[0][-1] !=1
        return non_end_token.flatten()
        

    def update_start_input(self, ready_idx, row_pos, top_cum_scores):      
        """Returns new input sequences"""
        # construct new sequence with respect to the row positions
        start_input_new = [x[row_pos] for x in self.start_input[0]]
        self.start_input[0] = start_input_new 
        start_score_new = self.start_input[1][row_pos]
        self.start_input[1] = start_score_new
        
        # append new indices and update scoring
        self.start_input[0].append(ready_idx)
        self.start_input[1] = top_cum_scores.detach()
        
    def update_hiddens(self, row_pos):
        """Returns new hidden states"""
        self.hiddens = [[x[i] for i in row_pos.tolist()] for x in self.hiddens]
        
    def update_step(self):
        """Updates step"""
        self.step += 1
    
    def generate_caption(self):
        """Iterates over the sequences and generates final caption"""
        while True:
            # make a beam search step 
            self.start_input = self.beam_search_step()
        
            non_end_token = self.get_nonend_tokens()
            if (len(non_end_token) != sum(non_end_token).item()) and (sum(non_end_token).item() !=0):
                #prepare complete sequences and scores
                complete_seq = torch.cat(self.start_input[0], dim =1)[non_end_token !=1]
                complete_score = self.start_input[1][non_end_token !=1]
                self.complete_seqs[0].extend(complete_seq)
                self.complete_seqs[1].extend(complete_score)  
            
                start_input_new = torch.cat(self.start_input[0], dim =1)[non_end_token]
                start_input_new = [x.view(len(x), -1) for x in start_input_new.view(len(start_input_new[0]), -1)]
                start_score_new = self.start_input[1][non_end_token]
                
                self.start_input[0] = start_input_new
                self.start_input[1] = start_score_new
                
                non_end_pos = torch.nonzero(non_end_token).flatten()
                self.update_hiddens(non_end_pos)
            elif (sum(non_end_token).item() ==0):
                # prepare complete sequences and scores
                complete_seq = torch.cat(self.start_input[0], dim =1)[non_end_token !=1]
                complete_score = self.start_input[1][non_end_token !=1]
                
                self.complete_seqs[0].extend(complete_seq)
                self.complete_seqs[1].extend(complete_score) 
            else:
                pass
            if (len(self.complete_seqs[0])>=3 or self.step == self.max_sentence):
                break
        
        return self.get_top_sequence()

                
            
    def get_top_sequence(self):
        """Gets the sentence and final set of scores""""
        lengths = [len(i) for i in self.complete_seqs[0]]
        final_scores = [self.complete_seqs[1][i] / lengths[i] for i in range(len(lengths))]
        best_score = np.argmax([i.item() for i in final_scores])
        sentence = self.complete_seqs[0][best_score].tolist()
        return sentence, final_scores



