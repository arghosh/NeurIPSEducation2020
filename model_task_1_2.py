import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, n_question,  n_user, n_subject, n_quiz, n_group, hidden_dim, q_dim,  task,  dropout=0.25,  s_dim=256, default_dim=16, num_gru_layers=1, is_dash=False, bidirectional=True):
        super().__init__()
        self.is_dash = is_dash
        self.q_embeddings = nn.Embedding(n_question, q_dim)
        self.s_embeddings = nn.Embedding(n_subject, s_dim)
        self.quiz_embeddings = nn.Embedding(n_quiz, default_dim)
        self.group_embeddings = nn.Embedding(n_group, default_dim)
        self.feature_layer = nn.Linear(2, default_dim)
        self.answer_embeddings = nn.Embedding(4, s_dim)
        self.label_embeddings = nn.Embedding(2, s_dim)
        self.user_feature_layer = nn.Linear(8, default_dim)
        self.dropout = nn.Dropout(dropout)
        # [questions, subjects,  ans_embed, correct_ans_embed, label_embed], dim=-1))
        self.in_feature = s_dim * 4 + q_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=self.in_feature, hidden_size=hidden_dim,
                          num_layers=num_gru_layers, batch_first=False, bidirectional=True)
        self.task = task
        #pred_input = [subjects, questions, quizs, groups, forward_ht, user_features, features]
        self.pred_in_feature = hidden_dim + s_dim + q_dim + 4*default_dim
        self.pred_in_feature += hidden_dim
        self.pred_in_feature += s_dim
        if self.is_dash:
            self.dash_layer = nn.Linear(16, q_dim)
            self.pred_in_feature += q_dim

        self.layers = nn.Sequential(
            nn.Linear(self.pred_in_feature,
                      self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout*2),
            nn.Linear(self.pred_in_feature, self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout*2))

        if self.task == '1':
            self.output_layer = nn.Linear(self.pred_in_feature, 1)
        elif self.task == '2':
            self.output_layer = nn.Linear(self.pred_in_feature, 4)

    def forward(self, batch):
        ans, correct_ans, labels = batch['ans'].to(
            device)-1, batch['correct_ans'].to(device) - 1, batch['labels'].to(device)
        seq_len, batch_size, data_length = ans.shape[0], ans.shape[1], batch['L']
        test_mask = batch['test_mask'].to(device).unsqueeze(2)
        valid_mask = batch['valid_mask'].to(device).unsqueeze(2)
        user_features = (self.user_feature_layer(
            batch['user_features'].to(device))).unsqueeze(0).expand(seq_len, -1, -1)  # T, B,uf_dim
        subjects = torch.sum((self.s_embeddings(
            batch['subject_ids'].to(device))) * batch['subject_mask'].to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        questions = (self.q_embeddings(
            batch['q_ids'].to(device)))  # T, B,q_dim
        quizs = (self.quiz_embeddings(
            batch['quiz_ids'].to(device)))  # T, B,q_dim
        groups = (self.group_embeddings(
            batch['group_ids'].to(device)))  # T, B,q_dim

        # apply test_mask
        #qs = batch['q_ids'].to(device)
        ans_ = ans  # + qs*4
        correct_ans_ = correct_ans  # + qs*4
        labels_ = labels.long()  # + qs*2
        ans_embed = (self.answer_embeddings(ans_))*test_mask*valid_mask
        correct_ans_embed = (self.answer_embeddings(correct_ans_))
        label_embed = self.label_embeddings(labels_)*test_mask*valid_mask

        input_feature = torch.cat([batch['times'].to(device).unsqueeze(
            2), batch['confidences'].to(device).unsqueeze(2)], dim=-1)
        features = (self.feature_layer(input_feature))

        lstm_input = [ans_embed, correct_ans_embed, label_embed]

        lstm_input.append(questions)
        lstm_input.append(subjects)

        lstm_input = self.dropout(torch.cat(lstm_input, dim=-1))

        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths=(data_length), batch_first=False, enforce_sorted=False)
        packed_output, ht = self.rnn(packed_data)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=False)
        output = output.view(seq_len, batch_size, 2, self.hidden_dim)
        init_state = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        forward_ht = torch.cat([init_state, output[:-1, :, 0, :]], dim=0)
        reverse_ht = torch.cat([output[1:, :, 1, :], init_state], dim=0)

        pred_input = [questions, quizs, groups,
                      forward_ht, user_features, features]

        pred_input.append(subjects)
        pred_input.append(reverse_ht)
        if self.is_dash:
            dash_features = self.dropout(
                self.dash_layer(batch['dash_features'].to(device)))
            pred_input.append(dash_features)
        # if self.task == '2':
        pred_input.append(correct_ans_embed)

        pred_input = self.dropout(torch.cat(pred_input, dim=-1))

        output = self.output_layer(self.layers(pred_input)+pred_input)

        #
        if self.task == '1':
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            output = output.squeeze()
            loss = loss_fn(output, labels)
            loss = loss * test_mask.squeeze(2) * valid_mask.squeeze(2)
            loss = loss.mean() / torch.mean((test_mask*valid_mask).float())
            m = nn.Sigmoid()
            return loss, m(output).detach().cpu().numpy()
        elif self.task == '2':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(output.view(-1, 4), ans.view(-1))
            loss = loss.view(-1, batch_size) * \
                test_mask.squeeze(2) * valid_mask.squeeze(2)
            loss = loss.mean() / torch.mean((test_mask*valid_mask).float())
            pred = torch.max(output, dim=-1)[1]+1
            return loss, pred.detach().cpu().numpy()


class NCFModel(nn.Module):
    def __init__(self, n_question, n_subject, n_user, n_quiz, n_group, task, dash=True, dropout=0.2, q_dim=256, s_dim=256, u_dim=256, uf_dim=32):
        super().__init__()
        self.q_embeddings = nn.Embedding(n_question, q_dim)
        self.u_embeddings = nn.Embedding(n_user, u_dim)
        self.s_embeddings = nn.Embedding(n_subject, s_dim)
        self.quiz_embeddings = nn.Embedding(n_quiz, uf_dim)
        self.group_embeddings = nn.Embedding(n_group, uf_dim)
        # if is_extra:
        self.dash = dash
        #self.dash_layer = nn.Linear(9, s_dim)
        self.user_feature_layer = nn.Linear(8, uf_dim)
        self.dropout = nn.Dropout(dropout)
        in_feature = q_dim+s_dim+u_dim+3*uf_dim
        if self.dash:
            self.dash_layer = nn.Linear(17, s_dim)
            in_feature += s_dim

        self.layers = nn.Sequential(
            nn.Linear(in_feature, in_feature), nn.ReLU(
            ), nn.Dropout(dropout*2),
            nn.Linear(in_feature, in_feature), nn.ReLU(), nn.Dropout(dropout*2)
        )
        self.task = task
        if self.task == '1':
            self.output_layer = nn.Linear(in_feature, 1)
        else:
            self.output_layer = nn.Linear(in_feature, 4)

    def forward(self, batch):
        # {'q_ids': q_ids, 'u_ids': u_ids, 'ans': answers, 'u_features': u_features,
        #    'u_confs': u_confs, 'q_confs': q_confs, 'subjects': subjects, 'sub_mask': sub_mask}
        u_ids = self.dropout(self.u_embeddings(
            batch['u_ids'].to(device)))  # B, u_dim
        q_ids = self.dropout(self.q_embeddings(
            batch['q_ids'].to(device)))  # B,q_dim
        if self.task == '1':
            a_ids = batch['ans'].float().to(device)  # B,1
        else:
            a_ids = batch['ans'].to(device) - 1  # B,
        u_features = self.dropout(self.user_feature_layer(
            batch['u_features'].to(device)))  # B,uf_dim
        q_sub = self.dropout(self.s_embeddings(
            batch['subjects'].to(device)))  # B, L, s_dim
        q_sub_mask = batch['sub_mask'].to(device)  # B,L
        subject_embeddings = torch.sum(
            q_sub, dim=1) / torch.sum(q_sub_mask, dim=1, keepdim=True)  # B, s_dim
        quiz_ids = self.dropout(
            self.quiz_embeddings(batch['quiz_ids'].to(device)))
        group_ids = self.dropout(
            self.group_embeddings(batch['group_ids'].to(device)))
        input_embeddings = [u_ids, q_ids, u_features,
                            subject_embeddings, quiz_ids, group_ids]
        if self.dash:
            dash_features = self.dropout(
                self.dash_layer(batch['dash_features'].to(device)))
            input_embeddings.append(dash_features)
        input_embeddings = torch.cat(input_embeddings, dim=-1)

        output = self.output_layer(self.layers(
            input_embeddings)+input_embeddings)
        if self.task == '1':
            loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
            loss = loss_fn(output.squeeze(), a_ids)
            m = nn.Sigmoid()
            return loss, m(output).detach().cpu().numpy()
        else:
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fn(output, a_ids)
            pred = torch.max(output, dim=1)[1]+1

            return loss, pred.detach().cpu().numpy()


class AttentionModel(nn.Module):
    def __init__(self, n_question,  n_user, n_subject, n_quiz, n_group, hidden_dim, q_dim,  task,  dropout=0.25, n_heads=1,  s_dim=256, default_dim=16, num_gru_layers=1, is_dash=False):
        super().__init__()
        self.is_dash = is_dash
        self.q_embeddings = nn.Embedding(n_question, q_dim)
        self.s_embeddings = nn.Embedding(n_subject, s_dim)
        self.quiz_embeddings = nn.Embedding(n_quiz, default_dim)
        self.group_embeddings = nn.Embedding(n_group, default_dim)
        self.feature_layer = nn.Linear(2, default_dim)
        self.answer_embeddings = nn.Embedding(4, s_dim)
        self.label_embeddings = nn.Embedding(2, s_dim)
        self.user_feature_layer = nn.Linear(8, default_dim)
        self.dropout = nn.Dropout(dropout)
        # [questions, subjects,  ans_embed, correct_ans_embed, label_embed], dim=-1))
        self.in_feature = s_dim * 4 + q_dim + default_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=self.in_feature, hidden_size=hidden_dim,
                          num_layers=num_gru_layers, batch_first=False, bidirectional=True)
        self.task = task
        #pred_input = [subjects, questions, quizs, groups, forward_ht, user_features, features]
        self.pred_in_feature = hidden_dim + s_dim + q_dim + 4*default_dim
        self.pred_in_feature += hidden_dim
        # if self.task == '2':
        self.pred_in_feature += s_dim
        if self.is_dash:
            self.dash_layer = nn.Linear(16, q_dim)
            self.pred_in_feature += q_dim
        # attention
        self.pred_in_feature += hidden_dim
        ####

        self.layers = nn.Sequential(
            nn.Linear(self.pred_in_feature,
                      self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout*2),
            nn.Linear(self.pred_in_feature, self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout*2))

        if self.task == '1':
            self.output_layer = nn.Linear(self.pred_in_feature, 1)
        elif self.task == '2':
            self.output_layer = nn.Linear(self.pred_in_feature, 4)
        ###
        d_key = default_dim
        d_val = s_dim
        self.attention_model = MultiHeadAttention(
            d_key=d_key, d_val=d_val, n_heads=n_heads, dropout=dropout, d_model=hidden_dim)

    def forward(self, batch):
        ans, correct_ans, labels = batch['ans'].to(
            device)-1, batch['correct_ans'].to(device) - 1, batch['labels'].to(device)
        seq_len, batch_size, data_length = ans.shape[0], ans.shape[1], batch['L']
        test_mask = batch['test_mask'].to(device).unsqueeze(2)
        valid_mask = batch['valid_mask'].to(device).unsqueeze(2)
        user_features = (self.user_feature_layer(
            batch['user_features'].to(device))).unsqueeze(0).expand(seq_len, -1, -1)  # T, B,uf_dim
        subjects = torch.sum((self.s_embeddings(
            batch['subject_ids'].to(device))) * batch['subject_mask'].to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        questions = (self.q_embeddings(
            batch['q_ids'].to(device)))  # T, B,q_dim
        quizs = (self.quiz_embeddings(
            batch['quiz_ids'].to(device)))  # T, B,q_dim
        groups = (self.group_embeddings(
            batch['group_ids'].to(device)))  # T, B,q_dim

        # apply test_mask
        #qs = batch['q_ids'].to(device)
        ans_ = ans  # + qs*4
        correct_ans_ = correct_ans  # + qs*4
        labels_ = labels.long()  # + qs*2
        ans_embed = (self.answer_embeddings(ans_))*test_mask*valid_mask
        correct_ans_embed = (self.answer_embeddings(correct_ans_))
        label_embed = self.label_embeddings(labels_)*test_mask*valid_mask

        input_feature = torch.cat([batch['times'].to(device).unsqueeze(
            2), batch['confidences'].to(device).unsqueeze(2)], dim=-1)
        features = (self.feature_layer(input_feature))

        lstm_input = [ans_embed, correct_ans_embed, label_embed, quizs]

        lstm_input.append(questions)
        lstm_input.append(subjects)

        lstm_input = self.dropout(torch.cat(lstm_input, dim=-1))

        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths=(data_length), batch_first=False, enforce_sorted=False)
        packed_output, ht = self.rnn(packed_data)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=False)
        output = output.view(seq_len, batch_size, 2, self.hidden_dim)
        init_state = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        forward_ht = torch.cat([init_state, output[:-1, :, 0, :]], dim=0)
        reverse_ht = torch.cat([output[1:, :, 1, :], init_state], dim=0)

        pred_input = [questions, quizs, groups,
                      forward_ht, user_features, features]

        pred_input.append(subjects)
        pred_input.append(reverse_ht)
        if self.is_dash:
            dash_features = self.dropout(
                self.dash_layer(batch['dash_features'].to(device)))
            pred_input.append(dash_features)
        # if self.task == '2':
        pred_input.append(correct_ans_embed)
        # Add Attention
        attn_mask = (test_mask * valid_mask).squeeze(2)  # T,B
        query = torch.cat([quizs], dim=-1)  # T, B, dim
        value = torch.cat([label_embed], dim=-1)  # T,B, dim
        attention_state = self.attention_model(
            query, query, value, attn_mask)  # T,B,dim
        #
        pred_input.append(attention_state)
        # End Attention
        pred_input = self.dropout(torch.cat(pred_input, dim=-1))

        output = self.output_layer(self.layers(pred_input)+pred_input)

        #
        if self.task == '1':
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            output = output.squeeze()
            loss = loss_fn(output, labels)
            loss = loss * test_mask.squeeze(2) * valid_mask.squeeze(2)
            loss = loss.mean() / torch.mean((test_mask*valid_mask).float())
            m = nn.Sigmoid()
            return loss, m(output).detach().cpu().numpy()
        elif self.task == '2':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(output.view(-1, 4), ans.view(-1))
            loss = loss.view(-1, batch_size) * \
                test_mask.squeeze(2) * valid_mask.squeeze(2)
            loss = loss.mean() / torch.mean((test_mask*valid_mask).float())
            pred = torch.max(output, dim=-1)[1]+1
            return loss, pred.detach().cpu().numpy()


class MultiHeadAttention(nn.Module):
    def __init__(self, d_key, d_val, d_model, n_heads, dropout, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.h = n_heads
        self.v_linear = nn.Linear(d_val, d_model, bias=bias)
        self.k_linear = nn.Linear(d_key, d_model, bias=bias)
        self.q_linear = nn.Linear(d_key, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        #
        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self._reset_parameters()
        self.position_embedding = CosinePositionalEmbedding(d_key)

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, qq, kk, vv, mask):
        T, B = qq.size(0), qq.size(1)
        position_embed = self.position_embedding(
            qq).expand(-1, B, -1)/math.sqrt(self.d_model)
        # perform linear operation and split into h heads
        #T,B, h,d_k
        k = self.k_linear(qq+position_embed).view(T, B, self.h, -1)
        q = self.q_linear(kk+position_embed).view(T, B, self.h, -1)
        v = self.v_linear(vv).view(T, B, self.h, -1)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.permute(1, 2, 0, 3)
        q = q.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        # calculate attention using function we will define next
        scores = attention(q, k, v, mask)  # BS,h, T, d_k

        # concatenate heads and put through final linear layer
        concat = scores.permute(2, 0, 1, 3).contiguous().view(T, B, -1)
        output = self.layer_norm1(self.dropout(
            self.out_proj(concat)))  # T,B,d_model
        #
        output_1 = self.linear2(self.dropout(
            self.activation(self.linear1(output))))
        output = output + self.layer_norm2(output_1)
        return output


def attention(q, k, v, mask):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(q.size(3))  # BS, h, seqlen, seqlen
    scores.masked_fill_(mask.transpose(1, 0)[:, None, None, :] == 0, -1e32)
    eye_mask = torch.eye(q.size(2))[None, None, :, :].to(device)
    scores.masked_fill_(eye_mask == 1, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    output = torch.matmul(scores, v)
    return output


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:x.size(0), :, :]  # ( seq, 1,  Feature)
