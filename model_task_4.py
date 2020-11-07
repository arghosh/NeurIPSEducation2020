import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np
import math
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data


class FFModel(nn.Module):
    def __init__(self, dropout=0.25, dim=8, hidden_dim=1024, alpha=0.1,concat_hidden_dim=128, concat_dim=512):
        super().__init__()
        self.alpha = alpha
        self.ans_embeddings = nn.Embedding(4, dim)
        self.label_embeddings = nn.Embedding(2, dim)
        self.dropout = nn.Dropout(dropout)
        self.in_feature = (dim + dim) * 948
        self.nonselected_layer = nn.Sequential(
            nn.Linear(948, concat_hidden_dim), nn.ReLU(
            ), nn.Dropout(dropout),
            nn.Linear(concat_hidden_dim, concat_dim), nn.ReLU(), nn.Dropout(dropout))
        self.in_feature += concat_dim

        self.layers = nn.Sequential(
            nn.Linear(self.in_feature, hidden_dim), nn.ReLU(
            ), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.output_layer = nn.Linear(hidden_dim, 948)


    def test(self, batch):  # 'brier' or 'eer'
        input_labels = batch['input_labels'].to(device).float()  # B,948
        modified_labels = batch['input_labels'].to(device)  # B,948
        input_ans = batch['input_ans'].to(device)-1  # B,948
        input_mask = batch['input_mask'].to(device)  # B,948
        label_embed = self.label_embeddings(modified_labels)
        ans_embed = self.ans_embeddings(input_ans)  # B,948, 4
        B = input_labels.shape[0]
        m = nn.Sigmoid()
        train_mask = torch.zeros(B, 948).long().to(device)
        other_embed = self.nonselected_layer(
            batch['input_mask'].to(device).float())
        for _ in range(10):
            labels_ = label_embed * train_mask.unsqueeze(2)
            ans_ = ans_embed * train_mask.unsqueeze(2)
            input_embedding = torch.cat(
                [labels_, ans_], dim=-1).view(B, -1)  # B,948x12
            input_embedding = torch.cat(
                [input_embedding, other_embed], dim=-1)
            output = self.output_layer(self.layers(input_embedding))  # B,948
            for b_idx in range(B):
                train_indices = torch.nonzero(
                    input_mask[b_idx, :] == 1).squeeze()  # 80
                scores = torch.min(
                    1.-m(output[b_idx, train_indices]), m(output[b_idx,           train_indices]))  # 80,
                index = train_indices[torch.argmax(scores)]
                train_mask[b_idx, index] = 1
                input_mask[b_idx, index] = 0
        labels_ = label_embed * train_mask.unsqueeze(2)
        ans_ = ans_embed * train_mask.unsqueeze(2)
        input_embedding = torch.cat(
            [labels_, ans_], dim=-1).view(B, -1)  # B,948x12
        input_embedding = torch.cat([input_embedding, other_embed], dim=-1)
        output = self.output_layer(self.layers(input_embedding))  # B,948
        return m(output).detach().cpu().numpy()
    
    def forward_active(self, batch):
        input_labels = batch['input_labels'].to(device).float()  # B,948
        modified_labels = batch['input_labels'].to(device)  # B,948
        input_ans = batch['input_ans'].to(device)-1  # B,948
        input_mask = batch['input_mask'].to(device)  # B,948
        output_labels = batch['output_labels'].to(device).float()  # B,948
        output_mask = batch['output_mask'].to(device)  # B,948
        B = input_labels.shape[0]
        train_mask = torch.zeros(B, 948).long().to(device)
        label_embed = self.label_embeddings(modified_labels)
        ans_embed = self.ans_embeddings(input_ans)  # B,948, 4
        m = nn.Sigmoid()
        other_embed = self.nonselected_layer(
            batch['input_mask'].to(device).float())
        with torch.no_grad():
            for _ in range(10):
                labels_ = label_embed * train_mask.unsqueeze(2)
                ans_ = ans_embed * train_mask.unsqueeze(2)
                input_embedding = torch.cat([labels_, ans_], dim=-1).view(B, -1)  # B,948x12
                input_embedding = torch.cat([input_embedding, other_embed], dim=-1)
                output = self.output_layer(self.layers(input_embedding))  # B,948
                for b_idx in range(B):
                    train_indices = torch.nonzero(input_mask[b_idx, :] == 1).squeeze()
                    scores = torch.min(1.-m(output[b_idx, train_indices]), m(output[b_idx,           train_indices]))  # 80,
                    index = train_indices[torch.argmax(scores)]
                    train_mask[b_idx, index] = 1
                    input_mask[b_idx, index] = 0
        label_embed = label_embed * train_mask.unsqueeze(2)  # B,948, 8
        ans_embed = ans_embed * train_mask.unsqueeze(2)  # B,948, 4
        input_embedding = [label_embed, ans_embed]  # B,948x12
        input_embedding = torch.cat(input_embedding, dim=-1).view(B, -1)
        input_embedding = torch.cat([input_embedding, other_embed], dim=-1)
        output = self.output_layer(self.layers(input_embedding))
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        output_loss = loss_fn(output, output_labels)
        output_loss = output_loss * output_mask
        output_loss = output_loss.sum()/output_mask.sum()
        input_loss = loss_fn(output, input_labels)
        input_loss = input_loss * train_mask
        input_loss = input_loss.sum()/train_mask.sum()
        return input_loss*self.alpha + output_loss, m(output).detach().cpu().numpy()

    def forward(self, batch):
        input_labels = batch['input_labels'].to(device).float()  # B,948
        modified_labels = batch['input_labels'].to(device)  # B,948
        input_ans = batch['input_ans'].to(device)-1  # B,948
        input_mask = batch['input_mask'].to(device)  # B,948
        output_labels = batch['output_labels'].to(device).float()  # B,948
        output_mask = batch['output_mask'].to(device)  # B,948
        #
        B = input_labels.shape[0]
        train_mask = torch.zeros(B, 948).long().to(device)
        for b_idx in range(B):
            train_indices = torch.nonzero(input_mask[b_idx, :] == 1).squeeze()
            indices = torch.randperm(len(train_indices)).to(device)
            train_mask[b_idx, train_indices[indices[:10]]] = 1
        #
        label_embed = self.label_embeddings(
            modified_labels) * train_mask.unsqueeze(2)  # B,948, 8
        # if self.method != 'eer':
        ans_embed = self.ans_embeddings(
            input_ans) * train_mask.unsqueeze(2)  # B,948, 4
        input_embedding = [label_embed, ans_embed]
        input_embedding = torch.cat(input_embedding, dim=-1).view(B, -1)
        other_embed = self.nonselected_layer(input_mask.float())
        input_embedding = torch.cat([input_embedding,other_embed], dim =-1)
        output = self.output_layer(self.layers(input_embedding))
        #
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        output_loss = loss_fn(output, output_labels)
        output_loss = output_loss * output_mask
        output_loss = output_loss.sum()/output_mask.sum()
        input_loss = loss_fn(output, input_labels)
        input_loss = input_loss * train_mask
        input_loss = input_loss.sum()/train_mask.sum()
        m = nn.Sigmoid()
        return input_loss*self.alpha + output_loss, m(output).detach().cpu().numpy()


