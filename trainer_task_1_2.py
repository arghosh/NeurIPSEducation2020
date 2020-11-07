import numpy as np
import torch
import time
import pandas as pd
import random
from utils import open_json, dump_json, compute_auc, compute_accuracy
from dataset_task_1_2 import LSTMDataset, lstm_collate
from model_task_1_2 import LSTMModel, AttentionModel, NCFModel
from copy import deepcopy
from pathlib import Path
import argparse
from shutil import copyfile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc, best_epoch = None, -1


def train_model():
    global best_acc, best_epoch
    batch_idx = 0
    model.train()
    N = len(train_loader.dataset)
    train_loss, all_preds, all_targets = 0., [], []
    val_preds, val_targets = [], []

    for batch in train_loader:
        optimizer.zero_grad()
        loss, output = model(batch)
        #
        if params.task == '1':
            target = batch['labels'].numpy()
            valid_mask = batch['valid_mask'].numpy()
            test_mask = batch['test_mask'].numpy()
            validation_flag = (1-valid_mask)*test_mask
            training_flag = test_mask*valid_mask
        elif params.task == '2':
            target = batch['ans'].numpy()
            valid_mask = batch['valid_mask'].numpy()
            test_mask = batch['test_mask'].numpy()
            validation_flag = (1-valid_mask)*test_mask
            training_flag = test_mask*valid_mask
        loss.backward()
        optimizer.step()
        all_preds.append(output[training_flag == 1])
        all_targets.append(target[training_flag == 1])
        val_preds.append(output[validation_flag == 1])
        val_targets.append(target[validation_flag == 1])
        train_loss += float(loss.detach().cpu().numpy())
        batch_idx += 1

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    val_pred = np.concatenate(val_preds, axis=0)
    val_target = np.concatenate(val_targets, axis=0)
    #model.eval()
    if params.task == '1':
        train_auc = compute_auc(all_target, all_pred)
        val_auc = compute_auc(val_target, val_pred)
        train_accuracy = compute_accuracy(all_target, all_pred)
        val_accuracy = compute_accuracy(val_target, val_pred)
        print('Train Epoch {} Loss: {} train auc: {} train acc: {} val auc: {} val accuracy: {} n_validation : {}'.format(
            epoch, train_loss/batch_idx, train_auc, train_accuracy, val_auc, val_accuracy, val_target.shape))
    if params.task == '2':
        train_accuracy = np.mean(all_target == all_pred)
        val_accuracy = np.mean(val_target == val_pred)
        print('Train Epoch {} Loss: {} train acc: {} val accuracy: {}'.format(
            epoch, train_loss/batch_idx, train_accuracy, val_accuracy))
    if best_acc is None or val_accuracy > best_acc:
        best_acc = val_accuracy
        best_epoch = epoch
    print('Train Epoch {} best val accuracy: {} best epoch: {}'.format(
        epoch, best_acc, best_epoch))
    #model.train()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KT')
    parser.add_argument('--task', type=str, default='2', help='type')
    parser.add_argument('--model', type=str, default='lstm', help='type')
    parser.add_argument('--hidden_dim', type=int, default=128, help='type')
    parser.add_argument('--question_dim', type=int, default=32, help='type')
    parser.add_argument('--user_dim', type=int, default=128, help='type')
    parser.add_argument('--default_dim', type=int, default=16, help='type')
    parser.add_argument('--lr', type=float, default=1e-4, help='type')
    parser.add_argument('--dropout', type=float, default=0.25, help='type')
    parser.add_argument('--valid_prob', type=float, default=0.2, help='type')
    parser.add_argument('--bidirectional', type=int, default=1, help='type')
    parser.add_argument('--dash', type=int, default=0, help='type')
    parser.add_argument('--batch_size', type=int, default=4, help='type')
    parser.add_argument('--head', type=int, default=2, help='type')
    params = parser.parse_args()

    seedNum = 221
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    file_name = [params.task, params.model, params.hidden_dim, params.question_dim,
                 params.lr, params.dropout, params.default_dim, params.valid_prob]
    if params.dash:
        file_name.append(params.dash)
        file_name.append(params.bidirectional)
    if params.model == 'attn':
        file_name.append(params.head)
    if params.dash:
        answer_filename = 'data_task_1_2/answer_dash_metadata_task_1_2_extra.json'
        answer_meta = open_json(answer_filename)
    else:
        answer_meta = None

    train_data = open_json('data_task_1_2/data_1_2.json')
    for d in train_data:
        d['valid_mask'] = [0 if np.random.rand(
        ) < params.valid_prob and ds else 1 for ds in d['test_mask']]

    train_dataset = LSTMDataset(train_data, answer_meta=answer_meta)
    collate_fn = lstm_collate(is_dash=params.dash == 1)
    num_workers = 2
    bs = params.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=bs, num_workers=num_workers, shuffle=True, drop_last=False)

    if params.model == 'lstm':
        model = LSTMModel(n_question=27613, n_user=118971, n_subject=389, task=params.task, s_dim=params.question_dim,
                          n_quiz=17305, n_group=11844, is_dash=params.dash == 1, hidden_dim=params.hidden_dim, q_dim=params.question_dim, dropout=params.dropout, default_dim=params.default_dim, bidirectional=params.bidirectional).to(device)
    elif params.model == 'attn':
        model = AttentionModel(n_question=27613, n_user=118971, n_subject=389, task=params.task, s_dim=params.question_dim,
                               n_quiz=17305, n_group=11844, is_dash=params.dash == 1, hidden_dim=params.hidden_dim, q_dim=params.question_dim, dropout=params.dropout,  default_dim=params.default_dim).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-6)
    start_time = time.time()
    for epoch in range(100):
        if (epoch-best_epoch) > 10:
            break
        train_model()
