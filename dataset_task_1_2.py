import numpy as np
import torch
from torch.utils import data
import time
import torch
from utils import open_json, dump_json


class LSTMDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, answer_meta=None, task='1'):
        'Initialization'
        # QuestionId,UserId,AnswerId,IsCorrect,CorrectAnswer,AnswerValue
        self.data = data
        self.answer_meta = answer_meta

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.answer_meta:
            dash_features = [self.answer_meta.get(str(d), {}).get(
                'f', [0]*17)[1:] for d in self.data[index]['a_ids']]
            return (self.data[index], dash_features)
        return self.data[index]


class lstm_collate(object):
    def __init__(self, is_dash=False):
        self.is_dash = is_dash
        pass

    def __call__(self, batch_raw):
        #{'user_id': int(user_id), 'user_feature': user_feature, 'subject_ids': subject_ids, 'q_ids': q_ids, 'a_ids': a_ids, 'correct_ans': correct_ans,'ans': ans, 'labels': labels, 'test_mask': test_mask, 'times': times, 'confidences': confidences, 'group_ids': group_ids, 'quiz_ids': quiz_ids}
        if self.is_dash:
            batch, d_features = [d[0]
                                 for d in batch_raw], [d[1] for d in batch_raw]
        else:
            batch = batch_raw
        L = [len(d['q_ids']) for d in batch]
        T, B = max(L), len(L)
        LSub = []
        max_sub_len = 0
        for d in batch:
            sub_len = [len(ds) for ds in d['subject_ids']]
            LSub.append(sub_len)
            max_sub_len = max(max_sub_len, max(sub_len))
        q_ids = torch.zeros(T, B).long()
        a_ids = torch.zeros(T, B).long()
        correct_ans = torch.zeros(T, B).long()+1
        ans = torch.zeros(T, B).long()+1
        labels = torch.zeros(T, B).float()
        test_masks = torch.zeros(T, B).long()
        valid_masks = torch.zeros(T, B).long()
        times = torch.zeros(T, B).float()
        confidences = torch.zeros(T, B).float()
        group_ids = torch.zeros(T, B).long()
        quiz_ids = torch.zeros(T, B).long()
        subject_ids = torch.zeros(T, B, max_sub_len).long()
        subject_ids_mask = torch.zeros(T, B, max_sub_len).long()
        u_features = torch.cat(
            [torch.FloatTensor(d['user_feature']).unsqueeze(0) for d in batch], dim=0)
        mask = torch.zeros(T, B).long()
        user_ids = [d['user_id'] for d in batch]
        if self.is_dash:
            dash_features = torch.zeros(T, B, 16)

        for idx in range(B):
            if self.is_dash:
                dash_features[:L[idx], idx, :] = torch.FloatTensor(
                    d_features[idx])
            q_ids[:L[idx], idx] = torch.LongTensor(batch[idx]['q_ids'])
            a_ids[:L[idx], idx] = torch.LongTensor(batch[idx]['a_ids'])
            correct_ans[:L[idx], idx] = torch.LongTensor(
                batch[idx]['correct_ans'])
            ans[:L[idx], idx] = torch.LongTensor(batch[idx]['ans'])
            labels[:L[idx], idx] = torch.FloatTensor(batch[idx]['labels'])
            test_masks[:L[idx], idx] = torch.LongTensor(
                batch[idx]['test_mask'])
            valid_masks[:L[idx], idx] = torch.LongTensor(
                batch[idx]['valid_mask'])
            times[:L[idx], idx] = torch.FloatTensor(batch[idx]['times'])
            confidences[:L[idx], idx] = torch.FloatTensor(
                batch[idx]['confidences'])
            group_ids[:L[idx], idx] = torch.LongTensor(batch[idx]['group_ids'])
            quiz_ids[:L[idx], idx] = torch.LongTensor(batch[idx]['quiz_ids'])
            mask[:L[idx], idx] = 1
            for l_idx in range(L[idx]):
                subject_ids[l_idx, idx, :LSub[idx]
                            [l_idx]] = torch.LongTensor(batch[idx]['subject_ids'][l_idx])
                subject_ids_mask[l_idx, idx, :LSub[idx]
                                 [l_idx]] = 1

        out = {'user_features': u_features, 'subject_ids': subject_ids, 'q_ids': q_ids, 'a_ids': a_ids, 'correct_ans': correct_ans, 'ans': ans, 'labels': labels, 'test_mask': test_masks,
               'valid_mask': valid_masks, 'times': times, 'confidences': confidences, 'group_ids': group_ids, 'quiz_ids': quiz_ids, 'mask': mask, 'subject_mask': subject_ids_mask, 'L': L, 'user_ids': user_ids}
        if self.is_dash:
            out['dash_features'] = dash_features
        return out


if __name__ == "__main__":
    pass
