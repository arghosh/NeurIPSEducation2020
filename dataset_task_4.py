import numpy as np
import torch
from torch.utils import data
import time
import torch
import random
from utils import open_json, dump_json



class FFDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, answers, labels, question_meta, seed =None):
        'Initialization'
        self.answers = answers
        self.labels = labels
        self.seed = seed
        #self.targets = targets
        self.question_meta = question_meta

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.answers)

    def __getitem__(self, index):
        'Generates one sample of data'
        ans = self.answers[index]
        label = self.labels[index]
        observed_index = np.where(label != -1.)[0]
        if not self.seed:
            np.random.shuffle(observed_index)
        else:
            random.Random(index+self.seed).shuffle(observed_index)
        N = len(observed_index)
        target_index = observed_index[-N//5:]
        trainable_index = observed_index[:-N//5]

        input_ans = ans[trainable_index]
        input_label = label[trainable_index]
        input_question = trainable_index
        input_subjects = [
            self.question_meta[str(d)]['child_map'] for d in trainable_index]

        #output_ans = ans[target_index]
        output_label = label[target_index]
        output_question = target_index
        output_subjects = [
            self.question_meta[str(d)]['child_map'] for d in target_index]

        output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
                  'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label),
                  'input_subjects': input_subjects, 'output_subjects': output_subjects, 'input_ans': torch.FloatTensor(input_ans)}

        return output


class ff_collate(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        # output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
        #          'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label),
        #          'input_subjects': input_subjects, 'output_subjects': output_subjects, 'input_ans': torch.FloatTensor(input_ans)}
        B = len(batch)
        input_labels =  torch.zeros(B,948).long()
        output_labels = torch.zeros(B, 948).long()
        input_ans = torch.ones(B, 948).long()
        input_mask  = torch.zeros(B,948).long()
        output_mask = torch.zeros(B, 948).long()
        for b_idx in range(B):
            input_labels[b_idx, batch[b_idx]['input_question'].long()] =  batch[b_idx]['input_label'].long()
            input_ans[b_idx, batch[b_idx]['input_question'].long()] = batch[b_idx]['input_ans'].long()
            input_mask[b_idx, batch[b_idx]['input_question'].long()] = 1
            output_labels[b_idx, batch[b_idx]['output_question'].long()] =  batch[b_idx]['output_label'].long()
            output_mask[b_idx, batch[b_idx]['output_question'].long()] = 1

        output = {'input_labels':input_labels, 'input_ans':input_ans, 'input_mask':input_mask, 'output_labels':output_labels, 'output_mask':output_mask}
        return output

