import os
import numpy as np
import torch
from copy import deepcopy
from model_task_4 import FFModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Submission:
    """
    API Wrapper class which loads a saved model upon construction, and uses this to implement an API for feature 
    selection and missing value prediction.
    """

    def __init__(self):
        #file_name = 'model/model_task_4_ff_1024_4_2e-05_0.25_0.75_128_256_e_266.pt'
        file_name = 'model/model_task_4_ff_1024_4_2e-05_0.5_0.67_128_1024_e_330.pt'
        words = file_name.split('_')
        hidden_dim = int(words[4])
        question_dim = int(words[5])
        dropout = float(words[7])
        concat_dim = int(words[10])
        concat_hidden_dim = int(words[9])

        checkpoint = torch.load(file_name, map_location=device)
        self.model = FFModel(hidden_dim=hidden_dim, dim=question_dim, dropout=dropout, concat_dim=concat_dim, concat_hidden_dim=concat_hidden_dim).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.set_mask = None
        print("Loaded params:")
    def select_feature(self, masked_data, masked_binary_data, can_query):
        """
        Use your model to select a new feature to observe from a list of candidate features for each student in the
            input data, with the goal of selecting features with maximise performance on a held-out set of answers for
            each student.
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing data revealed to the model
                at the current step. Unobserved values are denoted by -1.
            masked_binary_data (np.array): Array of shape (num_students, num_questions) containing binary data revealed 
                to the model at the current step. Unobserved values are denoted by -1.
            can_query (np.array): Binary array of shape (num_students, num_questions), indicating which features can be 
                queried by the model in the current step.
        Returns:
            selections (list): List of ints, length num_students, containing the index of the feature selected to query 
            for each student (row) in the dataset.
        """
        # Use the loaded model to perform feature selection.
        # selections = self.model.select_feature(masked_data, can_query)
        B = can_query.shape[0]
        masked_data = torch.from_numpy(masked_data).long().to(device)
        train_mask = torch.zeros_like(masked_data).to(device)
        train_mask[masked_data != 0] = 1
        input_labels = torch.from_numpy(masked_binary_data).long().to(device)
        input_ans = (masked_data.clone()-1) * train_mask
        m = torch.nn.Sigmoid()
        selections = []

        input_mask = torch.from_numpy(deepcopy(can_query)).long().to(device)

        if self.set_mask is None:
            self.set_mask = input_mask
        other_embed = self.model.nonselected_layer(self.set_mask.float())
        with torch.no_grad():
            label_embed = self.model.label_embeddings(input_labels)
            ans_embed = self.model.ans_embeddings(input_ans)  # B,948, 4
            labels_ = label_embed * train_mask.unsqueeze(2)
            ans_ = ans_embed * train_mask.unsqueeze(2)
            input_embedding = torch.cat(
                [labels_, ans_], dim=-1).view(B, -1)  # B,948x12
            input_embedding = torch.cat(
                [input_embedding, other_embed], dim=-1)
            output = self.model.output_layer(
                self.model.layers(input_embedding))  # B,948
            for b_idx in range(B):
                train_indices = torch.nonzero(
                    input_mask[b_idx, :] == 1).squeeze()  # 80
                scores = torch.min(
                    1.-m(output[b_idx, train_indices]), m(output[b_idx,           train_indices]))
                index = train_indices[torch.argmax(scores)]
                selections.append(int(index))
        return selections

    def update_model(self, masked_data, masked_binary_data, can_query):
        """
        Update the model to incorporate newly revealed data if desired (e.g. training or updating model state).
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing categorical data revealed to 
                the model at the current step. Unobserved values are denoted by -1.
            masked_binary_data (np.array): Array of shape (num_students, num_questions) containing binary data revealed 
                to the model at the current step. Unobserved values are denoted by -1.
            can_query (np.array): Binary array of shape (num_students, num_questions), indicating which features can be 
                queried by the model in the current step.
        Returns:
            selections (list): List of ints, length num_students, containing the index of the feature selected to query 
            for each student (row) in the dataset.
        """
        # Update the model after new data has been revealed, if required.
        pass

    def predict(self, masked_data, masked_binary_data):
        """
        Use your model to predict missing binary values in the input data. Both categorical and binary versions of the
        observed input data are available for making predictions with.
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing categorical data revealed to 
                the model at the current step. Unobserved values are denoted by -1.
            masked_binary_data (np.array): Array of shape (num_students, num_questions) containing binary data revealed 
                to the model at the current step. Unobserved values are denoted by -1.
        Returns:
            predictions (np.array): Array of shape (num_students, num_questions) containing predictions for the
                unobserved values in `masked_binary_data`. The values given to the observed data in this array will be 
                ignored.
        """
        # Use the loaded model to perform missing value prediction.
        # predictions = self.model.predict(masked_data, masked_binary_data)
        B = masked_data.shape[0]
        masked_data = torch.from_numpy(masked_data).long().to(device)
        train_mask = torch.zeros_like(masked_data).to(device)
        train_mask[masked_data != 0] = 1
        input_labels = torch.from_numpy(masked_binary_data).long().to(device)
        input_ans = (masked_data.clone()-1) * train_mask
        other_embed = self.model.nonselected_layer(
            self.set_mask.float())
        with torch.no_grad():
            label_embed = self.model.label_embeddings(input_labels)
            ans_embed = self.model.ans_embeddings(input_ans)  # B,948, 4
            labels_ = label_embed * train_mask.unsqueeze(2)
            ans_ = ans_embed * train_mask.unsqueeze(2)
            input_embedding = torch.cat(
                [labels_, ans_], dim=-1).view(B, -1)  # B,948x12
            input_embedding = torch.cat(
                [input_embedding, other_embed], dim=-1)
            output = self.model.output_layer(
                self.model.layers(input_embedding))  # B,948
            output[output > 0] = 1
            output[output <= 0] = 0
        predictions = output.detach().cpu().numpy()
        self.set_mask = None

        return predictions
