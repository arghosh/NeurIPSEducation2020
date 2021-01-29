import numpy as np
import json
import time
from utils import dump_json, open_json
import pandas as pd
from multiprocessing import Pool



student_metadata, df, question_metadata = None, None, None

def f(user_id):
    global student_metadata, df, question_metadata
    user_df = df[df.UserId == user_id].sort_values('DateAnswered')
    # Student->  q_ids: NxT,  answers: NxT, is_corrects NxT, test_mask : NxT, valid_mask : NxT,  Times: NxT, Confidence: NxT, groupd_id: NxT, quiz_id: NxT, user_feature: Nx8 , answerId NxT
    q_ids, a_ids, correct_ans, ans, labels, test_mask, times, confidences, group_ids, quiz_ids = [
    ], [], [], [], [], [], [], [], [], []
    subject_ids = []
    #
    user_feature = [0.]*8
    temp = student_metadata[str(user_id)]['feature']
    user_feature[int(temp[0])] = 1.
    user_feature[int(temp[2])+4] = 1.
    user_feature[-1] = (temp[1]-10.)/5.
    #

    last_timestamp = None
    for _, row in user_df.iterrows():
        q_ids.append(int(row['QuestionId']))
        a_ids.append(int(row['AnswerId']))
        correct_ans.append(int(row['CorrectAnswer']))
        ans.append(int(row['AnswerValue']))
        labels.append(int(row['IsCorrect']))
        test_mask.append(int(row['TestMask']))
        confidences.append(float(row['Confidence']))
        group_ids.append(int(row['GroupId']))
        quiz_ids.append(int(row['QuizId']))
        if len(times) > 0:
            times.append(float(pd.Timedelta(
                row['DateAnswered'] - last_timestamp).seconds/86400.))
        else:
            times.append(0.)
        last_timestamp = row['DateAnswered']
    subject_ids = [question_metadata[str(d)]['child_map'] for d in q_ids]
    out = {'user_id': int(user_id), 'user_feature': user_feature, 'subject_ids': subject_ids, 'q_ids': q_ids, 'a_ids': a_ids, 'correct_ans': correct_ans,
           'ans': ans, 'labels': labels, 'test_mask': test_mask, 'times': times, 'confidences': confidences, 'group_ids': group_ids, 'quiz_ids': quiz_ids}
    return out

def featurize():
    global student_metadata, df, question_metadata
    TRAIN_DATA = 'public_data/train_data/train_task_1_2.csv'


    TEST_DATA = 'starter_kit/submission_templates/submission_task_1_2.csv'
    ANSWER_DATA = 'public_data/metadata/answer_metadata_task_1_2.csv'
    QUESTION_SUBJECTS = 'public_data/personal_data/question_metadata_task_1_2.json'
    STUDENT_FEATURES = 'public_data/personal_data/student_metadata_task_1_2.json'

    question_metadata = open_json(QUESTION_SUBJECTS)  # child map
    student_metadata = open_json(STUDENT_FEATURES)

    #AnswerId,DateAnswered,Confidence,GroupId,QuizId,SchemeOfWorkId
    answer_df = pd.read_csv(ANSWER_DATA)[
        ['AnswerId', 'DateAnswered', 'Confidence', 'GroupId', 'QuizId']]
    answer_df['Confidence'].fillna((answer_df['Confidence'].mean()), inplace=True)
    answer_df['DateAnswered'] = pd.to_datetime(
        answer_df['DateAnswered'], errors='coerce')
    print(answer_df.shape)

    #QuestionId,UserId,AnswerId,IsCorrect,CorrectAnswer,AnswerValue
    train_df = pd.read_csv(TRAIN_DATA)
    train_df['TestMask'] = 1
    print('train_df shape: ', train_df.shape)
    #print(train_df.isnull().values.any())
    correct_df = train_df[['QuestionId', 'CorrectAnswer']
                        ].drop_duplicates('QuestionId')
    print('correct qs shape: ', correct_df.shape)

    #,QuestionId,UserId,AnswerId
    test_df = pd.read_csv(TEST_DATA)[['QuestionId', 'UserId', 'AnswerId']]
    test_df = pd.merge(test_df, correct_df, on='QuestionId')
    test_df['IsCorrect'] = 0
    test_df['TestMask'] = 0
    test_df['AnswerValue'] = 1
    print(test_df.shape)
    #print(test_df.isnull().values.any())
    #


    #get answer id info for train
    train_merged_df = pd.merge(train_df, answer_df, on='AnswerId')
    print(train_merged_df.shape)
    print(train_merged_df.isnull().values.any())

    #get answer id info for test
    test_merged_df = pd.merge(test_df, answer_df, on='AnswerId')
    print(test_merged_df.shape)
    print(test_merged_df.isnull().values.any())


    df = pd.concat([train_merged_df, test_merged_df],
                ignore_index=True, sort=False)
    print(df.shape)

    user_ids = df['UserId'].unique()
    user_data = []
    start_time = time.time()
    with  Pool(30) as p:
        user_data = p.map(f, user_ids)
    end_time = time.time()
    print(end_time-start_time)
        

    print('no of user: ',len(user_data))
    dump_json('public_data/converted_datasets/test_1_2.json', user_data)


if __name__ == "__main__":
    featurize()
