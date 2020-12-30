import numpy as np
import json
from utils import dump_json, open_json


def convert_student(infile, outfile, user_meta):
    data = {}
    total_age, total_count = 0., 0.
    with open(infile, 'r') as fp:
        lines = fp.readlines()[1:]
        for line in lines:
            line = line.strip('\n')
            words = line.split(',')
            student_id = int(words[0])
            mf = words[1]
            if mf =='':
                mf = 0
            else:
                mf = int(mf)
            date = words[2]
            if date == '':
                age  = None
            else:
                year, month = int(date.split('-')[0]), int(date.split('-')[1])
                age =  (2020.-year)- (12.-month)/12.
                total_age+=age
                total_count+=1
            pupil = words[3]
            if pupil=='':
                pupil = 2.
            else:
                pupil = (float(pupil))
            data[student_id] = [mf,age,pupil]
    
    mean_age = total_age/total_count
    for k,v in data.items():
        if not v[1]:
            v[1] = mean_age
            data[k] = {'feature': v, 'conf_frac': user_meta.get(
                k, [0., 0., 0., 0., 0., 1.])}
        else:
            data[k] = {'feature': v, 'conf_frac': user_meta.get(k, [0.,0.,0.,0.,0.,1.])}
    dump_json(outfile,data)

def student():
    global user_meta_1, user_meta_3
    input_data = 'public_data/metadata/student_metadata_task_1_2.csv'
    output_data = 'public_data/personal_data/student_metadata_task_1_2.json'
    convert_student(input_data,output_data,user_meta_1)  

    input_data = 'public_data/metadata/student_metadata_task_3_4.csv'
    output_data = 'public_data/personal_data/student_metadata_task_3_4.json'
    convert_student(input_data, output_data, user_meta_3)


def convert_question(infile, outfile, question_meta):
    data = {}
    with open(infile, 'r') as fp:
        lines = fp.readlines()[1:]
        for line in lines:
            line = line.strip('\n')
            words = line.split(',')
            q_id = int(words[0])
            subjects = eval(eval(','.join(words[1:])))
            data[q_id] = {'subjects':subjects, 'conf_frac': question_meta[q_id]}
    dump_json(outfile,data)


def question():
    global  question_meta_1, question_meta_3
    input_data = 'public_data/metadata/question_metadata_task_1_2.csv'
    output_data = 'public_data/personal_data/question_metadata_task_1_2.json'
    convert_question(input_data, output_data,question_meta_1)

    input_data = 'public_data/metadata/question_metadata_task_3_4.csv'
    output_data = 'public_data/personal_data/question_metadata_task_3_4.json'
    convert_question(input_data, output_data,question_meta_3)

def convert_answer(infile, trainfile, output_data):
    answer_meta = {}
    with open(infile, 'r') as fp:
        lines = fp.readlines()[1:]
        for line in lines:
            line = line.strip('\n')
            words = line.split(',')
            a_id = words[0]
            if a_id =='':
                break
            a_id = int(float(a_id))
            conf = words[2]
            if conf =='':
                conf = None
            else:
                conf = int(float(conf))//25
                answer_meta[a_id] = conf
    dump_json(output_data,answer_meta)
    user_meta, question_meta = {},{}
    with open(trainfile, 'r') as fp:
        lines = fp.readlines()[1:]
        for line in lines:
            line = line.strip('\n')
            words = line.split(',')
            q_id = int(words[0])
            u_id = int(words[1])
            a_id = int(words[2])
            if q_id not in question_meta:
                question_meta[q_id] = [0.]*6
            if u_id not in user_meta:
                user_meta[u_id] = [0.]*6
            c_id = answer_meta.get(a_id, 5)
            question_meta[q_id][c_id]+=1
            user_meta[u_id][c_id] += 1
    for k,v in user_meta.items():
        total = sum(v)+0.
        v = [d/total for d in v]
        user_meta[k] = v

    for k, v in question_meta.items():
        total = sum(v)+0.
        v = [d/total for d in v]
        question_meta[k] = v
    
    return user_meta, question_meta


def answer():
    input_data = 'public_data/metadata/answer_metadata_task_3_4.csv'
    output_data = 'public_data/personal_data/answer_metadata_task_3_4.json'
    train_data = 'public_data/train_data/train_task_3_4.csv'
    user_meta_3, question_meta_3 = convert_answer(input_data, train_data,output_data)

    input_data = 'public_data/metadata/answer_metadata_task_1_2.csv'
    output_data = 'public_data/personal_data/answer_metadata_task_1_2.json'
    train_data = 'public_data/train_data/train_task_1_2.csv'
    user_meta_1, question_meta_1 = convert_answer(input_data, train_data, output_data)
    # u_id/q_id = [frac of conf in 0, 25, 50, 75, 100, unknown]
    return user_meta_1, question_meta_1, user_meta_3, question_meta_3

def convert_subjects():
    file_name = 'public_data/metadata/subject_metadata.csv'
    output_data = 'public_data/personal_data/subject_metadata.json'
    data = {}
    cnt = 1
    with open(file_name, 'r') as fp:
        lines = fp.readlines()[1:]
        lines = [line.strip('\n') for line in lines]
        for line in lines:
            words = line.split(',')
            subject_id = int(words[0])
            if words[-2]=='NULL':
                parent_id = 0
            else:
                parent_id = int(words[-2])
            level = int(words[-1])
            name = ','.join(words[1:-2])
            data[subject_id] = {'name':name, 'level': level, 'parent_id':parent_id, 'parents':[parent_id], 'new_id': cnt}
            cnt += 1
    for subject_id in data:
        while True:
            last_parent = data[subject_id]['parents'][-1]
            if last_parent <= 0:
                break
            data[subject_id]['parents'].append(data[last_parent]['parent_id'])
    
    dump_json(output_data, data)
    return data
    

def add_question_ids(infile, subject_metadata):
    question_data = open_json(infile)
    max_q = 0
    for q_id in question_data:
        subjects = question_data[q_id]['subjects']
        new_subject_map = [subject_metadata[d]['new_id'] for d in subjects]
        child_subjects = []
        for d1 in subjects:
            is_ok = True
            for d2 in subjects:
                if d1==d2:
                    continue
                if d1 in subject_metadata[d2]['parents']:
                    is_ok = False
                    break
            if is_ok:
                child_subjects.append(d1)
        question_data[q_id]['new_sub_map'] = new_subject_map
        child_subject_map = [subject_metadata[d]['new_id'] for d in child_subjects]
        question_data[q_id]['child_map'] = child_subject_map
        question_data[q_id]['childs'] = child_subjects 
        child_whole_map = []
        for child in child_subjects:
            parent = subject_metadata[child]['parents']
            parent = [d for d in parent if d]
            parent = [subject_metadata[d]['new_id'] for d in parent]
            child_whole_map.append(parent)
        question_data[q_id]['child_whole_map'] = child_whole_map
        max_q= max(len(child_whole_map),max_q)
        
    print(max_q)
    dump_json(infile, question_data)
def update_question():
    global subject_metadata
    input_data = 'public_data/personal_data/question_metadata_task_1_2.json'
    add_question_ids(input_data,subject_metadata)

    input_data = 'public_data/personal_data/question_metadata_task_3_4.json'
    add_question_ids(input_data, subject_metadata)

user_meta_1, question_meta_1, user_meta_3, question_meta_3 = answer()
student()
question()

subject_metadata = convert_subjects()
update_question()



