# Option Tracing  and Personalized Question Selection

This is the implementation of our submission in the [NeurIPS 2020 Education Challenge](https://competitions.codalab.org/competitions/25449).

> **User: _arighosh_** (Rank 3 in Task 1 and 2, and Rank 1 in Task 4 in the private leaderboard).

## Reports

[Option Tracing: Beyond Binary Knowledge Tracing](https://dqanonymousdata.blob.core.windows.net/neurips-public/papers/arigosh/task_1_2_edu_neurips_ghosh_lan.pdf)

[A Meta-learning Framework for Personalized Question Selection](https://dqanonymousdata.blob.core.windows.net/neurips-public/papers/arigosh/task_4_edu_neurips_ghosh_lan.pdf)

## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 1.5.0
- Scikit-learn 0.22.1
- Scipy 1.4.1
- Numpy 1.18.0

## Task 4
For training the model, use the trainer script.
```
python3 trainer_task_4.py
```
We added two models in the _/model/_ folder (including the best performing model in the private leaderboard). For evaluation in the challenge set up, run,
```
python3 local_evaluation_task_4.py
```


## Task 1 and 2
We added a sample dataset in the _data_task_1_2/_ folder. You can download the full dataset from [here][gdrive]. Download the data_task_1_2.zip in the _data_task_1_2/_ folder and unzip it. If you want to run with DAS3H features, download the metadata file from [here][gdrive]. Download in the _data_task_1_2/_ folder and unzip it. 

For training a model, run,
```
python3 trainer_task_1_2.py --bidirectional {0,1} --dash {0,1} --model {attn, lstm} --task {'1', '2'}
```


[gdrive]: https://drive.google.com/drive/folders/1QyH2561LJTLLaGadF47UkTHXo2E1o9zL?usp=sharing



Contact: Aritra Ghosh (aritraghosh.iem@gmail.com).

