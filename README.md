# Option Tracing  and Personalized Question Selection

This is the implementation of our submission in the [NeurIPS 2020 Education Challenge](https://competitions.codalab.org/competitions/25449).

> **User: _arighosh_** (Rank 3 in Task 1 and 2, and Rank 1 in Task 4 in the private leaderboard).

## Updates
We published an improved CAT algorithm [BOBCAT: Bilevel Optimization-Based Computerized Adaptive Testing](https://arxiv.org/pdf/2108.07386.pdf) in IJCAI 2021. The code is available at [Github](https://github.com/arghosh/BOBCAT). The proposed algorithm uses similar meta-learning framework used in our winning submission with the addition of a data-driven question selection algorithm. 

If you find the code for Task 4 useful in your research then please cite  
```(bash)
@inproceedings{ghosh-bobcat,
  title     = {BOBCAT: Bilevel Optimization-Based Computerized Adaptive Testing},
  author    = {Ghosh, Aritra and Lan, Andrew},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {2410--2417},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/332},
  url       = {https://doi.org/10.24963/ijcai.2021/332},
}
``` 

We also published Option Tracing paper [Option Tracing: Beyond Correctness Analysis in Knowledge Tracing](https://arxiv.org/abs/2104.09043) in AIED 2021. The code is available at [Github](https://github.com/arghosh/OptionTracing). The paper compares several methods (including methods used for this challenge) for Option tracing.

If you find the code for Task 1 and 2 useful in your research then please cite  
```(bash)
@inproceedings{ghosh2021option,
  title={Option Tracing: Beyond Correctness Analysis in Knowledge Tracing},
  author={Ghosh, Aritra and Raspat, Jay and Lan, Andrew},
  booktitle={International Conference on Artificial Intelligence in Education},
  pages={137--149},
  year={2021},
  organization={Springer}
}
``` 

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

