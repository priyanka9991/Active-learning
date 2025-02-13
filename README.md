# Active Learning

This repository contains scripts for the paper: Vasanthakumari, Priyanka, et al. "A Comprehensive Investigation of Active Learning Strategies for Conducting Anti-Cancer Drug Screening." Cancers 16.3 (2024): 530.  
Abstract:  
Preclinical drug screening experiments for anti-cancer drug discovery typically involve testing candidate drugs against cancer cell lines. This process can be expensive and time consuming since the possible experimental space can be quite huge, involving all of the combinations of candidate cell lines and drugs. Guiding drug screening experiments with active learning strategies could potentially identify promising candidates for successful experimentation. This study investigates various active learning strategies for selecting experiments to generate response data for identifying effective treatments and improving the performance of drug response prediction models. We have demonstrated that most active learning strategies are more efficient than random selection for identifying effective treatments.

## Script descriptions:   
```
1.  active_learning_process.py - Script to run all active learning processess. Make sure to change the following parameter:
study: Cell-line study. eg: CTRPv2, GDSCv2, CCLE
mode: 'drug' - drug specific, 'cell_line' - cell-line specific, 'both' - pan drug, pan cell-line
model: 'lgbm' - LightGBM model, 'dnn'- DNN or 'svr' - Support Vector Regression
sampling: 'al'- uncertainty+greedy, 'uncertainty' - uncertainty, 'greedy' - greedy, 'diversity' - diversity, 'random' - random
cross_test : True if testig with another study (Given in study2) False if testing within the same study
study2: Study name for cross-study testing
R2_filter: True to filter response data using R2 fit of dose response curve
hybrid_iter: True if you need hybrid strategy where few of initial iterations are from random sampling and rest from active learning. Also change the parameter - 'hybrid_random_itr_ratio' to set the iteration ratio
hybrid_sample: True if you need hybrid strategy where some samples are from random sampling in every iteration. Also change the parameter 'hybrid_random_sample_ratio' to set the ratio




```

