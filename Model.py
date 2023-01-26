import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import json
import logging
import sys

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('Active learning')

file_path = os.path.dirname(os.path.realpath(__file__))


def lgb_model(params, X, Y, seeds, lgb_params,train_val_split_size):
    #seeds = [1,2]
    models = []
    for seed in seeds:
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=train_val_split_size, random_state=seed) 
        #create dataset
        lgb_train = lgb.Dataset(X_train.drop(columns=['Sample']).astype(float), Y_train)
        lgb_eval = lgb.Dataset(X_val.drop(columns=['Sample']).astype(float), Y_val, reference=lgb_train)
        # train
        evals_result = {}
        models.append(lgb.train(params,
                        lgb_train,
                        num_boost_round=lgb_params['num_boost_round'], #number of boosting iterations
                        valid_sets=lgb_eval,
                        callbacks=[lgb.early_stopping(stopping_rounds=lgb_params['stopping_rounds']),
                        lgb.record_evaluation(evals_result)]))
    #_ = lgb.plot_metric(evals_result)
    return models

def train(X_train, Y_train, num_leaves, seeds, lgb_params,train_val_split_size):
    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse', #metric(s) to be evaluated on the evaluation set(s)
        'num_leaves': 50,   #max number of leaves in one tree
        'learning_rate': lgb_params['learning_rate'],
        #'feature_fraction': 0.9, #LightGBM will randomly select a subset of features on each iteration (tree)
        #'bagging_fraction': 0.8, #like feature_fraction, but this will randomly select part of data without resampling
        #'bagging_freq': 5, #frequency for bagging
        'verbose': -1,
        #'baggingSeed': 9,
        'random_state': lgb_params['random_state'],
	    'num_threads': 40
        #'device_type':'cuda'
    }
    models = []
    logger.info('Starting training...')
    for i in range(len(num_leaves)):
        params['num_leaves'] = num_leaves[i]
        models.extend(lgb_model(params, X_train, Y_train, seeds, lgb_params,train_val_split_size))
    return(models)

def predict(models, X_train, Y_train, X_test, Y_test, feat, num_add, kappa): 
    # Predictions
    last = False
    pred = pd.DataFrame(columns = range(len(models)))
    rmse = []
    r2 = []
    pred['AUC'] = Y_test.values
    for i in range(len(models)):
        pred[i] = models[i].predict(X_test.drop(columns=['Sample']).astype(float))
        rmse.append(np.sqrt(np.mean((pred[i] - pred['AUC'])**2)))
        r2.append(r2_score(y_true=pred['AUC'], y_pred=pred[i]))
    pred['mean'] = pred.median(axis=1)
    pred['std'] = pred.std(axis=1)
    pred['score'] = pred['mean'] + kappa * pred['std']
    #pred['score'] = kappa * pred['std']
    pred.insert(loc=0, column = 'Sample', value = X_test['Sample'].values)
    pred = pred.sort_values(by = ['score'], ascending=False)
    if len(pred)<=num_add:
        pred_select = pred
        last = True
    else:
        pred_select = pred.iloc[0:num_add]

    feat_new = feat[feat.Sample.isin(pred_select['Sample'])]
    X_new= feat_new.drop(columns=['AUC'])
    Y_new = feat_new['AUC'].astype(float)

    X_train_new = pd.concat([X_train, X_new])
    Y_train_new = pd.concat([Y_train, Y_new])
    test_new = X_test
    test_new['AUC'] = Y_test
    test_new = test_new[~test_new.Sample.isin(pred_select['Sample'])]
    X_test_new = test_new.drop(columns = ['AUC'])
    Y_test_new = test_new['AUC']
    return X_train_new, Y_train_new, X_test_new, Y_test_new, rmse, r2, last

def predict_hold_out(models, X_hold_out, Y_hold_out):
    pred = pd.DataFrame(columns = range(len(models)))
    rmse = []
    r2 = []
    pred['AUC'] = Y_hold_out.values
    for i in range(len(models)):
        pred[i] = models[i].predict(X_hold_out.drop(columns=['Sample']).astype(float))
        rmse.append(np.sqrt(np.mean((pred[i] - pred['AUC'])**2)))
        r2.append(r2_score(y_true=pred['AUC'], y_pred=pred[i]))
    return rmse, r2

def predict_random(models, X_train, Y_train, X_test, Y_test, feat, num_add):
    # Predictions
    last = False
    pred = pd.DataFrame(columns = range(len(models)))
    rmse = []
    pred['AUC'] = Y_test.values
    for i in range(len(models)):
        pred[i] = models[i].predict(X_test.drop(columns=['Sample']).astype(float))
        rmse.append(np.sqrt(np.mean((pred[i] - pred['AUC'])**2)))
    pred['mean'] = pred.median(axis=1)
    pred['std'] = pred.std(axis=1)
    kappa = 1
    pred['score'] = -np.abs(pred['mean']-pred['AUC']) + kappa * pred['std']
    pred.insert(loc=0, column = 'Sample', value = X_test['Sample'].values)
    #pred = pred.sort_values(by = ['score'], ascending=False)
    if len(pred)<=num_add:
        pred_select = pred
        last = True
    else:
        pred_select = pred.sample(n=num_add)

    feat_new = feat[feat.Sample.isin(pred_select['Sample'])]
    X_new= feat_new.drop(columns=['AUC'])
    Y_new = feat_new['AUC'].astype(float)

    X_train_new = pd.concat([X_train, X_new])
    Y_train_new = pd.concat([Y_train, Y_new])
    test_new = X_test
    test_new['AUC'] = Y_test
    test_new = test_new[~test_new.Sample.isin(pred_select['Sample'])]
    X_test_new = test_new.drop(columns = ['AUC'])
    Y_test_new = test_new['AUC']
    return X_train_new, Y_train_new, X_test_new, Y_test_new, rmse, last

def feature_extraction(drug_unique_id, study, gene_filter_path, use_drug):
    # Data frame
    logger.info('Loading response data')
    path = file_path+'/Data/Cell_Line_Drug_Screening_Data/Response_Data/drug_response_data.txt'
    resp = pd.read_table(path)
    rsp_data = resp.iloc[np.where(resp['Drug_UniqueID']== drug_unique_id)[0]]  
    rsp_data = rsp_data.iloc[np.where(rsp_data['SOURCE']==study)[0]].reset_index(drop=True)

    #Cell-line data (gene expression)
    logger.info('Loading gene expression data')
    path = file_path + '/Data/Cell_Line_Drug_Screening_Data/Gene_Expression_Data/combined_rnaseq_data_combat'
    ge = pd.read_table(path)
    cell_lines = pd.unique(rsp_data['CELL'])
    ge_data = ge[ge.Sample.isin(cell_lines)].reset_index(drop=True)
    # Filter genes based on LINCS
    lincs_path = file_path + '/Data/Cell_Line_Drug_Screening_Data/Gene_Expression_Data/'+gene_filter_path # Lincs genes or oncogenes
    with open (lincs_path, 'r') as file:
        lincs_file = file.readlines()
    lincs = ['Sample']
    for l in lincs_file:
        lincs.append(l.replace('\n',''))
    ge_data_lincs = ge_data[lincs] # Only lincs genes
  
    if use_drug:
        #Drug features - Mordred fingerprints (single drug) (from CSA data)
        logger.info('Loading drug data')
        if study == 'GDSC':
            path = file_path + '/Data/CSA_data/data.gdsc1'+'/mordred_gdsc1.csv'
        else:
            path = file_path + '/Data/CSA_data/data.'+study.lower()+'/mordred_'+study.lower()+'.csv'
        drug_feat = pd.read_csv(path)
        drug_feat = drug_feat.iloc[np.where(drug_feat['DrugID']==rsp_data['DRUG'][0])[0]].reset_index(drop=True)
        #Concatenate drug and cell_line features 
        drug_feat = drug_feat.drop(columns = ['DrugID'])
        drug_rep = pd.DataFrame(np.repeat(drug_feat.values, ge_data_lincs.shape[0], axis=0), columns = drug_feat.columns)
        feat = pd.concat([ge_data_lincs, drug_rep], axis=1)
    else:
        feat = ge_data_lincs.copy()
    #Add AUC value
    feat.insert(loc = 1, column = 'AUC', value = ['' for i in range(feat.shape[0])])
    for i in range(len(feat)):
        ind = np.where(rsp_data['CELL']==feat['Sample'][i])[0]
        feat['AUC'][i] = -rsp_data['AUC'].iloc[ind].values[0] # Negative AUC
    logger.info(f'Dimension of data frame is: {feat.shape}')
    return feat

def ensemble_process(feat,train_holdout_split_size, data_split_seed, train_test_split_size, num_leaves, seeds, lgb_params, train_val_split_size, num_add, kappa):
    rmse_all = []
    rmse_hold_out = []
    r2_hold_out = []
    last = False
    X = feat.drop(columns=['AUC'])
    Y = feat['AUC'].astype(float)
    X_train, X_hold_out, Y_train, Y_hold_out = train_test_split(X, Y, test_size=train_holdout_split_size, random_state=data_split_seed[0]) 
    X_test, X_train, Y_test, Y_train = train_test_split(X_train, Y_train, test_size=train_test_split_size, random_state=data_split_seed[1]) 
    while not last: 
        models = train(X_train, Y_train, num_leaves, seeds, lgb_params,train_val_split_size)
        X_train, Y_train, X_test, Y_test, rmse, r2, last = predict(models, X_train, Y_train, X_test, Y_test, feat,num_add, kappa)
        rmse_all.append(np.mean(rmse))
        rmse_h, r2_h = predict_hold_out(models, X_hold_out, Y_hold_out)
        rmse_hold_out.append(np.mean(rmse_h))
        r2_hold_out.append(np.mean(r2_h))

    # Compare with random sampling
    X_train, X_hold_out, Y_train, Y_hold_out = train_test_split(X, Y, test_size=0.15, random_state=data_split_seed[0]) 
    X_test, X_train, Y_test, Y_train = train_test_split(X_train, Y_train, test_size=0.15, random_state=data_split_seed[1]) 
    rmse_all_random = []
    rmse_hold_out_random = []
    r2_hold_out_random = []
    last = False

    while not last:
        models = train(X_train, Y_train, num_leaves, seeds, lgb_params,train_val_split_size)
        X_train, Y_train, X_test, Y_test, rmse, last = predict_random(models, X_train, Y_train, X_test, Y_test, feat, num_add)
        rmse_all_random.append(np.mean(rmse))
        rmse_h, r2_h = predict_hold_out(models, X_hold_out, Y_hold_out)
        rmse_hold_out_random.append(np.mean(rmse_h))
        r2_hold_out_random.append(np.mean(r2_h))

    return rmse_hold_out, r2_hold_out, rmse_hold_out_random, r2_hold_out_random


def run(params):
    drug_unique_id = params['drug_unique_id'] 
    study = params['study']
    use_drug = params['use_drug'] 
    output_dir = params['output_dir']
    num_leaves_all = params['num_leaves_all'] 
    seeds_all = params['seeds_all'] 
    num_add = params['num_add'] 
    data_split_seed = params['data_split_seed'] 
    lgb_params = params['lgb_params'] 
    gene_filter = params['gene_filter']
    train_holdout_split_size = params['train_holdout_split_size']
    train_test_split_size = params['train_test_split_size']
    train_val_split_size = params['train_val_split_size']
    num_sample = params['num_sample']
    kappa = params['kappa']

    if gene_filter =='lincs':
        gene_filter_path = 'lincs1000_list.txt'
    elif gene_filter == 'oncogenes':
        gene_filter_path = 'oncogenes_list.txt'
    # Extract drug and cell-line features
    feat = feature_extraction(drug_unique_id, study, gene_filter_path, use_drug) 
    if use_drug:
        dir = 'with_drug'
    else:
        dir = 'no_drug'
    save_dir = os.path.join(file_path, 'Output', output_dir ,str(drug_unique_id+'_'+study+'_'+gene_filter), dir, str(data_split_seed[0])+'_'+str(data_split_seed[1]))
    os.makedirs(save_dir, exist_ok=True)

    ############   ENSEMBLE BY RESAMPLING ONLY ############
    logger.info('Start Active learning - Resampling only')
    num_leaves = [31]
    savename = 'output_resampl.json'
    seeds = seeds_all
    rmse_hold_out, r2_hold_out, rmse_hold_out_random, r2_hold_out_random = ensemble_process (feat,train_holdout_split_size, data_split_seed, train_test_split_size, num_leaves, seeds, lgb_params, train_val_split_size, num_add, kappa)
    output={'rmse_hold_out':rmse_hold_out, 'r2_hold_out':r2_hold_out, 
            'rmse_hold_out_random':rmse_hold_out_random, 'r2_hold_out_random':r2_hold_out_random,
            'num_leaves':num_leaves, 'seeds':seeds, 'data_split_seed':data_split_seed, 'num_add':num_add, 'gene_filter': gene_filter,
            'drug_unique_id':drug_unique_id, 'study':study, 'use_drug': use_drug, 'lgb_params':lgb_params, 'num_sample':num_sample}
    with open(save_dir + "/"+ savename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    logger.info(f'Done Resampling! Output saved to {savename}')


    ############   ENSEMBLE BY Num_leaves ONLY ############
    logger.info('Start Active learning - Hyperparameters only')
    num_leaves = num_leaves_all
    seeds = [seeds_all[int(len(seeds_all)/2)-1]]
    savename = 'output_hp.json'
    rmse_hold_out, r2_hold_out, rmse_hold_out_random, r2_hold_out_random = ensemble_process (feat,train_holdout_split_size, data_split_seed, train_test_split_size, num_leaves, seeds, lgb_params, train_val_split_size, num_add, kappa)
    output={'rmse_hold_out':rmse_hold_out, 'r2_hold_out':r2_hold_out, 
            'rmse_hold_out_random':rmse_hold_out_random, 'r2_hold_out_random':r2_hold_out_random,
            'num_leaves':num_leaves, 'seeds':seeds, 'data_split_seed':data_split_seed, 'num_add':num_add, 'gene_filter': gene_filter,
            'drug_unique_id':drug_unique_id, 'study':study, 'use_drug': use_drug, 'lgb_params':lgb_params, 'num_sample':num_sample}
    with open(save_dir + "/"+ savename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    logger.info(f'Done Hyperparameter ensemble! Output saved to {savename}')


    ############   ENSEMBLE BY both Num_leaves and resampling ############
    logger.info('Start Active learning - Hyperparameters and Resampling')
    num_leaves = num_leaves_all
    seeds = seeds_all
    savename = 'output_hp_resampl.json'
    rmse_hold_out, r2_hold_out, rmse_hold_out_random, r2_hold_out_random = ensemble_process (feat,train_holdout_split_size, data_split_seed, train_test_split_size, num_leaves, seeds, lgb_params, train_val_split_size, num_add, kappa)
    output={'rmse_hold_out':rmse_hold_out, 'r2_hold_out':r2_hold_out, 
            'rmse_hold_out_random':rmse_hold_out_random, 'r2_hold_out_random':r2_hold_out_random,
            'num_leaves':num_leaves, 'seeds':seeds, 'data_split_seed':data_split_seed, 'num_add':num_add, 'gene_filter': gene_filter,
            'drug_unique_id':drug_unique_id, 'study':study, 'use_drug': use_drug, 'lgb_params':lgb_params, 'num_sample':num_sample}
    with open(save_dir + "/"+ savename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    logger.info(f'Done Resampling and Hyperparameters! Output saved to {savename}')
    logger.info('Done! Shutting down....')

if __name__ == "__main__":
    run(params)

