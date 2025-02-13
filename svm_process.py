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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('Active learning')

file_path = os.path.dirname(os.path.realpath(__file__))

def feature_extraction_drug(params, gene_filter_path):
    # Data frame
    logger.info('Loading response data')
    path = file_path+'/Data/Cell_Line_Drug_Screening_Data/Response_Data/drug_response_data.txt'
    resp = pd.read_table(path)
    rsp_data = resp.iloc[np.where(resp['Drug_UniqueID']== params['drug_unique_id'])[0]]  
    rsp_data = rsp_data.iloc[np.where(rsp_data['SOURCE']==params['study'])[0]].reset_index(drop=True)

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
  
    if params['use_drug']:
        #Drug features - Mordred fingerprints (single drug) (from CSA data)
        logger.info('Loading drug data')
        if params['study'] == 'GDSC':
            path = file_path + '/Data/CSA_data/data.gdsc2'+'/mordred_gdsc2.csv'
        else:
            path = file_path + '/Data/CSA_data/data.'+params['study'].lower()+'/mordred_'+params['study'].lower()+'.csv'
        drug_feat = pd.read_csv(path)
        drug_feat = drug_feat.iloc[np.where(drug_feat['DrugID']==rsp_data['DRUG'][0])[0]].reset_index(drop=True)
        #Concatenate drug and cell_line features 
        drug_feat = drug_feat.drop(columns = ['DrugID'])
        drug_rep = pd.DataFrame(np.repeat(drug_feat.values, ge_data_lincs.shape[0], axis=0), columns = drug_feat.columns)
        feat = pd.concat([ge_data_lincs, drug_rep], axis=1)
        if params['data_standardize']:
            sc = StandardScaler()
            X = feat.iloc[:,1:].values.astype(float)
            X = sc.fit_transform(X)
            feat.iloc[:,1:] = X
    else:
        if params['data_standardize']:
            sc = StandardScaler()
            X = ge_data_lincs.iloc[:,1:].values.astype(float)
            X = sc.fit_transform(X)
            ge_data_lincs.iloc[:,1:] = X
        feat = ge_data_lincs.copy()
    #Add AUC value
    feat.insert(loc = 1, column = 'AUC', value = ['' for i in range(feat.shape[0])])
    for i in range(len(feat)):
        ind = np.where(rsp_data['CELL']==feat['Sample'][i])[0]
        feat['AUC'][i] = -rsp_data['AUC'].iloc[ind].values[0] # Negative AUC
    logger.info(f'Dimension of data frame is: {feat.shape}')
    if params['use_drug']:
        dir = 'with_drug'
    else:
        dir = 'no_drug'
    save_dir = os.path.join(file_path, 'Output', params['output_dir'] ,str(params['drug_unique_id']+'_'+params['gene_filter']), dir, str(params['data_split_seed'][0])+'_'+str(params['data_split_seed'][1]))
    return feat, save_dir

def feature_extraction_cell_line(params, gene_filter_path):
    # Data frame
    logger.info('Loading response data')
    path = file_path+'/Data/Cell_Line_Drug_Screening_Data/Response_Data/drug_response_data.txt'
    resp = pd.read_table(path)
    rsp_data = resp.iloc[np.where(resp['CELL']== params['cell_line'])[0]]  
    rsp_data = rsp_data.iloc[np.where(rsp_data['SOURCE']==params['study'])[0]].reset_index(drop=True)

    #Drug features - Mordred fingerprints (single drug) (from CSA data)
    logger.info('Loading drug data')
    if params['study'] == 'GDSC':
        path = file_path + '/Data/CSA_data/data.gdsc2'+'/mordred_gdsc2.csv'
    else:
        path = file_path + '/Data/CSA_data/data.'+params['study'].lower()+'/mordred_'+params['study'].lower()+'.csv'
    drug_feat = pd.read_csv(path)
    drugs = pd.unique(rsp_data['DRUG'])
    drug_feat = drug_feat[drug_feat.DrugID.isin(drugs)].reset_index(drop=True)
    
    #Cell-line data (gene expression)
    if params['use_cell_line']:
        logger.info('Loading gene expression data')
        path = file_path + '/Data/Cell_Line_Drug_Screening_Data/Gene_Expression_Data/combined_rnaseq_data_combat'
        ge = pd.read_table(path)
        ge_feat = ge.iloc[np.where(ge['Sample']==rsp_data['CELL'][0])[0]].reset_index(drop=True)
         # Filter genes based on LINCS
        lincs_path = file_path + '/Data/Cell_Line_Drug_Screening_Data/Gene_Expression_Data/'+gene_filter_path # Lincs genes or oncogenes
        with open (lincs_path, 'r') as file:
            lincs_file = file.readlines()
        lincs = ['Sample']
        for l in lincs_file:
            lincs.append(l.replace('\n',''))
        ge_data_lincs = ge_feat[lincs] # Only lincs genes
        #Concatenate drug and cell_line features 
        #ge_feat = ge_feat.drop(columns = ['Sample'])
        ge_data_lincs = ge_data_lincs.drop(columns = ['Sample'])
        ge_rep = pd.DataFrame(np.repeat(ge_data_lincs.values, drug_feat.shape[0], axis=0), columns = ge_data_lincs.columns)
        feat = pd.concat([drug_feat, ge_rep], axis=1)
        if params['data_standardize']:
            sc = StandardScaler()
            X = feat.iloc[:,1:].values.astype(float)
            X = sc.fit_transform(X)
            feat.iloc[:,1:] = X
    else:
        feat = drug_feat.copy()
        if params['data_standardize']:
            sc = StandardScaler()
            X = feat.iloc[:,1:].values.astype(float)
            X = sc.fit_transform(X)
            feat.iloc[:,1:] = X
    feat = feat.rename(columns={'DrugID':'Sample'})
    #Add AUC value
    feat.insert(loc = 1, column = 'AUC', value = ['' for i in range(feat.shape[0])])
    for i in range(len(feat)):
        ind = np.where(rsp_data['DRUG']==feat['Sample'][i])[0]
        feat['AUC'][i] = -rsp_data['AUC'].iloc[ind].values[0] # Negative AUC
    logger.info(f'Dimension of data frame is: {feat.shape}')
    if params['use_cell_line']:
        dir = 'with_cell_line'
    else:
        dir = 'no_cell_line'
    save_dir = os.path.join(file_path, 'Output', params['output_dir'] ,str(params['cell_line']+'_'+params['gene_filter']), dir, str(params['data_split_seed'][0])+'_'+str(params['data_split_seed'][1]))
    return feat, save_dir


def train(X, Y, cost, seeds,train_val_split_size):
    models = []
    logger.info('Starting training...')
    for seed in seeds: # Ensemble over resampling
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=train_val_split_size, random_state=seed) 
        r2_sel = -np.inf
        for i in range(len(cost)): # Optimizing cost parameter
            regressor = SVR(kernel='rbf', C=cost[i])
            regressor.fit(X_train.drop(columns=['Sample']).astype(float),Y_train)
            pred = regressor.predict(X_val.drop(columns=['Sample']).astype(float))
            r2 = r2_score(y_true=Y_val, y_pred=pred)
            if r2 > r2_sel:
                r2_sel = r2
                model_sel = regressor
        models.append(model_sel)
    return(models)

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

def predict(models, X_test, Y_test, feat, num_add, kappa, sampling, random_sample_ratio=0): 
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
    pred.insert(loc=0, column = 'Sample', value = X_test['Sample'].values)

    if sampling == 'al': # Active_learning
        pred['score'] = pred['mean'] + kappa * pred['std']
    if sampling =='uncertainty': # Uncertainty sampling
        pred['score'] = pred['std']
    if sampling =='greedy': # Greedy sampling
        pred['score'] = pred['mean']
    if sampling =='diversity': # Diversity sampling
        num_samples_add = round((1-random_sample_ratio)*num_add)
        if len(pred)<=num_samples_add:
            pred_select = pred
            last = True
        else:
            k_means = KMeans(n_clusters=num_samples_add, n_init =100, max_iter=500)
            k_means.fit(pred.drop(columns =['Sample','AUC', 'mean', 'std'])) 
            labels = k_means.labels_
            centers = k_means.cluster_centers_
            closest, _ = pairwise_distances_argmin_min(centers, pred.drop(columns =['Sample','AUC', 'mean', 'std'])) # The array closest contains the index of the point in X that is closest to each centroid.
            ind_select = list(closest)
            # for i in range(num_samples_add):
            #     ind = list(np.where(labels == i)[0])
            #     ind_select.append(random.sample(ind, 1)[0])
            pred_select = pred.iloc[ind_select].reset_index(drop=True)

    if sampling != 'diversity':    
        pred = pred.sort_values(by = ['score'], ascending=False)
        num_samples_add = round((1-random_sample_ratio)*num_add)
        if len(pred)<=num_samples_add:
            pred_select = pred
            last = True
        else:
            pred_select = pred.iloc[0:num_samples_add]

    feat_new = feat[feat.Sample.isin(pred_select['Sample'])]
    X_new= feat_new.drop(columns=['AUC'])
    Y_new = feat_new['AUC'].astype(float)
    return X_new, Y_new, pred_select, rmse, r2, last

def predict_random(models, X_test, Y_test, feat, num_add, seed_random, random_sample_ratio=0):
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
    if random_sample_ratio == 0:
        num_samples_add = num_add
    else:
        num_samples_add = round((random_sample_ratio)*num_add)
    if len(pred)<=num_samples_add:
        pred_select = pred
        last = True
    else:
        pred_select = pred.sample(n=num_samples_add, random_state = seed_random)

    feat_new = feat[feat.Sample.isin(pred_select['Sample'])]
    X_new= feat_new.drop(columns=['AUC'])
    Y_new = feat_new['AUC'].astype(float)
    return X_new, Y_new, pred_select, rmse, last

def ensemble_process(feat,params, cost, seeds, save_dir, savename):
    random_seeds = params['random_seeds'].copy()
    rmse_hold_out_r = []
    r2_hold_out_r = []
    auc_pred = []
    last = False
    X = feat.drop(columns=['AUC'])
    Y = feat['AUC'].astype(float)
    X_train, X_hold_out_r, Y_train, Y_hold_out_r = train_test_split(X, Y, test_size=params['train_holdout_split_size'], random_state=params['data_split_seed'][0]) # hold-out set based on random sampling
    hold_size = X_hold_out_r.shape[0]
    X_hold_out_r = X_hold_out_r.reset_index(drop=True)
    Y_hold_out_r = Y_hold_out_r.reset_index(drop=True)
    # Diversity based sampling for holdout set
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    #rand_seed = random.randint(0,100)
    #X_train_sample = X_train.sample(n=int(X_train.shape[0]*0.6), random_state = rand_seed) # Randomly sample 60% of samples from train set
    #Y_train_sample = Y_train.sample(n=int(Y_train.shape[0]*0.6), random_state = rand_seed)
    #k_means = KMeans(n_clusters=hold_size, n_init =100, max_iter=500)
    #k_means.fit(X_train_sample.drop(columns ='Sample')) 
    #labels = k_means.labels_
    #centers = k_means.cluster_centers_
    #closest, _ = pairwise_distances_argmin_min(centers, X_train_sample.drop(columns ='Sample')) # The array closest contains the index of the point in X that is closest to each centroid.
    #ind_select = list(closest)
    # ind_select = []
    # for i in range(hold_size):
    #     ind = list(np.where(labels == i)[0])
    #     ind_select.append(random.sample(ind, 1)[0])
    #X_hold_out_d = X_train.iloc[ind_select].reset_index(drop=True)
    #Y_hold_out_d = Y_train.iloc[ind_select].reset_index(drop=True)
    #X_train = X_train.drop(ind_select).reset_index(drop=True)
    #Y_train = Y_train.drop(ind_select).reset_index(drop=True)

    X_test, X_train, Y_test, Y_train = train_test_split(X_train, Y_train, test_size=params['train_test_split_size'], random_state=params['data_split_seed'][1]) 
    # preserving the data splits
    X_train_master = X_train.copy()
    Y_train_master = Y_train.copy()
    X_test_master = X_test.copy()
    Y_test_master = Y_test.copy()
    X_hold_out_r_master = X_hold_out_r.copy()
    Y_hold_out_r_master = Y_hold_out_r.copy()

    if params['sampling'] != 'random':
        random_iter = round(params['hybrid_random_itr_ratio']*((X_test.shape[0])/params['num_add']))
        random_sample_ratio = params['hybrid_random_sample_ratio']
        # Check for errors - parameter mismatch
        if params['hybrid_iter'] and params['hybrid_sample']:
            raise Exception('Error! params[hybrid_iter] and params[hybrid_sample] cannot be True at the same time.')
        if params['hybrid_sample'] and params['hybrid_random_sample_ratio']==0:
            raise Exception('Error! Specify non-zero params[hybrid_random_sample_ratio] for hybrid sample strategy.')
        if params['hybrid_iter'] and params['hybrid_random_itr_ratio']==0:
            raise Exception('Error! Specify non-zero params[hybrid_random_itr_ratio] for hybrid iteration strategy.')
        if params['hybrid_random_itr_ratio']<0 or params['hybrid_random_itr_ratio']>1:
            raise Exception('Error! params[hybrid_random_itr_ratio] should be between 0 and 1')
        if params['hybrid_random_sample_ratio']<0 or params['hybrid_random_sample_ratio']>1:
            raise Exception('Error! params[hybrid_random_sample_ratio] should be between 0 and 1')
        #Info
        if params['hybrid_iter'] and params['hybrid_random_itr_ratio']!=0:
            logger.info('Hybrid iteration strategy with {} random iterations'.format(random_iter))
        if not params['hybrid_iter'] and not params['hybrid_sample'] and params['hybrid_random_itr_ratio']==0:
            logger.info('Pure active learning strategy')
        if params['hybrid_sample'] and params['hybrid_random_sample_ratio']!=0:
            logger.info('Hybrid sample strategy with {} random sampling ratio in every iteration'.format(random_sample_ratio))
        
        while not last: 
            models = train(X_train, Y_train, cost, seeds,params['train_val_split_size'])
            if not params['hybrid_sample']: # For hybrid_iteration and pure active learning (random_iter = 0) strategies
                if random_iter !=0:
                    seed_random =  random_seeds.pop()
                    X_new, Y_new, pred_select, rmse, last = predict_random(models, X_test, Y_test, feat, params['num_add'], seed_random)
                    random_iter = random_iter-1
                else:
                    X_new, Y_new, pred_select, rmse, r2, last = predict(models, X_test, Y_test, feat,params['num_add'], params['kappa'], params['sampling'])
                X_train = pd.concat([X_train, X_new])
                Y_train = pd.concat([Y_train, Y_new])
                test_new = X_test
                test_new['AUC'] = Y_test
                test_new = test_new[~test_new.Sample.isin(pred_select['Sample'])]
                X_test = test_new.drop(columns = ['AUC'])
                Y_test = test_new['AUC']
                auc_pred.append(list(-pred_select['AUC'].values))

            if params['hybrid_sample']: # For hybrid sample strategy
                seed_random =  random_seeds.pop()
                X_new_r, Y_new_r, pred_select_r, rmse_r, last = predict_random(models, X_test, Y_test, feat, params['num_add'],seed_random, random_sample_ratio=random_sample_ratio)
                if last:
                    auc_pred.append(list(-pred_select_r['AUC'].values))
                    break
                test_new = X_test
                test_new['AUC'] = Y_test
                test_new = test_new[~test_new.Sample.isin(pred_select_r['Sample'])]
                X_test = test_new.drop(columns = ['AUC'])
                Y_test = test_new['AUC']
                X_new, Y_new, pred_select, rmse, r2, last = predict(models, X_test, Y_test, feat,params['num_add'], params['kappa'], params['sampling'], random_sample_ratio=random_sample_ratio)
                X_train = pd.concat([X_train, X_new, X_new_r])
                Y_train = pd.concat([Y_train, Y_new, Y_new_r])
                test_new = X_test
                test_new['AUC'] = Y_test
                test_new = test_new[~test_new.Sample.isin(pred_select['Sample'])]
                X_test = test_new.drop(columns = ['AUC'])
                Y_test = test_new['AUC']
                auc_pred.append(list(-(pd.concat([pred_select['AUC'], pred_select_r['AUC']])).values))

            rmse_h, r2_h = predict_hold_out(models, X_hold_out_r, Y_hold_out_r)
            rmse_hold_out_r.append(np.mean(rmse_h))
            r2_hold_out_r.append(np.mean(r2_h))

    if params['sampling'] == 'random':
        # Compare with random sampling
        X_train = X_train_master.copy()
        Y_train = Y_train_master.copy()
        X_test = X_test_master.copy()
        Y_test = Y_test_master.copy()
        X_hold_out_r = X_hold_out_r_master.copy()
        Y_hold_out_r = Y_hold_out_r_master.copy()

        rmse_all = []
        rmse_hold_out_r = []
        r2_hold_out_r = []

        last = False
        while not last:
            models = train(X_train, Y_train, cost, seeds,params['train_val_split_size'])
            seed_random =  random_seeds.pop()
            X_new, Y_new, pred_select, rmse, last = predict_random(models, X_test, Y_test, feat, params['num_add'], seed_random)
            X_train = pd.concat([X_train, X_new])
            Y_train = pd.concat([Y_train, Y_new])
            test_new = X_test
            test_new['AUC'] = Y_test
            test_new = test_new[~test_new.Sample.isin(pred_select['Sample'])]
            X_test = test_new.drop(columns = ['AUC'])
            Y_test = test_new['AUC']
            rmse_all.append(np.mean(rmse))
            rmse_h, r2_h = predict_hold_out(models, X_hold_out_r, Y_hold_out_r)
            rmse_hold_out_r.append(np.mean(rmse_h))
            r2_hold_out_r.append(np.mean(r2_h))
            auc_pred.append(list(-pred_select['AUC'].values))
    output = params
    output['cost'] = cost
    output['seeds'] = seeds
    output['rmse_hold_out'] = rmse_hold_out_r
    output['r2_hold_out'] = r2_hold_out_r
    output['query_auc'] = auc_pred
    with open(save_dir + "/"+ savename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def run_svr(params):
    mode = params['mode']
    sampling = params['sampling']
    study = params['study']
    model = params['model']
    ensemble = params['ensemble']
    output_dir = params['output_dir']
    logger.info(f'MODEL: {model}')
    logger.info(f'STUDY: {study}')
    logger.info(f'MODE: {mode}')
    logger.info(f'SAMPLING STRATEGY: {sampling}')
    logger.info(f'ENSEMBLE STRATEGY: {ensemble}')
    logger.info(f'OUTPUT DIR: {output_dir}')   
     
    cost_all = params['cost_all'] 
    seeds_all = params['seeds_all'] 
    # Choose gene filter file
    if params['gene_filter'] =='lincs':
        gene_filter_path = 'lincs1000_list.txt'
    elif params['gene_filter'] == 'oncogenes':
        gene_filter_path = 'oncogenes_list.txt'

    # Extract features based on mode : cell-line specific, drug-specific or both
    if mode == 'drug':
        feat, save_dir = feature_extraction_drug(params, gene_filter_path) 
    elif mode =='cell_line':
        feat, save_dir = feature_extraction_cell_line(params, gene_filter_path)

    os.makedirs(save_dir, exist_ok=True)

    ############   ENSEMBLE BY RESAMPLING ONLY ############
    if params['ensemble'] == 'resampling' or params['ensemble'] == 'all':
        logger.info('Start Active learning - Resampling only')
        cost = [10]
        savename = 'output_resampl.json'
        seeds = seeds_all
        ensemble_process (feat,params, cost, seeds, save_dir, savename)
        logger.info(f'Done Resampling! Output saved to {savename}')


    ############   ENSEMBLE BY Num_leaves ONLY ############
    if params['ensemble'] == 'hp' or params['ensemble'] == 'all':
        logger.info('Start Active learning - Hyperparameters only')
        cost = cost_all
        seeds = [seeds_all[int(len(seeds_all)/2)-1]]
        savename = 'output_hp.json'
        ensemble_process (feat,params, cost, seeds, save_dir, savename)
        logger.info(f'Done Hyperparameter ensemble! Output saved to {savename}')


    ############   ENSEMBLE BY both Num_leaves and resampling ############
    if params['ensemble'] == 'hp_resampling' or params['ensemble'] == 'all':
        logger.info('Start Active learning - Hyperparameters and Resampling')
        cost = cost_all
        seeds = seeds_all
        savename = 'output_hp_resampl.json'
        ensemble_process (feat,params, cost, seeds, save_dir, savename)
        logger.info(f'Done Resampling and Hyperparameters! Output saved to {savename}')
        logger.info('Done! Shutting down....')

if __name__ == "__main__":
    run_svr(params)

