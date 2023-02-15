import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import json
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from DNN_model import Network

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('Active learning')

file_path = os.path.dirname(os.path.realpath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    else:
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
        ge_feat = ge_feat.drop(columns = ['Sample'])
        ge_rep = pd.DataFrame(np.repeat(ge_data_lincs.values, drug_feat.shape[0], axis=0), columns = ge_data_lincs.columns)
        feat = pd.concat([drug_feat, ge_rep], axis=1)
    else:
        feat = drug_feat.copy()
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


def ensemble_process(feat,params, num_layers, seeds, save_dir, savename, en_type):
    al_rs = 'al'
    rmse_all = []
    rmse_hold_out = []
    r2_hold_out = []
    last = False
    X = feat.drop(columns=['AUC'])
    Y = feat['AUC'].astype(float)
    X_train, X_hold_out, Y_train, Y_hold_out = train_test_split(X, Y, test_size=params['train_holdout_split_size'], random_state=params['data_split_seed'][0]) 
    X_test, X_train, Y_test, Y_train = train_test_split(X_train, Y_train, test_size=params['train_test_split_size'], random_state=params['data_split_seed'][1]) 
    while not last: 
        models = train(X_train, Y_train, num_layers, seeds, params['dnn_params'],params['train_val_split_size'], save_dir, en_type, al_rs)
        X_train, Y_train, X_test, Y_test, rmse, r2, last = predict(models, X_train, Y_train, X_test, Y_test, feat,params['num_add'], params['kappa'], params['dnn_params'], num_layers)
        rmse_all.append(np.mean(rmse))
        rmse_h, r2_h = predict_hold_out(models, X_hold_out, Y_hold_out, params['dnn_params'], num_layers)
        rmse_hold_out.append(np.mean(rmse_h))
        r2_hold_out.append(np.mean(r2_h))

    # Compare with random sampling
    al_rs = 'rs'
    X_train, X_hold_out, Y_train, Y_hold_out = train_test_split(X, Y, test_size=params['train_holdout_split_size'], random_state=params['data_split_seed'][0]) 
    X_test, X_train, Y_test, Y_train = train_test_split(X_train, Y_train, test_size=params['train_test_split_size'], random_state=params['data_split_seed'][1]) 
    rmse_all_random = []
    rmse_hold_out_random = []
    r2_hold_out_random = []
    last = False
    while not last:
        models = train(X_train, Y_train, num_layers, seeds, params['dnn_params'],params['train_val_split_size'], save_dir, en_type, al_rs)
        X_train, Y_train, X_test, Y_test, rmse, last = predict_random(models, X_train, Y_train, X_test, Y_test, feat, params['dnn_params'], num_layers, params['num_add'])
        rmse_all_random.append(np.mean(rmse))
        rmse_h, r2_h = predict_hold_out(models, X_hold_out, Y_hold_out, params['dnn_params'], num_layers)
        rmse_hold_out_random.append(np.mean(rmse_h))
        r2_hold_out_random.append(np.mean(r2_h))
    output = params
    output['num_layers'] = num_layers
    output['seeds'] = seeds
    output['rmse_hold_out'] = rmse_hold_out
    output['r2_hold_out'] = r2_hold_out
    output['rmse_hold_out_random'] = rmse_hold_out_random
    output['r2_hold_out_random'] = r2_hold_out_random
    with open(save_dir + "/"+ savename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

def train(X, Y, num_layers, seeds, dnn_params,train_val_split_size, save_dir, en_type, al_rs):
    models=[]
    batch_size = dnn_params['batch_size']
    cnt = 0
    for seed in seeds: # Ensemble over resampling
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=train_val_split_size, random_state=seed) 
        X_train = torch.from_numpy(np.vstack(X_train.drop(columns= ['Sample']).to_numpy()).astype('float32'))
        Y_train = torch.from_numpy(np.vstack(Y_train).astype('float32'))
        X_val = torch.from_numpy(np.vstack(X_val.drop(columns= ['Sample']).to_numpy()).astype('float32'))
        Y_val = torch.from_numpy(np.vstack(Y_val).astype('float32'))
        input_size = X_train.shape[1]
        # Pytorch train and val sets
        train = torch.utils.data.TensorDataset(X_train,Y_train)
        val = torch.utils.data.TensorDataset(X_val,Y_val)
        # data loader
        train_loader = torch.utils.data.DataLoader(train, num_workers=10, pin_memory = True, batch_size = batch_size, shuffle = False)
        val_loader = torch.utils.data.DataLoader(val, num_workers=10,pin_memory = True, batch_size = batch_size, shuffle = False)
        for num_layer in num_layers:  # Ensemble over hyperparameters
            h_sizes = [2**(l) for l in range(1,num_layer+1)]
            h_sizes.reverse()
            Model = Network(seed = dnn_params['net_seed'], input_size = input_size, h_sizes = h_sizes, lr = dnn_params['learning_rate'], dropout=dnn_params['drop_out'])
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=5, verbose=False, mode="min")
            checkpoint_callback = ModelCheckpoint(save_top_k=1,monitor='val_loss', mode="min",verbose=False,dirpath=os.path.join(save_dir,'Checkpoints',en_type, al_rs) ,filename='dnn-{epoch:02d}-{val_loss:.2f}-{cnt:02d}', auto_insert_metric_name=False)
            trainer = pl.Trainer(accelerator = 'auto', devices=1, precision=16, strategy="ddp", max_epochs = dnn_params['epochs'],callbacks=[early_stop_callback, checkpoint_callback])
            trainer.fit(Model, train_loader, val_loader)
            cnt=cnt+1
            models.append(checkpoint_callback.best_model_path)
            
    return models

def predict(models, X_train, Y_train, X_test, Y_test, feat, num_add, kappa, dnn_params, num_layers): 
    h_sizes = [2**(l) for l in range(1,num_layers[0]+1)]
    h_sizes.reverse()
    # Predictions
    last = False
    pred = pd.DataFrame(columns = range(len(models)))
    rmse = []
    r2 = []
    pred['AUC'] = Y_test.values
    #Test data
    X = torch.from_numpy(np.vstack(X_test.drop(columns= ['Sample']).to_numpy()).astype('float32'))
    Y = torch.from_numpy(np.vstack(Y_test).astype('float32'))
    input_size = X.shape[1]

    for i in range(len(models)):
        checkpoint = torch.load(models[i])
        Model_test = Network(seed = dnn_params['net_seed'], input_size = input_size, h_sizes = h_sizes, lr = dnn_params['learning_rate'], dropout=dnn_params['drop_out'])
        Model_test.load_state_dict(checkpoint["state_dict"])
        Model_test.eval()
        with torch.no_grad():
            pred[i] = Model_test(X)
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

def predict_hold_out(models, X_hold_out, Y_hold_out, dnn_params, num_layers):
    h_sizes = [2**(l) for l in range(1,num_layers[0]+1)]
    h_sizes.reverse()
    pred = pd.DataFrame(columns = range(len(models)))
    rmse = []
    r2 = []
    pred['AUC'] = Y_hold_out.values
    #Test data
    X = torch.from_numpy(np.vstack(X_hold_out.drop(columns= ['Sample']).to_numpy()).astype('float32'))
    Y = torch.from_numpy(np.vstack(Y_hold_out).astype('float32'))
    input_size = X.shape[1]
    for i in range(len(models)):
        checkpoint = torch.load(models[i])
        Model_test = Network(seed = dnn_params['net_seed'], input_size = input_size, h_sizes = h_sizes, lr = dnn_params['learning_rate'], dropout=dnn_params['drop_out'])
        Model_test.load_state_dict(checkpoint["state_dict"])
        Model_test.eval()
        with torch.no_grad():
            pred[i] = Model_test(X)
        rmse.append(np.sqrt(np.mean((pred[i] - pred['AUC'])**2)))
        r2.append(r2_score(y_true=pred['AUC'], y_pred=pred[i]))
    return rmse, r2

def predict_random(models, X_train, Y_train, X_test, Y_test, feat, dnn_params, num_layers, num_add):
    h_sizes = [2**(l) for l in range(1,num_layers[0]+1)]
    h_sizes.reverse()
    # Predictions
    last = False
    pred = pd.DataFrame(columns = range(len(models)))
    rmse = []
    r2 = []
    pred['AUC'] = Y_test.values
    #Test data
    X = torch.from_numpy(np.vstack(X_test.drop(columns= ['Sample']).to_numpy()).astype('float32'))
    Y = torch.from_numpy(np.vstack(Y_test).astype('float32'))
    input_size = X.shape[1]
    for i in range(len(models)):
        checkpoint = torch.load(models[i])
        Model_test = Network(seed = dnn_params['net_seed'], input_size = input_size, h_sizes = h_sizes, lr = dnn_params['learning_rate'], dropout=dnn_params['drop_out'])
        Model_test.load_state_dict(checkpoint["state_dict"])
        Model_test.eval()
        with torch.no_grad():
            pred[i] = Model_test(X)
        rmse.append(np.sqrt(np.mean((pred[i] - pred['AUC'])**2)))
        r2.append(r2_score(y_true=pred['AUC'], y_pred=pred[i]))
    pred['mean'] = pred.median(axis=1)
    pred['std'] = pred.std(axis=1)
    pred.insert(loc=0, column = 'Sample', value = X_test['Sample'].values)

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

def run_dnn(params):
    mode = params['mode']
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
    logger.info('Start Active learning - Resampling only')
    en_type = 'resampling'
    num_layers = [params['num_layers_all'][round(len(params['num_layers_all'])/2)]]
    savename = 'output_resampl.json'
    seeds = params['seeds_all']
    ensemble_process (feat,params, num_layers, seeds, save_dir, savename, en_type)
    logger.info(f'Done Resampling! Output saved to {savename}')

    ############   ENSEMBLE BY Hyperparameters ONLY ############
    logger.info('Start Active learning - Hyperparameters only')
    en_type = 'hp'
    num_layers = params['num_layers_all']
    seeds = [params['seeds_all'][int(len(params['seeds_all'])/2)-1]]
    savename = 'output_hp.json'
    ensemble_process (feat,params, num_layers, seeds, save_dir, savename, en_type)
    logger.info(f'Done Hyperparameter ensemble! Output saved to {savename}')

    ############   ENSEMBLE BY both Hyperparameters and resampling ############
    logger.info('Start Active learning - Hyperparameters and Resampling')
    en_type = 'hp_resampling'
    num_layers = params['num_layers_all']
    seeds = params['seeds_all']
    savename = 'output_hp_resampl.json'
    ensemble_process (feat,params, num_layers, seeds, save_dir, savename, en_type)
    logger.info(f'Done Resampling and Hyperparameters! Output saved to {savename}')
    logger.info('Done! Shutting down....')

if __name__ == "__main__":
    run_dnn(params)

