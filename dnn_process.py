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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, DeviceStatsMonitor
from DNN_hypermodel import Network
import warnings
import optuna
from optuna.trial import TrialState
from optuna.integration import PyTorchLightningPruningCallback

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
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
        feat['AUC'][i] = rsp_data['AUC'].iloc[ind].values[0] # Negative AUC
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
        feat['AUC'][i] = rsp_data['AUC'].iloc[ind].values[0] # Negative AUC
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
    dropouts_al = []
    dropouts_rs = []
    auc_pred_al = []
    auc_pred_rs = []
    last = False
    X = feat.drop(columns=['AUC'])
    Y = feat['AUC'].astype(float)
    X_train, X_hold_out, Y_train, Y_hold_out = train_test_split(X, Y, test_size=params['train_holdout_split_size'], random_state=params['data_split_seed'][0]) 
    X_test, X_train, Y_test, Y_train = train_test_split(X_train, Y_train, test_size=params['train_test_split_size'], random_state=params['data_split_seed'][1]) 
    
    random_iter = round(params['hybrid_random_itr_ratio']*((X.shape[0]-X_train.shape[0])/params['num_add']))
    random_sample_ratio = params['hybrid_random_sample_ratio']
    random_seeds = [] # stores all the seeds used for random iterations in AL during hybrid strategies
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
        models, dropouts = train(X_train, Y_train, num_layers, seeds, params['dnn_params'],params['train_val_split_size'],save_dir, en_type, al_rs)
        dropouts_al.append(dropouts)
        if not params['hybrid_sample']: # For hybrid_iteration and pure active learning (random_iter = 0) strategies
            if random_iter !=0:
                seed_random =  random.randint(0, 100)
                random_seeds.append(seed_random)
                X_new, Y_new, pred_select, rmse, last =  predict_random(models, X_test, Y_test, feat, params['dnn_params'], num_layers, params['num_add'], dropouts, seed_random)
                random_iter = random_iter-1
            else:
                X_new, Y_new, pred_select, rmse, r2, last = predict(models, X_test, Y_test, feat,params['num_add'], params['kappa'], params['dnn_params'], num_layers, params['sampling'], dropouts)
            X_train = pd.concat([X_train, X_new])
            Y_train = pd.concat([Y_train, Y_new])
            test_new = X_test
            test_new['AUC'] = Y_test
            test_new = test_new[~test_new.Sample.isin(pred_select['Sample'])]
            X_test = test_new.drop(columns = ['AUC'])
            Y_test = test_new['AUC']
            auc_pred_al.append(list(-pred_select['AUC'].values))

        if params['hybrid_sample']: # For hybrid sample strategy
            seed_random =  random.randint(0, 100)
            random_seeds.append(seed_random)
            X_new_r, Y_new_r, pred_select_r, rmse_r, last = predict_random(models, X_test, Y_test, feat, params['dnn_params'], num_layers, params['num_add'], dropouts, seed_random, random_sample_ratio=random_sample_ratio)
            if last:
                auc_pred_al.append(list(-pred_select_r['AUC'].values))
                break
            test_new = X_test
            test_new['AUC'] = Y_test
            test_new = test_new[~test_new.Sample.isin(pred_select_r['Sample'])]
            X_test = test_new.drop(columns = ['AUC'])
            Y_test = test_new['AUC']
            X_new, Y_new, pred_select, rmse, r2, last = predict(models, X_test, Y_test, feat,params['num_add'], params['kappa'], params['dnn_params'], num_layers, params['sampling'], dropouts, random_sample_ratio=random_sample_ratio)
            X_train = pd.concat([X_train, X_new, X_new_r])
            Y_train = pd.concat([Y_train, Y_new, Y_new_r])
            test_new = X_test
            test_new['AUC'] = Y_test
            test_new = test_new[~test_new.Sample.isin(pred_select['Sample'])]
            X_test = test_new.drop(columns = ['AUC'])
            Y_test = test_new['AUC']
            auc_pred_al.append(list(-(pd.concat([pred_select['AUC'], pred_select_r['AUC']])).values))

        rmse_h, r2_h = predict_hold_out(models, X_hold_out, Y_hold_out, params['dnn_params'], num_layers, dropouts)
        rmse_hold_out.append(np.mean(rmse_h))
        r2_hold_out.append(np.mean(r2_h))

    # Compare with random sampling
    X_train, X_hold_out, Y_train, Y_hold_out = train_test_split(X, Y, test_size=params['train_holdout_split_size'], random_state=params['data_split_seed'][0]) 
    X_test, X_train, Y_test, Y_train = train_test_split(X_train, Y_train, test_size=params['train_test_split_size'], random_state=params['data_split_seed'][1]) 
    rmse_all_random = []
    rmse_hold_out_random = []
    r2_hold_out_random = []
    last = False
    while not last:
        models, dropouts = train(X_train, Y_train, num_layers, seeds, params['dnn_params'],params['train_val_split_size'],save_dir, en_type, al_rs)
        dropouts_rs.append(dropouts)
        if len(random_seeds)!=0:
            seed_random = random_seeds[0]
            random_seeds = random_seeds[1:]
        else:
            seed_random =  random.randint(0, 100)
        X_new, Y_new, pred_select, rmse, last = predict_random(models, X_test, Y_test, feat, params['dnn_params'], num_layers, params['num_add'], dropouts, seed_random)
        X_train = pd.concat([X_train, X_new])
        Y_train = pd.concat([Y_train, Y_new])
        test_new = X_test
        test_new['AUC'] = Y_test
        test_new = test_new[~test_new.Sample.isin(pred_select['Sample'])]
        X_test = test_new.drop(columns = ['AUC'])
        Y_test = test_new['AUC']
        rmse_all_random.append(np.mean(rmse))
        rmse_h, r2_h = predict_hold_out(models, X_hold_out, Y_hold_out, params['dnn_params'], num_layers, dropouts)
        rmse_hold_out_random.append(np.mean(rmse_h))
        r2_hold_out_random.append(np.mean(r2_h))
        auc_pred_rs.append(list(-pred_select['AUC'].values))
    output = params
    output['num_layers'] = num_layers
    output['seeds'] = seeds
    output['rmse_hold_out'] = rmse_hold_out
    output['r2_hold_out'] = r2_hold_out
    output['rmse_hold_out_random'] = rmse_hold_out_random
    output['r2_hold_out_random'] = r2_hold_out_random
    output['query_auc_al'] = auc_pred_al
    output['query_auc_rs'] = auc_pred_rs
    output['dropouts_al'] = dropouts_al
    output['dropouts_rs'] = dropouts_rs
    with open(save_dir + "/"+ savename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def objective(trial: optuna.trial.Trial)-> float:
    n_layers = 4
    layer1 = 614
    rem = round((layer1-1)/n_layers)
    out_features = []
    layer = layer1
    for i in range(n_layers):
        out_features.append(layer)
        layer = layer-rem
    #out_features = [33,111,123]
    #dropouts = [trial.suggest_float("dropout{}".format(i), 0.2, 0.5)for i in range(n_layers)]
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    dropouts = [dropout] * n_layers
    lr = 0.0022
    dnn_params = params['dnn_params']
    Model = Network(seed = dnn_params['net_seed'], input_size = input_size, n_layers = n_layers, dropouts=dropouts, out_features = out_features, lr=lr)
    #early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(logger=True, enable_checkpointing=False, accelerator = 'gpu', devices=1,precision=16,max_epochs = dnn_params['epochs'],callbacks=[early_stop_callback])
    hyperparameters = dict(dropout=dropouts)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(Model, train_loader, val_loader)
           
    return trainer.callback_metrics["val_loss"].item()

def train(X, Y, num_layers, seeds, dnn_params,train_val_split_size, save_dir, en_type, al_rs):
    models=[]
    batch_size = dnn_params['batch_size']
    cnt=0
    #out_features = [33,111,123]
    n_layers = 4
    layer1 = 614
    rem = round((layer1-1)/n_layers)
    out_features = []
    layer = layer1
    for i in range(n_layers):
        out_features.append(layer)
        layer = layer-rem
    lr = 0.0022
    
    for seed in seeds: # Ensemble over resampling
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=train_val_split_size, random_state=seed) 
        X_train = torch.from_numpy(np.vstack(X_train.drop(columns= ['Sample']).to_numpy()).astype('float32'))
        Y_train = torch.from_numpy(np.vstack(Y_train).astype('float32'))
        X_val = torch.from_numpy(np.vstack(X_val.drop(columns= ['Sample']).to_numpy()).astype('float32'))
        Y_val = torch.from_numpy(np.vstack(Y_val).astype('float32'))
        global input_size
        input_size = X_train.shape[1]
        # Pytorch train and val sets
        train = torch.utils.data.TensorDataset(X_train,Y_train)
        val = torch.utils.data.TensorDataset(X_val,Y_val)
        # data loader
        global train_loader
        global val_loader
        train_loader = torch.utils.data.DataLoader(train,num_workers=0, pin_memory = True, batch_size = batch_size, shuffle = False)
        val_loader = torch.utils.data.DataLoader(val,num_workers=0, pin_memory = True, batch_size = batch_size, shuffle = False)
        for num_layer in num_layers:  # Ensemble over hyperparameters
            ###Dropout optimization
            pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
            study = optuna.create_study(direction="minimize", pruner=pruner)
            study.optimize(objective, n_trials=params['n_trials'], timeout=600)
            print("Number of finished trials: {}".format(len(study.trials)))
            print("Best trial:")
            trial = study.best_trial
            print("  Value: {}".format(trial.value))
            print("  Params: ")
            best_params = {}
            dropouts = []
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                best_params[key] = value
                #dropouts.append(value)
                dropouts = [value] * n_layers
            ##
            Model = Network(seed = dnn_params['net_seed'], input_size = input_size, n_layers = num_layer, dropouts=dropouts, out_features = out_features, lr=lr)
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10, verbose=False, mode="min")
            checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=1, mode="min",verbose=False,dirpath=os.path.join(save_dir,'Checkpoints',en_type, al_rs),filename='dnn-{epoch:02d}-{val_loss:.2f}-{cnt:02d}', auto_insert_metric_name=False)
            trainer = pl.Trainer(profiler="simple",accelerator = 'gpu', devices=1, strategy="ddp",precision=16,max_epochs = dnn_params['epochs'],callbacks=[early_stop_callback, checkpoint_callback, DeviceStatsMonitor()])
            trainer.fit(Model, train_loader, val_loader)
            cnt=cnt+1
            models.append(checkpoint_callback.best_model_path)
            
    return models, dropouts

def predict(models, X_test, Y_test, feat, num_add, kappa, dnn_params, num_layers, sampling, dropouts, random_sample_ratio=0): 
    n_layers = 4
    layer1 = 614
    rem = round((layer1-1)/n_layers)
    out_features = []
    layer = layer1
    for i in range(n_layers):
        out_features.append(layer)
        layer = layer-rem
    lr = 0.0022
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
        Model_test = Network(seed = dnn_params['net_seed'], input_size = input_size, n_layers = num_layers[0], dropouts=dropouts, out_features = out_features, lr=lr)
        Model_test.load_state_dict(checkpoint["state_dict"])
        Model_test.eval()
        with torch.no_grad():
            pred[i] = Model_test(X)
        rmse.append(np.sqrt(np.mean((pred[i] - pred['AUC'])**2)))
        r2.append(r2_score(y_true=pred['AUC'], y_pred=pred[i]))
    pred['mean'] = pred.median(axis=1)
    pred['std'] = pred.std(axis=1)
    if sampling == 'al': # Active_learning
        pred['score'] = -pred['mean'] + kappa * pred['std']
    if sampling =='uncertainty': # Uncertainty sampling
        pred['score'] = pred['std']
    if sampling =='greedy': # Greedy sampling
        pred['score'] = -pred['mean']
    pred.insert(loc=0, column = 'Sample', value = X_test['Sample'].values)
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

def predict_hold_out(models, X_hold_out, Y_hold_out, dnn_params, num_layers, dropouts):
    n_layers = 4
    layer1 = 614
    rem = round((layer1-1)/n_layers)
    out_features = []
    layer = layer1
    for i in range(n_layers):
        out_features.append(layer)
        layer = layer-rem
    lr = 0.0022
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
        Model_test = Network(seed = dnn_params['net_seed'], input_size = input_size, n_layers = num_layers[0], dropouts=dropouts, out_features = out_features, lr=lr)
        Model_test.load_state_dict(checkpoint["state_dict"])
        Model_test.eval()
        with torch.no_grad():
            pred[i] = Model_test(X)
        rmse.append(np.sqrt(np.mean((pred[i] - pred['AUC'])**2)))
        r2.append(r2_score(y_true=pred['AUC'], y_pred=pred[i]))
    return rmse, r2

def predict_random(models, X_test, Y_test, feat, dnn_params, num_layers, num_add, dropouts, seed_random, random_sample_ratio=0):
    n_layers = 4
    layer1 = 614
    rem = round((layer1-1)/n_layers)
    out_features = []
    layer = layer1
    for i in range(n_layers):
        out_features.append(layer)
        layer = layer-rem
    lr = 0.0022
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
        Model_test = Network(seed = dnn_params['net_seed'], input_size = input_size, n_layers = num_layers[0], dropouts=dropouts, out_features = out_features, lr=lr)
        Model_test.load_state_dict(checkpoint["state_dict"])
        Model_test.eval()
        with torch.no_grad():
            pred[i] = Model_test(X)
        rmse.append(np.sqrt(np.mean((pred[i] - pred['AUC'])**2)))
        r2.append(r2_score(y_true=pred['AUC'], y_pred=pred[i]))
    pred['mean'] = pred.median(axis=1)
    pred['std'] = pred.std(axis=1)
    pred.insert(loc=0, column = 'Sample', value = X_test['Sample'].values)
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

def run_dnn(params1):
    global params
    params = params1
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
        en_type = 'resampling'
        #num_layers = [params['num_layers_all'][round(len(params['num_layers_all'])/2)]]
        num_layers = [4]
        savename = 'output_resampl.json'
        seeds = params['seeds_all']
        ensemble_process (feat,params, num_layers, seeds, save_dir, savename, en_type)
        logger.info(f'Done Resampling! Output saved to {savename}')

    ############   ENSEMBLE BY Hyperparameters ONLY ############
    if params['ensemble'] == 'hp' or params['ensemble'] == 'all':
        logger.info('Start Active learning - Hyperparameters only')
        en_type = 'hp'
        num_layers = params['num_layers_all']
        seeds = [params['seeds_all'][int(len(params['seeds_all'])/2)-1]]
        savename = 'output_hp.json'
        ensemble_process (feat,params, num_layers, seeds, save_dir, savename, en_type)
        logger.info(f'Done Hyperparameter ensemble! Output saved to {savename}')

    ############   ENSEMBLE BY both Hyperparameters and resampling ############
    if params['ensemble'] == 'hp_resampling' or params['ensemble'] == 'all':
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

