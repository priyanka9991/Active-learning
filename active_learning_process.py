from lgbm_process import run_lgbm
from dnn_process import run_dnn
from svm_process import run_svr
import random 
import os
import pandas as pd
import numpy as np

study = 'CTRP'
mode = 'drug'   # mode = 'drug' or 'cell_line' or 'both'
model = 'lgbm'    # model = 'lgbm' or 'dnn'or 'svr' 
sampling = 'al' # sampling = 'al', 'uncertainty', 'greedy', 'diversity', 'random'
ensemble = 'resampling' # Choose from 'resampling', 'hp', 'hp_resampling', or 'all'
cross_test=True
study2='CCLE'
R2_filter = False # Filter using R2 fit of dose response curve
hybrid_iter = False # Hybrid strategy where few of initial iterations are from random sampling and rest from active learning. Parameter - 'hybrid_random_itr_ratio'
hybrid_sample = True # Hybrid strategy where some samples are from random sampling in every iteration. Parameter - 'hybrid_random_sample_ratio'
hybrid_random_itr_ratio= 0 # Percentage of iterations coming from random sampling
hybrid_random_sample_ratio= 0.5
expt_name = 'lgbm_HS_0.5_cross_CCLE'
file_path = os.path.dirname(os.path.realpath(__file__))

# TODO Input study, model typr(lgbm or dnn), cell_specific, drug_specific or both as command line arguments
# Add DNN model pipeline
# Transfer learning between different studies
# Use all te studies

# Get list of drugs and cell-lines
if mode =='cell_line':
    path = file_path+'/Data/Cell_Line_Drug_Screening_Data/Response_Data/drug_response_data.txt'
    resp = pd.read_table(path)
    ind = [ind for ind,cell in enumerate(list(resp['CELL'])) if cell.split('.')[0] == study]
    aid = resp.iloc[ind].reset_index(drop=True)
    cell_id = aid[['CELL','CCLE_CCL_UniqueID','NCI60_CCL_UniqueID' ]]
    cell_uniq = cell_id['CELL'].drop_duplicates()
    cell_names = pd.DataFrame({'Cell_unique':cell_uniq.values, 'Response_size':['' for i in range(cell_uniq.shape[0])]})
    for i in range(len(cell_names)):
        ind = np.where(cell_id['CELL']==cell_names['Cell_unique'].iloc[i])[0]
        # Check size of response matrix for each cell line
        ind1 = np.where(aid['CELL'] == cell_names['Cell_unique'].iloc[i])[0]
        cell_names['Response_size'].iloc[i] = len(ind1)
    cell_names_sort = cell_names.sort_values(by = ['Response_size'], ascending=False).reset_index(drop=True)
    cells = list(cell_names_sort['Cell_unique'][0:100])

if mode =='drug':
    path = file_path+'/Data/Cell_Line_Drug_Screening_Data/Response_Data/drug_response_data.txt'
    resp = pd.read_table(path)
    ind = [ind for ind,drg in enumerate(list(resp['DRUG'])) if drg.split('.')[0] == study]
    aid = resp.iloc[ind].reset_index(drop=True)
    if R2_filter:
        aid = aid[aid['R2fit']>=0.8].reset_index(drop=True) # Filter based on R2 fit of dose response curve
    drug_id = aid[['DRUG','Drug_UniqueID']]
    drug_uniq = drug_id['Drug_UniqueID'].drop_duplicates()
    drug_names = pd.DataFrame({'Drug_unique':drug_uniq.values, 'Argonne_ID':['' for i in range(drug_uniq.shape[0])], 'Response_size':['' for i in range(drug_uniq.shape[0])],
                                'Hits_size':['' for i in range(drug_uniq.shape[0])], 'Response_std_dev':['' for i in range(drug_uniq.shape[0])], 
                                'Response_mean':['' for i in range(drug_uniq.shape[0])], 'Hits/Response_size':['' for i in range(drug_uniq.shape[0])]})
    for i in range(len(drug_names)):
        ind = np.where(drug_id['Drug_UniqueID']==drug_names['Drug_unique'].iloc[i])[0]
        drug_names['Argonne_ID'].iloc[i] = set(list(drug_id['DRUG'].iloc[ind]))
        # Check size of response matrix for each drug
        ind1 = np.where(aid['Drug_UniqueID'] == drug_names['Drug_unique'].iloc[i])[0]
        drug_names['Response_size'].iloc[i] = len(ind1)
        drug_names['Hits_size'].iloc[i] = sum(aid.iloc[ind]['AUC'].values<0.5)
        drug_names['Response_std_dev'].iloc[i] = np.std(aid.iloc[ind]['AUC'])
        drug_names['Response_mean'].iloc[i] = np.mean(aid.iloc[ind]['AUC'])
        drug_names['Hits/Response_size'].iloc[i] = drug_names['Hits_size'].iloc[i]/drug_names['Response_size'].iloc[i]
        
    drug_names = drug_names[drug_names['Hits_size']>=20]
    drug_names = drug_names[drug_names['Hits/Response_size']<=0.7]
    drug_names = drug_names[drug_names['Response_size']>=500]
    drug_names_sort = drug_names.sort_values(by = ['Response_size'], ascending=False).reset_index(drop=True)
    drugs = list(drug_names_sort['Drug_unique'])
    if cross_test:
        ##### - find drugs with unique cell-lines in CCLE
        ind = [ind for ind,drg in enumerate(list(resp['DRUG'])) if drg.split('.')[0] == study]
        data1 = resp.iloc[ind].reset_index(drop=True)
        ind = [ind for ind,drg in enumerate(list(resp['DRUG'])) if drg.split('.')[0] == study2]
        data2 = resp.iloc[ind].reset_index(drop=True)
        unique_cl = pd.DataFrame(columns=['Drug_unique', 'Response_size_CTRP','Response_size_CCLE', 'Unique_cell-lines'])

        for drug in drugs:
            ind2 = np.where(data2['Drug_UniqueID']==drug)[0]
            if len(ind2)==0:
                    continue
            
            data_sub2 = data2.iloc[ind2].reset_index(drop=True)
            ind1 = np.where(data1['Drug_UniqueID']==drug)[0]
            data_sub1 = data1.iloc[ind1].reset_index(drop=True)
            cl1 = data_sub1['CCLE_CCL_UniqueID'].values
            cl2 = data_sub2['CCLE_CCL_UniqueID'].values
            not_cl=[]
            for cl in cl2:
                    if cl not in cl1:
                        not_cl.append(cl)
            if len(not_cl)!=0:
                    row = pd.DataFrame(columns=unique_cl.columns, index=range(1))
                    row['Drug_unique'] = drug
                    row['Response_size_CTRP']=len(ind1)
                    row['Response_size_CCLE']=len(ind2)
                    row['Unique_cell-lines']=len(not_cl)
                    unique_cl=pd.concat([unique_cl, row])
        unique_cl=unique_cl.reset_index(drop=True)
        drugs = unique_cl['Drug_unique'].values
################################
###### Start experiments #######
################################

################### LIGHTGBM MODEL ##################
random.seed(1)
seeds_all = random.sample(range(0, 100), 20) #Seeds for ensemble models
#random.seed(1)
#random_seeds = random.sample(range(0, 1000), 1000) # seeds for data splitting within iterations

if model == 'lgbm':
    params = {
        'study' : study,
        'mode' : mode,
        'model' : model,
        'sampling' : sampling,
        'ensemble' : ensemble,
        'R2_filter' : R2_filter,
        #'random_seeds' : random_seeds,
        'num_leaves_all' : [5,10,15,20,25,30,35,40,45,50,55,60],
        'seeds_all' : seeds_all,
        'num_add' :20,
        'output_dir': str('New_expt/'+model+'/'+mode+'/'+study+'/'+expt_name),
        'lgb_params' : {'learning_rate': 0.05, 'random_state':5, 'num_boost_round':500, 'stopping_rounds':30},
        #'gene_filter' : 'oncogenes',
        'gene_filter' : 'lincs',
        'train_holdout_split_size' : 0.15,
        'train_test_split_size' : 0.15,   
        'train_val_split_size' : 0.15, 
        'kappa': 1,
        'cross_test':cross_test,
        'study2':study2,
        'hybrid_iter':hybrid_iter,
        'hybrid_sample':hybrid_sample,
        'hybrid_random_itr_ratio':hybrid_random_itr_ratio, # Percentage of iterations coming from random sampling
        'hybrid_random_sample_ratio':hybrid_random_sample_ratio #Percentage of newly added samples coming from random sampling in each iteration
        }    
    if mode =='cell_line':
        for cell in cells:
            random.seed(0)
            data_split_seed = random.sample(range(0, 300), 100)
            ### Hyper-parameters ###
            while len(data_split_seed)!=0:
                params['cell_line'] = cell
                params['use_cell_line'] = False
                params['data_split_seed'] = data_split_seed[-2:]
                params['num_sample'] = cell_names_sort['Response_size'][np.where(cell_names_sort['Cell_unique']==cell)[0]].values[0]
                data_split_seed.pop()
                data_split_seed.pop()
                run_lgbm(params)

    if mode =='drug':
        for drug in drugs: # Loop over the selected drugs
            random.seed(0)
            data_split_seed = random.sample(range(0, 300), 100)
            ### Hyper-parameters ###
            while len(data_split_seed)!=0:
                random.seed(1)
                random_seeds = random.sample(range(0, 1000), 1000) # seeds for data splitting within iterations
                params['random_seeds'] = random_seeds
                params['drug_unique_id'] = drug
                params['use_drug'] = False
                params['data_split_seed'] = data_split_seed[-2:]
                params['num_sample'] = drug_names_sort['Response_size'][np.where(drug_names_sort['Drug_unique']==drug)[0]].values[0]
                data_split_seed.pop()
                data_split_seed.pop()
                run_lgbm(params)



################### SVR MODEL ##################
random.seed(1)
seeds_all = random.sample(range(0, 100), 20)
if model == 'svr':
    params = {
        'study' : study,
        'mode' : mode,
        'model' : model,
        'sampling' : sampling,
        'ensemble' : ensemble,
        'R2_filter' : R2_filter,
        'data_standardize': True,
        'cost_all' : [0.1, 1, 10, 100],
        'seeds_all' : seeds_all,
        'num_add' :20,
        'output_dir': str('New_expt/'+model+'/'+mode+'/'+study+'/'+expt_name),
        #'lgb_params' : {'learning_rate': 0.05, 'random_state':5, 'num_boost_round':500, 'stopping_rounds':30},
        #'gene_filter' : 'oncogenes',
        'gene_filter' : 'lincs',
        'train_holdout_split_size' : 0.15,
        'train_test_split_size' : 0.15,   
        'train_val_split_size' : 0.15, 
        'kappa': 1,
        'hybrid_iter':hybrid_iter,
        'hybrid_sample':hybrid_sample,
        'hybrid_random_itr_ratio':hybrid_random_itr_ratio, # Percentage of iterations coming from random sampling
        'hybrid_random_sample_ratio':hybrid_random_sample_ratio #Percentage of newly added samples coming from random sampling in each iteration
        }    
    if mode =='cell_line':
        for cell in cells:
            random.seed(0)
            data_split_seed = random.sample(range(0, 300), 100)
            ### Hyper-parameters ###
            while len(data_split_seed)!=0:
                params['cell_line'] = cell
                params['use_cell_line'] = False
                params['data_split_seed'] = data_split_seed[-2:]
                params['num_sample'] = cell_names_sort['Response_size'][np.where(cell_names_sort['Cell_unique']==cell)[0]].values[0]
                data_split_seed.pop()
                data_split_seed.pop()
                run_svr(params)

    if mode =='drug':
        for drug in drugs: # Loop over the selected drugs
            random.seed(0)
            data_split_seed = random.sample(range(0, 300), 100)
            ### Hyper-parameters ###
            while len(data_split_seed)!=0:
                random.seed(1)
                random_seeds = random.sample(range(0, 1000), 1000) # seeds for data splitting within iterations
                params['random_seeds'] = random_seeds
                params['drug_unique_id'] = drug
                params['use_drug'] = False
                params['data_split_seed'] = data_split_seed[-2:]
                params['num_sample'] = drug_names_sort['Response_size'][np.where(drug_names_sort['Drug_unique']==drug)[0]].values[0]
                data_split_seed.pop()
                data_split_seed.pop()
                run_svr(params)


################### DNN MODEL ##################
random.seed(1)
seeds_all = random.sample(range(0, 100), 10)
if model =='dnn':
    params = {
        'study' : study,
        'mode' : mode,
        'model' : model,
        'sampling' : sampling,
        'ensemble' : ensemble,
        'num_layers_all' : [4],
        'seeds_all' : seeds_all,
        'num_add' :50,
        'output_dir': str(model+'/'+mode+'/'+study+'/'+expt_name),
        'dnn_params' : {'learning_rate': 0.01, 'optimizer':'SGD', 'loss':'MSELoss', 'net_seed':0,
                        'activation':'tanh', 'final_activation':'sigmoid', 'batch_size': 32, 'epochs':200, 'drop_out':0.4},
        #'gene_filter' : 'oncogenes',
        'gene_filter' : 'lincs',
        'train_holdout_split_size' : 0.15,
        'train_test_split_size' : 0.15,   
        'train_val_split_size' : 0.15, 
        'kappa': 1,
        'n_trials' : 5,
        'hybrid_iter':hybrid_iter,
        'hybrid_sample':hybrid_sample,
        'hybrid_random_itr_ratio':0, # Percentage of iterations coming from random sampling
        'hybrid_random_sample_ratio':0 #Percentage of newly added samples coming from random sampling in each iteration
        }    
    if mode =='cell_line':
        for cell in cells:
            random.seed(0)
            data_split_seed = random.sample(range(0, 300), 100)
            ### Hyper-parameters ###
            while len(data_split_seed)!=0:
                params['cell_line'] = cell
                params['use_cell_line'] = False
                params['data_split_seed'] = data_split_seed[-2:]
                params['num_sample'] = cell_names_sort['Response_size'][np.where(cell_names_sort['Cell_unique']==cell)[0]].values[0]
                data_split_seed.pop()
                data_split_seed.pop()
                run_dnn(params)

    if mode == 'drug':
        for drug in drugs: # Loop over the selected drugs
            random.seed(0)
            data_split_seed = random.sample(range(0, 300), 100)
            ### Hyper-parameters ###
            while len(data_split_seed)!=0:
                params['drug_unique_id'] = drug
                params['use_drug'] = False
                params['data_split_seed'] = data_split_seed[-2:]
                params['num_sample'] = drug_names_sort['Response_size'][np.where(drug_names_sort['Drug_unique']==drug)[0]].values[0]
                data_split_seed.pop()
                data_split_seed.pop()
                run_dnn(params)



    
