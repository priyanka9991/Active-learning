from Model_cell_line import run
import random 
import os
import pandas as pd
import numpy as np

study = 'CTRP'
mode = 'cell_line'   # mode = 'drug' or 'cell_line' or 'both'
model = 'lgbm'
file_path = os.path.dirname(os.path.realpath(__file__))

# TODO Input study, model typr(lgbm or dnn), cell_specific, drug_specific or both as command line arguments
# Add DNN model pipeline
# Transfer learning between different studies
# Use all te studies


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

    random.seed(1)
    seeds_all = random.sample(range(0, 100), 10)
    for cell in cells:
        random.seed(0)
        data_split_seed = random.sample(range(0, 300), 100)
        ### Hyper-parameters ###
        while len(data_split_seed)!=0:
            params = {
                'cell_line' : cell,
                'study' : study,
                'mode' : mode,
                'model' : model,
                'use_cell_line' : False,
                'output_dir' : str('Cell_line/Output_1_kappa'+ study),
                'num_leaves_all' : [5,10,15,20,25,30,35,40,45,50,55,60],
                'seeds_all' : seeds_all,
                'num_add' :20,
                'data_split_seed' : data_split_seed[-2:],
                'lgb_params' : {'learning_rate': 0.05, 'random_state':5, 'num_boost_round':500, 'stopping_rounds':30},
                #'gene_filter' : 'oncogenes',
                'gene_filter' : 'lincs',
                'train_holdout_split_size' : 0.15,
                'train_test_split_size' : 0.15,   
                'train_val_split_size' : 0.15, 
                'num_sample': cell_names_sort['Response_size'][np.where(cell_names_sort['Cell_unique']==cell)[0]].values[0],
                'kappa': 1
                }    
            data_split_seed.pop()
            data_split_seed.pop()
            run(params)

if mode =='drug':
    path = file_path+'/Data/Cell_Line_Drug_Screening_Data/Response_Data/drug_response_data.txt'
    resp = pd.read_table(path)
    ind = [ind for ind,drg in enumerate(list(resp['DRUG'])) if drg.split('.')[0] == study]
    aid = resp.iloc[ind].reset_index(drop=True)
    drug_id = aid[['DRUG','Drug_UniqueID']]
    drug_uniq = drug_id['Drug_UniqueID'].drop_duplicates()
    drug_names = pd.DataFrame({'Drug_unique':drug_uniq.values, 'Argonne_ID':['' for i in range(drug_uniq.shape[0])], 'Response_size':['' for i in range(drug_uniq.shape[0])]})
    for i in range(len(drug_names)):
        ind = np.where(drug_id['Drug_UniqueID']==drug_names['Drug_unique'].iloc[i])[0]
        drug_names['Argonne_ID'].iloc[i] = set(list(drug_id['DRUG'].iloc[ind]))
        # Check size of response matrix for each drug
        ind1 = np.where(aid['Drug_UniqueID'] == drug_names['Drug_unique'].iloc[i])[0]
        drug_names['Response_size'].iloc[i] = len(ind1)
    drug_names_sort = drug_names.sort_values(by = ['Response_size'], ascending=False).reset_index(drop=True)
    drugs = list(drug_names_sort['Drug_unique'][0:100])

    random.seed(1)
    seeds_all = random.sample(range(0, 100), 10)

    for drug in drugs: # Loop over the selected drugs
        random.seed(0)
        data_split_seed = random.sample(range(0, 300), 100)
        ### Hyper-parameters ###
        while len(data_split_seed)!=0:
            params = {
                'drug_unique_id' : drug,
                'study' : study,
                'mode' : mode,
                'model' : model,
                'use_drug' : False,
                'output_dir' : str('Drug/Output_1_kappa'+ study),
                'num_leaves_all' : [5,10,15,20,25,30,35,40,45,50,55,60],
                'seeds_all' : seeds_all,
                'num_add' :20,
                'data_split_seed' : data_split_seed[-2:],
                'lgb_params' : {'learning_rate': 0.05, 'random_state':5, 'num_boost_round':500, 'stopping_rounds':30},
                #'gene_filter' : 'oncogenes',
                'gene_filter' : 'lincs',
                'train_holdout_split_size' : 0.15,
                'train_test_split_size' : 0.15,   
                'train_val_split_size' : 0.15, 
                'num_sample': drug_names_sort['Response_size'][np.where(drug_names_sort['Drug_unique']==drug)[0]].values[0],
                'kappa': 1
                }    
            data_split_seed.pop()
            data_split_seed.pop()
            run(params)
