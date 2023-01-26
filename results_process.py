import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats
import pandas as pd


file_path = os.path.dirname(os.path.realpath(__file__))

#drug_unique_id = 'Drug_9'
study = 'GDSC'
dir = 'no_drug'
output = 'Output_100_drugs_kappa1_5CTRP'
savename = ['output_resampl.json', 'output_hp.json', 'output_hp_resampl.json']
path_name = os.path.join(file_path, 'Output',output)
result = pd.DataFrame(columns=['Drug','RMSE_resamp', 'RMSE_resamp_trend','R2_resamp','R2_resamp_trend', 'RMSE_HP','RMSE_HP_trend', 'R2_HP','R2_HP_trend', 'RMSE_HP_resamp','RMSE_HP_resamp_trend', 'R2_HP_resamp','R2_HP_resamp_trend'])

for path in os.listdir(path_name):
    new_path = os.path.join(path_name, path, dir)
    drug = path
    rmse_hp = []
    rmse_random_hp = []
    r2_hp = []
    r2_random_hp = []
    rmse_resmp = []
    rmse_random_resmp = []
    r2_resmp = []
    r2_random_resmp = []

    rmse_hp_resmp = []
    rmse_random_hp_resmp = []
    r2_hp_resmp = []
    r2_random_hp_resmp = []
    cnt = 1
    for subdir, dirs, files in os.walk(new_path):
        if cnt ==1:
            cnt+=1
            continue
        cnt+=1
        for file in files:
            output = json.load(open(os.path.join(subdir,file)))
            if file == 'output_hp.json':
                rmse_hp.append(output['rmse_hold_out'])
                r2_hp.append(output['r2_hold_out'])
                rmse_random_hp.append(output['rmse_hold_out_random'])
                r2_random_hp.append(output['r2_hold_out_random'])
            if file == 'output_resampl.json':
                rmse_resmp.append(output['rmse_hold_out'])
                r2_resmp.append(output['r2_hold_out'])
                rmse_random_resmp.append(output['rmse_hold_out_random'])
                r2_random_resmp.append(output['r2_hold_out_random'])
            if file == 'output_hp_resampl.json':
                rmse_hp_resmp.append(output['rmse_hold_out'])
                r2_hp_resmp.append(output['r2_hold_out'])
                rmse_random_hp_resmp.append(output['rmse_hold_out_random'])
                r2_random_hp_resmp.append(output['r2_hold_out_random'])


    # AUC calculation
    auc_rmse_hp = []
    auc_rmse_hp_random = []
    auc_r2_hp = []
    auc_r2_hp_random = []
    for i in range(len(rmse_hp)):
        x = np.arange(0,len(rmse_hp[i]))
        y = rmse_hp[i]
        auc_rmse_hp.append(metrics.auc(x, y))
        y = rmse_random_hp[i]
        auc_rmse_hp_random.append(metrics.auc(x, y))
        y = r2_hp[i]
        auc_r2_hp.append(metrics.auc(x, y))
        y = r2_random_hp[i]
        auc_r2_hp_random.append(metrics.auc(x, y))

    auc_rmse_resmp = []
    auc_rmse_resmp_random = []
    auc_r2_resmp = []
    auc_r2_resmp_random = []
    for i in range(len(rmse_resmp)):
        x = np.arange(0,len(rmse_resmp[i]))
        y = rmse_resmp[i]
        auc_rmse_resmp.append(metrics.auc(x, y))
        y = rmse_random_resmp[i]
        auc_rmse_resmp_random.append(metrics.auc(x, y))
        y = r2_resmp[i]
        auc_r2_resmp.append(metrics.auc(x, y))
        y = r2_random_resmp[i]
        auc_r2_resmp_random.append(metrics.auc(x, y))

    auc_rmse_hp_resmp = []
    auc_rmse_hp_resmp_random = []
    auc_r2_hp_resmp = []
    auc_r2_hp_resmp_random = []
    for i in range(len(rmse_hp_resmp)):
        x = np.arange(0,len(rmse_hp_resmp[i]))
        y = rmse_hp_resmp[i]
        auc_rmse_hp_resmp.append(metrics.auc(x, y))
        y = rmse_random_hp_resmp[i]
        auc_rmse_hp_resmp_random.append(metrics.auc(x, y))
        y = r2_hp_resmp[i]
        auc_r2_hp_resmp.append(metrics.auc(x, y))
        y = r2_random_hp_resmp[i]
        auc_r2_hp_resmp_random.append(metrics.auc(x, y))

    # Pairwise t-test
    row = pd.DataFrame(columns= result.columns, index=range(1))
    row['Drug'] =  drug
    row['RMSE_HP'] = stats.ttest_rel(auc_rmse_hp, auc_rmse_hp_random).pvalue
    row['RMSE_HP_trend'] = np.median(auc_rmse_hp)< np.median(auc_rmse_hp_random)
    row['R2_HP']=stats.ttest_rel(auc_r2_hp, auc_r2_hp_random).pvalue
    row['R2_HP_trend'] = np.median(auc_r2_hp) > np.median(auc_r2_hp_random)
    row['RMSE_resamp'] = stats.ttest_rel(auc_rmse_resmp, auc_rmse_resmp_random).pvalue
    row['RMSE_resamp_trend'] = np.median(auc_rmse_resmp)< np.median(auc_rmse_resmp_random)
    row['R2_resamp'] = stats.ttest_rel(auc_r2_resmp, auc_r2_resmp_random).pvalue
    row['R2_resamp_trend'] = np.median(auc_r2_resmp)>np.median(auc_r2_resmp_random)
    row['RMSE_HP_resamp'] = stats.ttest_rel(auc_rmse_hp_resmp, auc_rmse_hp_resmp_random).pvalue
    row['RMSE_HP_resamp_trend'] = np.median(auc_rmse_hp_resmp)<np.median(auc_rmse_hp_resmp_random)
    row['R2_HP_resamp'] = stats.ttest_rel(auc_r2_hp_resmp, auc_r2_hp_resmp_random).pvalue
    row['R2_HP_resamp_trend'] = np.median(auc_r2_hp_resmp) > np.median(auc_r2_hp_resmp_random)
    result = pd.concat([result, row])

result.to_csv(os.path.join(file_path,'Output','GDSC_Drug100.csv'))

cnt =1
for subdir, dirs, files in os.walk(path_name):
    if cnt ==1:
        cnt+=1
        continue
    cnt+=1
    print(subdir)
    for file in dirs:


path = os.path.join(file_path, 'Output',output, str(drug_unique_id+'_'+study),dir)
rmse_hp = []
rmse_random_hp = []
r2_hp = []
r2_random_hp = []
rmse_resmp = []
rmse_random_resmp = []
r2_resmp = []
r2_random_resmp = []

rmse_hp_resmp = []
rmse_random_hp_resmp = []
r2_hp_resmp = []
r2_random_hp_resmp = []
cnt =1
for subdir, dirs, files in os.walk(path):
    if cnt ==1:
        cnt+=1
        continue
    cnt+=1
    for file in files:
        output = json.load(open(os.path.join(subdir,file)))
        if file == 'output_hp.json':
            rmse_hp.append(output['rmse_hold_out'])
            r2_hp.append(output['r2_hold_out'])
            rmse_random_hp.append(output['rmse_hold_out_random'])
            r2_random_hp.append(output['r2_hold_out_random'])
        if file == 'output_resampl.json':
            rmse_resmp.append(output['rmse_hold_out'])
            r2_resmp.append(output['r2_hold_out'])
            rmse_random_resmp.append(output['rmse_hold_out_random'])
            r2_random_resmp.append(output['r2_hold_out_random'])
        if file == 'output_hp_resampl.json':
            rmse_hp_resmp.append(output['rmse_hold_out'])
            r2_hp_resmp.append(output['r2_hold_out'])
            rmse_random_hp_resmp.append(output['rmse_hold_out_random'])
            r2_random_hp_resmp.append(output['r2_hold_out_random'])


# AUC calculation
auc_rmse_hp = []
auc_rmse_hp_random = []
auc_r2_hp = []
auc_r2_hp_random = []
for i in range(len(rmse_hp)):
    x = np.arange(0,len(rmse_hp[i]))
    y = rmse_hp[i]
    auc_rmse_hp.append(metrics.auc(x, y))
    y = rmse_random_hp[i]
    auc_rmse_hp_random.append(metrics.auc(x, y))
    y = r2_hp[i]
    auc_r2_hp.append(metrics.auc(x, y))
    y = r2_random_hp[i]
    auc_r2_hp_random.append(metrics.auc(x, y))

auc_rmse_resmp = []
auc_rmse_resmp_random = []
auc_r2_resmp = []
auc_r2_resmp_random = []
for i in range(len(rmse_resmp)):
    x = np.arange(0,len(rmse_resmp[i]))
    y = rmse_resmp[i]
    auc_rmse_resmp.append(metrics.auc(x, y))
    y = rmse_random_resmp[i]
    auc_rmse_resmp_random.append(metrics.auc(x, y))
    y = r2_resmp[i]
    auc_r2_resmp.append(metrics.auc(x, y))
    y = r2_random_resmp[i]
    auc_r2_resmp_random.append(metrics.auc(x, y))

auc_rmse_hp_resmp = []
auc_rmse_hp_resmp_random = []
auc_r2_hp_resmp = []
auc_r2_hp_resmp_random = []
for i in range(len(rmse_hp_resmp)):
    x = np.arange(0,len(rmse_hp_resmp[i]))
    y = rmse_hp_resmp[i]
    auc_rmse_hp_resmp.append(metrics.auc(x, y))
    y = rmse_random_hp_resmp[i]
    auc_rmse_hp_resmp_random.append(metrics.auc(x, y))
    y = r2_hp_resmp[i]
    auc_r2_hp_resmp.append(metrics.auc(x, y))
    y = r2_random_hp_resmp[i]
    auc_r2_hp_resmp_random.append(metrics.auc(x, y))

# Pairwise t-test
stats.ttest_rel(auc_rmse_hp, auc_rmse_hp_random)
stats.ttest_rel(auc_r2_hp, auc_r2_hp_random)

stats.ttest_rel(auc_rmse_resmp, auc_rmse_resmp_random)
stats.ttest_rel(auc_r2_resmp, auc_r2_resmp_random)

stats.ttest_rel(auc_rmse_hp_resmp, auc_rmse_hp_resmp_random)
stats.ttest_rel(auc_r2_hp_resmp, auc_r2_hp_resmp_random)

#Box plots
plt.figure()
data = [auc_rmse_hp, auc_rmse_hp_random]
fig, ax = plt.subplots()
ax.boxplot(data)
plt.xticks([1, 2], ['Active learning', 'Random sampling'])
plt.ylabel('AUC of RMSE plots - HP only')
plt.show()

plt.figure()
data = [auc_r2_hp, auc_r2_hp_random]
fig, ax = plt.subplots()
ax.boxplot(data)
plt.xticks([1, 2], ['Active learning', 'Random sampling'])
plt.ylabel('AUC of R2 plots - HP only')
plt.show()


plt.figure()
data = [auc_rmse_resmp, auc_rmse_resmp_random]
fig, ax = plt.subplots()
ax.boxplot(data)
plt.xticks([1, 2], ['Active learning', 'Random sampling'])
plt.ylabel('AUC of RMSE plots - Resampling only')
plt.show()

plt.figure()
data = [auc_r2_resmp, auc_r2_resmp_random]
fig, ax = plt.subplots()
ax.boxplot(data)
plt.xticks([1, 2], ['Active learning', 'Random sampling'])
plt.ylabel('AUC of R2 plots - Resampling only')
plt.show()

plt.figure()
data = [auc_rmse_hp_resmp, auc_rmse_hp_resmp_random]
fig, ax = plt.subplots()
ax.boxplot(data)
plt.xticks([1, 2], ['Active learning', 'Random sampling'])
plt.ylabel('AUC of RMSE plots - Resampling and HP')
plt.show()

plt.figure()
data = [auc_r2_hp_resmp, auc_r2_hp_resmp_random]
fig, ax = plt.subplots()
ax.boxplot(data)
plt.xticks([1, 2], ['Active learning', 'Random sampling'])
plt.ylabel('AUC of R2 plots - Resampling and HP')
plt.show()

