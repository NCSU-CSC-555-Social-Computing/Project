# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:39:49 2022

@author: kaiva
"""
import pandas as pd

base_path = "../datasets/Transactions/"
file = 'transaction_data.csv'
dataset = pd.read_csv(base_path + file)

new_dataset = pd.DataFrame(dataset[['Source','Weight','typeTrans']])

res = []
for row in range(len(new_dataset)):
    typ = new_dataset['typeTrans'][row].replace("'","")
    typ = typ.replace("es_","")
    res.append([new_dataset['Source'][row].replace("'",""), int(new_dataset['Weight'][row]),typ])
    
res = pd.DataFrame(res)
res.columns = ['Source','Weight','Transaction_Type']
res_group = res.groupby(['Source','Transaction_Type'], as_index=False)["Weight"].sum()


categories = res_group.Transaction_Type.unique()
res = []
for cat in categories:
    sub_group = pd.DataFrame(res_group[res_group['Transaction_Type']==cat])
    sub_group.columns = ['Source','Transaction_Type', 'Weight']
    mi = min(sub_group['Weight'])
    ma = max(sub_group['Weight'])
    for i,j in sub_group.iterrows():
        res.append([j['Source'], j['Transaction_Type'], (int(j['Weight'])-mi)/(ma-mi)])

res = pd.DataFrame(res)
res.columns = ['Source','Transaction_Type','Weight']
#res_group = res.groupby(['Source','Transaction_Type'], as_index=False)["Weight"].sum()
res.to_csv('../datasets/Transactions/transaction_data_normalized.csv', index=False)

#print(res_group[res_group['Source']=="C1000148617" and res_group['Transaction_Type']=="food"])