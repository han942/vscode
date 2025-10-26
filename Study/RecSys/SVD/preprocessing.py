# movielens 100K

#SVD 알아오기

import pandas as pd

path = 'C:/Users/hanan/.vscode/DS/vscode/datafile/Recsys/ml-100k'
raw_data = pd.read_csv(path+'/u.data',sep='\t',
                   usecols=[0,1,2],names=['user','item','rating'],
                   dtype={'user':int,'item':int,'rating':float})
R_large = pd.pivot_table(data=raw_data,columns='user',index='item',values='rating')