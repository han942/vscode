# movielens 100K

#SVD 알아오기

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

path = Path('../../../datafile/Recsys/ml-100k')
raw_data = pd.read_csv(path / 'u.data',sep='\t',
                   usecols=[0,1,2],names=['user','item','rating'],
                   dtype={'user':int,'item':int,'rating':float})
train,test = train_test_split(raw_data,test_size=0.2,random_state=42)

R_train = pd.pivot_table(train,values='rating',index='user',columns='item')
