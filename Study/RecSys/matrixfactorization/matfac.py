import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

class  MatrixFactorization():
    def __init__(self,k,lr,reg_param,epochs):
        self.k = k
        self.lr = lr
        self.reg_param = reg_param
        self.epochs = epochs
        pass

    def fit(self,R):
        self.R = R
        self.n_users,self.n_items = self.R.shape

        self.rows,self.cols = np.nonzero(self.R)
        self.obs_ind = list(zip(self.rows,self.cols))
        
        #P,Q random initialization
        self.P = np.random.normal(size=(self.n_users,self.k))
        self.Q = np.random.normal(size=(self.n_items,self.k))

        for n in range(self.epochs):
            if n % 5 == 0:
                R_pred = np.dot(self.P,self.Q.T)
                rsme = root_mean_squared_error(self.R,R_pred) #학습오차
                print(f'Epoch : {n} , RMSE : {rsme:.4f}')
            for u,i in self.obs_ind:
                    if self.R[u,i] == 0:
                        pass
                    else:
                        e = self.R[u,i] - np.dot(self.P[u,:],self.Q[i,:]) #loss-function
                        #user-update
                        self.P[u,:] = self.P[u,:] + self.lr * (e* self.Q[i,:] - self.reg_param*self.P[u,:])
                        #item-update
                        self.Q[i,:] = self.Q[i,:] + self.lr * (e* self.P[u,:] - self.reg_param*self.Q[i,:])
        pass

    def predict(self,test,exclude_unknowns=True):
        self.R_pred = np.dot(self.P, self.Q.T)
        if exclude_unknowns == True:
            
            ulist_test = list(zip(test['user'],test['item']))
        