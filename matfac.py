import numpy as np

class  MatrixFactorization():
    def __init__(self,R,k,lr,reg_param,epochs):
        self.R = R
        self.k = k
        self.lr = lr
        self.reg_param = reg_param
        self.epochs = epochs

        self.n_users,self.n_items = R.shape

        #P,Q random initialization
        self.P = np.random.rand(self.n_users,self.k)
        self.Q = np.random.rand(self.n_items,self.k)
        pass

    def fit(self):  #1차 - observed data / numpy indexing / epochs
        for n in range(self.epochs):
            for i in range(self.n_items):
                for u in range(self.n_users):
                    if self.R[u,i] == 0:
                        pass
                    else:
                        e = self.R[u,i] - np.dot(self.P[u,:],self.Q[i,:]) # [1*m] * [n*1] = scalar
                        #user-update
                        self.P[u,:] = self.P[u,:] + self.lr * (e* self.Q[i,:] - self.reg_param*self.P[u,:])
                        #item-update
                        self.Q[i,:] = self.Q[i,:] + self.lr * (e* self.P[u,:] - self.reg_param*self.Q[i,:])
        pass

    def predict(self,user_id,item_id):
        self.R_pred[user_id,item_id] = np.dot(self.P[user_id,:],self.Q[item_id,:])
        return self.R_pred