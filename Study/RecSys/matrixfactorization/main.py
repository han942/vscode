from preprocessing import *
from matfac import MatrixFactorization
import numpy as np
from sklearn.metrics import root_mean_squared_error

if __name__ == '__main__':
    R_large = R_large.fillna(0)
    R = np.array(R_large)
    k = 10
    lr = 0.0001
    reg_param = 0.01
    epochs = 50

    mf_model  = MatrixFactorization(R,k,lr,reg_param,epochs)

    print('Start Model Training')
    mf_model.fit()
    print('\nModel Training Success')

    #Prediction
    R_pred = mf_model.predict()