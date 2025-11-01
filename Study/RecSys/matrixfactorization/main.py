from preprocessing import *
from matfac import MatrixFactorization
import numpy as np
from sklearn.metrics import root_mean_squared_error

if __name__ == '__main__':
    R_train = R_train.fillna(0)
    k = 10
    lr = 0.0001
    reg_param = 0.01
    epochs = 30

    mf_model  = MatrixFactorization(k,lr,reg_param,epochs)

    print('Start Model Training')
    mf_model.fit(R_train)
    print('\nModel Training Success')

    #Prediction
    R_pred = mf_model.predict(test)

    #Evaluation
    