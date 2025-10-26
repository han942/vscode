from preprocessing import *
from SVD import SVD
import numpy as np
from sklearn.metrics import root_mean_squared_error

if __name__ == '__main__':
    R_large = R_large.fillna(0)
    R = np.array(R_large)
    k = 10
    method = 0
    svd_model = SVD(R,k,method)

    print('----Start svd model training-----')
    svd_model.fit()

    print('---Start Prediction---')
    R_pred = svd_model.predict()

    print(f'Method : {method} , RMSE: ',root_mean_squared_error(R_large,R_pred))
