from preprocessing import *
from matfac import MatrixFactorization

if __name__ == '__main__':
    R = R_large
    k = 10
    lr = 0.01
    reg_param = 0.01
    epochs = 50

    mf_model  = MatrixFactorization(R,k,lr,reg_param,epochs)

    print('Start Model Training')
    mf_model.fit()
    print('Model Training Success')

    #Prediction
    R_pred = mf_model.predict()
    


