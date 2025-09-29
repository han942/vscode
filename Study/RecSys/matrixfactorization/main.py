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
    pred_rating = mf_model.predict(0,0)
    print(f'\n Test : User 0, item 0 rating {pred_rating:.3f}')
    print(f'Observed Rating: {R[0,0]}')


