import copy
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import datasets
from utils import XyScaler
import os
import matplotlib.pyplot as plt
import random


 

class Modeling(object):

    def __init__(self, df, y_index, x_index, model, kfolds, cleaner=None):
        self.df = df
        self.y_index = y_index
        self.x_index = x_index
        self.model = model
        self.kfolds = kfolds
        self.cleaner = cleaner
    
    def df_to_x_y(self):
        self.X = df.iloc[:, self.x_index:].to_numpy()
        self.y = df.iloc[:, self.y_index].to_numpy()

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state = 4)


    def standardize(self):
        standardizer = XyScaler()
        standardizer.fit(self.X_train, self.y_train)
        self.X_train_standardized, self.y_train_standardized = standardizer.transform(self.X_train, self.y_train)
        self.X_test_standardized, self.y_test_standardized = standardizer.transform(self.X_test, self.y_test)

    def fit(self):
        self.model.fit(self.X_train_standardized, self.y_train_standardized)
         
    def coeffs(self):
        return self.model.coef_


    def predict(self):
        y_pred = self.model.predict(self.X_test_standardized)
        error = self.rmse(self.y_test_standardized, y_pred)
        return error
    
    def rmse(self, y_true, y_pred):
        mse = ((y_true - y_pred)**2).mean()
        return np.sqrt(mse)
    
    def cross_val_score(self):
        kf = KFold(n_splits=self.kfolds)
        error = np.empty(self.kfolds)
        index = 0
        kf_model = copy.deepcopy(self.model)
        for train, test in kf.split(self.X_train_standardized):
            # Clean features
            # X_train = cleaner.clean(self.X[train])
            # X_test = cleaner.clean(self.X[test])
            x_train, x_test = self.X_train_standardized[train], self.X_train_standardized[test]
            y_train, y_test = self.y_train_standardized[train], self.y_train_standardized[test]
            kf_model.fit(x_train, y_train)
            y_pred = kf_model.predict(x_test)
            error[index] = self.rmse(y_test, y_pred)
            index += 1
        
        return np.mean(error)
        


if __name__ == '__main__':
    DATA_DIRECTORY = os.path.join(os.path.split(os.getcwd())[0], 'data')
    df = pd.read_csv(os.path.join(DATA_DIRECTORY, 'reg_games_stats_2018.csv'))
    df.h_top = df.h_top.apply(lambda x: sum([a*b for a,b in zip([60, 1], map(int,x.split(':')))])/60)
    df.a_top = df.a_top.apply(lambda x: sum([a*b for a,b in zip([60, 1], map(int,x.split(':')))])/60)









\
    # n_alphas = 200
    # alphas = np.logspace(-5, 2, n_alphas)
    

    # # ax = plt.gca()
    # # ax.plot(alphas, coefs)
    # # ax.set_xscale('log')
    # # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    # # plt.xlabel('alpha')
    # # plt.ylabel('weights')
    # # plt.title('Ridge coefficients as a function of the regularization')
    # # plt.axis('tight')
    # # plt.show()




    # errors = []
    # errors1 = []
    # errors2 = []
    # errors3 = []
    # for alpha in alphas:
    #     modeler = Modeling(df, 6, 8, Ridge(alpha = alpha), 10)
    #     # modeler1 = Modeling(df, 6, 8, Lasso(alpha = alpha), 10)
    #     # modeler2 = Modeling(df, 6, 8, ElasticNet(alpha = alpha), 10)
    #     # modeler3 = Modeling(df, 6, 8, LinearRegression(), 10)
    #     modeler.df_to_x_y()
    #     modeler.split_data()
    #     modeler.standardize()
    #     modeler.fit()
    #     # modeler1.df_to_x_y()
    #     # modeler1.split_data()
    #     # modeler1.standardize()
    #     # modeler1.fit()
    #     # modeler2.df_to_x_y()
    #     # modeler2.split_data()
    #     # modeler2.standardize()
    #     # modeler2.fit()
    #     # modeler3.df_to_x_y()
    #     # modeler3.split_data()
    #     # modeler3.standardize()
    #     # modeler3.fit()
    #     # modeler.fit()
    #     # error = modeler.predict()
    #     error = modeler.cross_val_score()
    #     errors.append(error)
    #     # error1 = modeler1.cross_val_score()
    #     # errors1.append(error1)
    #     # error2 = modeler2.cross_val_score()
    #     # errors2.append(error2)
    #     # error3 = modeler3.cross_val_score()
    #     # errors3.append(error3)

    
    # errors_idx = np.argmin(errors)
    # # errors1_idx = np.argmin(errors1)
    # # errors2_idx = np.argmin(errors2)
    # # errors3_idx = np.argmin(errors3)


    # # fig, ax = plt.subplots(figsize = (20, 10))
    # # ax.plot(alphas, errors, label = 'Ridge')
    # # ax.plot(alphas, errors1, label = 'Lasso')
    # # ax.plot(alphas, errors2, label = 'Elastic Net')
    # # ax.plot(alphas, errors3, label = 'Linear Regression')
    # # ax.axvline(alphas[errors_idx])
    # # ax.axvline(alphas[errors1_idx])
    # # ax.axvline(alphas[errors2_idx])
    # # ax.axvline(alphas[errors3_idx])
    # # ax.set_xscale('log')
    # # # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    # # ax.set_title('CV Model Performance Across Alpha Values', size = 20)
    # # ax.set_xlabel('Alpha Value')
    # # ax.set_ylabel('RMSE')
    # # fig.legend()
    # # fig.savefig('model_performance_across_alphas.png')
    # # plt.show()


    # coefs = []
    # for alpha in alphas:
    #     modeler = Modeling(df, 6, 8, ElasticNet(alpha = alpha), 10)
    #     modeler.df_to_x_y()
    #     modeler.split_data()
    #     modeler.standardize()
    #     modeler.fit()
    #     coeff = modeler.coeffs()
    #     print(len(coeff))
    #     coefs.append(coeff)


    # coefs_df = pd.DataFrame(coefs) 
    # coefs_df = coefs_df.rename(columns={i: df.iloc[:, 8:].columns[i] for i in range(30) })
    # fig, ax = plt.subplots(figsize = (20, 10))
    # for i in range(len(coefs_df.columns)):
    #     ax.plot(alphas, coefs_df.iloc[:, i], label = coefs_df.iloc[:, i].name)
    # ax.axvline(alphas[errors_idx])
    # ax.set_xscale('log')
    # # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    # plt.xlabel('alpha')
    # plt.ylabel('weights')
    # plt.title('Ridge coefficients as a function of the regularization', fontsize = 20)
    # plt.axis('tight')
    # plt.legend()
    # fig.savefig('ridge_coefs.png')
    # plt.show()


    # # feature_names = data1['feature_names']
    # # raw_data_x = data[0][:100]
    # # raw_data_y = data[1][:100]