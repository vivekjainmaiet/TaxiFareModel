# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from TaxiFareModel.utils import *
from TaxiFareModel.utils import DistanceTransformer, TimeFeaturesEncoder
from sklearn.linear_model import LinearRegression


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                          ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
        "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
        'dropoff_longitude']), ('time', time_pipe, ['pickup_datetime'])],
                                     remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                     ('linear_model', LinearRegression())])
        return self.pipe

    def run(self):
        """set and train the pipeline"""
        '''returns a trained pipelined model'''
        self.pipeline.fit(self.X,self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        self.rmse = compute_rmse(y_pred, y_test)
        print(self.rmse)
        return self.rmse


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
