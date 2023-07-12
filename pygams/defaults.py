from sklearn.metrics import roc_auc_score
import pandas as pd

#############
# Pass Pipe #
#############
class PassPipe():
    def __init__(self):
        '''
        Description - default pipeline that returns the same dataframe it was given

        Returns
        -------
        None.

        '''
        return None
    
    def fit(self):
        return None
    
    def transform(x: pd.DataFrame):
        return x

####################
# Feature Pipeline #
####################
class FeaturePipeline():
    def __init__(self, features, scaler, imputer):
        '''
        Description - A basic pipeline that will subset the feature space, scale the features, and impute missing values

        Parameters
        ----------
        features : list
            A list of features to be included in the final model
        scaler : callable
            A class with a fit and transform a method which re-scales the data to a normalized range
        imputer : callable
            A class with a fit and transform method which imputes missing values in the data
        '''
        self.features = features
        self.scaler = scaler
        self.imputer = imputer
        
        return None
    
    def fit(self, x):
        df = x[self.features]

        self.scaler.fit(df)
        self.imputer.fit(df)

        return None
    
    def transform(self, x):
        df = x[self.features]

        output = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
        output = pd.DataFrame(self.scaler.transform(output), columns=df.columns)

        return output
    
#########
# AUROC #
#########
def auroc(y_test, y_pred):
    return roc_auc_score(y_test, y_pred[:, 1])