###########
# Imports #
###########
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import make_classification
import pandas as pd

# Custom Import
from pygams.pygams import PyGAMS
from pygams.space import Space

#################
# Generate Data #
#################
x, y = make_classification(n_samples=10000, n_features=100, n_informative=10)

panda = pd.DataFrame(y, columns=['y'])
panda = panda.merge(pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])]),
                    how='outer', left_index=True, right_index=True)

x = panda.drop('y', axis=1).copy()
y = panda['y']

############
# Pipeline #
############
class FeaturePipeline():
    def __init__(self, features, scaler, imputer):
        self.features = features
        
        return None
    
    def fit(self, df):
        return None
    
    def transform(self, df):
        return df[self.features]

pipes = Space(FeaturePipeline)
pipes.Categories(parameter='features', 
                 choices=x.columns, low=4, high=15)
pipes.Category(parameter='scaler', choices=['MinMaxScaler', 'MaxAbsScaler', 'StandardScaler'])
pipes.Category(parameter='imputer', choices=['Mean', 'Median'])

rf = Space(RandomForestClassifier, name='RandomForest')
rf.Integer('n_estimators', low=10, high=1000, distribution='exponential-decay')
rf.Category('max_features', choices=['sqrt', 'log2'])
rf.Real('min_impurity_decrease', low=0.0001, high=1, distribution='log-uniform')

et = Space(ExtraTreesClassifier, name='ExtraTrees')
et.Integer('n_estimators', low=10, high=1000, distribution='exponential-decay')
et.Category('max_features', choices=['sqrt', 'log2'])
et.Real('min_impurity_decrease', low=0.0001, high=1, distribution='log-uniform')

gams = PyGAMS(models=[rf, et], pipes=pipes, generations=2)
model_selection = gams.run(x, y)


