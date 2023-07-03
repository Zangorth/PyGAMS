###########
# Imports #
###########
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from pygams.example_helpers import FeaturePipeline as FP
from sklearn.datasets import make_classification
import pandas as pd

# Custom Import
from pygams.pygams import PyGAMS
from pygams.space import Space

#################
# Generate Data #
#################
x, y = make_classification(n_samples=1000, n_features=100, n_informative=10)

panda = pd.DataFrame(y, columns=['y'])
panda = panda.merge(pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])]),
                    how='outer', left_index=True, right_index=True)

x = panda.drop('y', axis=1).copy()
y = panda['y']

pipes = Space(FP, name='FeaturePipeline')
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

gams = PyGAMS(models=[rf, et], pipes=pipes, generations=10, survivors=5, population_size=40)
model_selection = gams.run(x, y, n_jobs=10, verbose=True)

for i, model in enumerate(model_selection):
    print(f'Model {i+1} Score: {model["fitness"]}')
    print(f'Model Type: {model["model_species"]}')
    print(f'Model Params: {model["model_params"]}')
    print(f'Pipeline Type: {model["pipe_species"]}')
    print(f'Pipeline Params: {model["pipe_params"]}')
    print('')
