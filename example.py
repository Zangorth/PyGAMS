###########
# Imports #
###########
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from pygams.defaults import FeaturePipeline as FP
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
from pygams.pygams import PyGAMS
from pygams.space import Space
import pandas as pd
import numpy as np

#################
# Generate Data #
#################
x, y = make_classification(n_samples=1000, n_features=100, n_informative=10)

panda = pd.DataFrame(y, columns=['y'])
panda = panda.merge(pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])]),
                    how='outer', left_index=True, right_index=True)

x = panda.drop('y', axis=1).copy()
y = panda['y']

##########################
# Establish Search Space #
##########################
pipes = Space(FP, name='FeaturePipeline')
pipes.Categories(parameter='features', 
                 choices=x.columns, low=4, high=15)
pipes.Category(parameter='scaler', choices=[MinMaxScaler(), MaxAbsScaler(), StandardScaler()])
pipes.Category(parameter='imputer', choices=[SimpleImputer(), SimpleImputer(strategy='median')])

rf = Space(RandomForestClassifier, name='RandomForest')
rf.Integer('n_estimators', low=10, high=1000, distribution='exponential-decay')
rf.Category('max_features', choices=['sqrt', 'log2'])
rf.Real('min_impurity_decrease', low=0.0001, high=1, distribution='log-uniform')

et = Space(ExtraTreesClassifier, name='ExtraTrees')
et.Integer('n_estimators', low=10, high=1000, distribution='exponential-decay')
et.Category('max_features', choices=['sqrt', 'log2'])
et.Real('min_impurity_decrease', low=0.0001, high=1, distribution='log-uniform')

##################
# Optimize Model #
##################
gams = PyGAMS(models=[rf, et], pipes=pipes, generations=40, survivors=5, population_size=20)
model_selection = gams.run(x, y, n_jobs=20, verbose=True)

################
# View Results #
################
# model_selection output
for i, model in enumerate(model_selection):
    print(f'Model {i+1} Score: {np.mean(model["fitness"])}')
    print(f'Generations Survived: {len(model["fitness"])}')
    print(f'Model Type: {model["model_species"]}')
    print(f'Model Params: {model["model_params"]}')
    print(f'Pipeline Type: {model["pipe_species"]}')
    print(f'Pipeline Params: {model["pipe_params"]}')
    print('')

# Display Scores over time
gams.plot_scores()
plt.close()

gams.plot_scores(kind='mean')
plt.close()

# Display parameter distribution
for parameter in gams.distribution_options():
    print(parameter)
    if 'features' not in parameter:
        gams.plot_parameter(parameter)


gams.plot_parameter(param=f'{model_selection[0]["model_species"]} x {model_selection[0]["pipe_species"]} | features',
                    categories_to_examine=model_selection[0]['pipe_params']['features'][0:5])

