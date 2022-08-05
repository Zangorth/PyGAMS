###########
# Imports #
###########
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

##################
# Assess Fitness #
##################
def assess_fitness(x, y, pipe, pipe_params, model, model_params,
                   metric=roc_auc_score, cv=ShuffleSplit, proba=True):
    '''
    Parameters
    ----------
    x : pd.DataFrame
        Independent variables / features for a statistical model
    y : pd.Series
        Dependent variables / target for a statistical model
    pipe : Callable
        Function that preprocesses data preparation for statistical modeling
        Must be a class that has a fit and transform method
    pipe_params : dict
        Dictionary of parameters for the pipe function
    model : Callable
        Function that models and scores data
        Must take x and y as an input; must have fit and predict (or predict_proba) methods
    model_params : TYPE
        Dictionary of parameters for the model function
    metric : Callable, optional
        A function that evaluates a model. The default is roc_auc_score.
    cv : Callable, optional
        A function that splits the data into train and test sets. 
        Must have a split  method that takes x and y as arguments. The default is ShuffleSplit.
    proba : str, optional
        Whether or not to use the predict proba method during model evaluation rather than the predict method. The default is True.

    Returns
    -------
    float
        The mean value of the fitness metric over the n-splits specified by the cv function

    '''
    pipeline = pipe(**pipe_params)
    pipeline.fit(x)
    
    x = pipeline.transform(x)
    x = x.values if type(x) == pd.core.frame.DataFrame else x
    
    splitter = cv()
    
    fitness = []
    for train_idx, test_idx in splitter.split(x, y):
        fit_model = model(**model_params)
        fit_model.fit(x[train_idx], y[train_idx])
        
        predictions = (fit_model.predict_proba(x[test_idx])[:, 1] if proba else
                       fit_model.predict(x[test_idx]))
        
        fitness.append(metric(y[test_idx], predictions))
    
    return np.mean(fitness)