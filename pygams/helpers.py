###########
# Imports #
###########
from sklearn.feature_extraction.text import CountVectorizer
from pygams.space import Space
import pandas as pd
import pygams

#############
# Pass Pipe #
#############
class PassPipe():
    def __init__(self):
        '''
        Description default pipeline that returns the same dataframe it was given

        Returns
        -------
        None.

        '''
        return None
    
    def fit(self):
        return None
    
    def transform(x: pd.DataFrame):
        return x

######################
# Convert Space Type #
######################
def space_converter(space: Space):
    '''
    Description - Converts Space object into a list of Space objects

    Parameters
    ----------
    space : Space
        The space object to be converted into list of space objects

    Returns
    -------
    TYPE
        A list of space objects

    '''
    if type(space) == pygams.space.Space:
        return [space]
    
    elif type(space) == list and all(type(item) == pygams.space.Space for item in space):
        return space
    
    else:
        return None


###################
# Name that Space #
###################
def speciation(space: Space, kind):
    '''
    Description - Ensures that each space has a name
    
    Parameters
    ----------
    space : Space
        A list of space objects

    Returns
    -------
    space : Space
        A list of named spaced objects
    '''
    for i in range(len(space)):
        if space[i].name is None:
            space[i].name = f'{kind}_{i}'

    return space

###################
# KWARG Generator #
###################
def kwarg_gen(x, y, population, metric, cv, proba, parallel):
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
    parallel : int
        Number of cores to be used in parallel processing

    Returns
    -------
    Iterable of keyword arguments
    '''
    output = []
    for creature in population:
        kwarg_dict = {'x': x, 'y': y,
                      'pipe': creature['pipe'], 'pipe_params': creature['pipe_params'],
                      'model': creature['model'], 'model_params': creature['model_params'],
                      'metric': metric, 'cv': cv, 'proba': proba}

        if parallel > 1:
            output.append(list(kwarg_dict.values()))
        else:
            output.append(kwarg_dict)

    return output


#################
# Param Counter #
#################
def param_counter(df):
    param_counts = []
    for param in df:
        param_counts.append(' '.join(param))

    vectorizer = CountVectorizer()

    param_counts = vectorizer.fit_transform(param_counts)
    param_counts = pd.DataFrame(param_counts.toarray(), 
                                columns=vectorizer.get_feature_names_out())

    param_counts = param_counts.sum().sort_values(ascending=False)
    param_counts = param_counts / param_counts.sum()

    return param_counts