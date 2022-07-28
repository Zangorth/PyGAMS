###########
# Imports #
###########
from scipy.stats import loguniform
from numpy import random
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\PyGAMS\pygams')

import rng

##################
# Pipeline Adder #
##################
# Descsription - Defines the search space for a model or pipeline object in GAMS
class GAMS_Space():
    '''
    Description - Defines the search space for a model or pipeline object in GAMS
    
    Arguments:
        space_object - A pipeline or model function. 
                       Pipelines should have a fit and transform method
                       Models should have a fit and predict method
    '''
    def __init__(self, space_object):
        self.space_object = space_object
        self.generators = {}
        
        return None

    def Integer(self, parameter: str, low: int, high: int, distribution='uniform'):
        '''
        Description - Creates a search space for integer parameters
        
        Arguments:
            parameter    - The name of the parameter to be optimized, for example the learning rate of n_estimators
                           parameter names should match with what they are called in the model/pipeline function
            low          - The lower bound of the search space
            high         - The upper bound of the search space
            distribution - The method of drawing samples from the search space
                           options include 'uniform', 'log-uniform', and 'exponential-decay'
        '''
        if distribution == 'uniform':
            def generator():
                return random.randint(low, high)
        
        elif distribution == 'log-uniform':
            def generator():
                return rng.lu_int(low, high)
            
        elif distribution == 'exponential-decay':
            def generator():
                return rng.exp_int(low, high)
            
        self.generators[parameter] = generator
        
        return None
        
    def Real(self, parameter: str, low: float, high: float, distribution='uniform'):
        '''
        Description - Creates a search space for real valued parameters
        
        Arguments:
            parameter    - The name of the parameter to be optimized, for example the learning rate of n_estimators
                           parameter names should match with what they are called in the model/pipeline function
            low          - The lower bound of the search space
            high         - The upper bound of the search space
            distribution - The method of drawing samples from the search space
                           options include 'uniform', 'log-uniform', and 'exponential-decay'
        '''
        if distribution == 'uniform':
            def generator():
                return random.uniform(low, high)
            
        elif distribution == 'log-uniform':
            def generator():
                return loguniform.rvs(low, high)
            
        elif distribution == 'exponential-decay':
            def generator():
                return rng.exp_real(low, high)
            
        self.generators[parameter] = generator
        
    def Category(self, parameter: str, choices: list, p=None):
        '''
        Description - Creates a search space for categorical parameters
        
        Arguments:
            parameter    - The name of the parameter to be optimized, for example the learning rate of n_estimators
                           parameter names should match with what they are called in the model/pipeline function
            choices      - A list of options that define the search space for this parameter
            p            - A list of probability (0, 1) corresponding to the initial likelihood of choosing each option
        '''
        def generator():
            return random.choice(choices, p=p)
        
        self.generators[parameter] = generator
            


########
# GAMS #
########
class GAMS():
    def __init__(self, pipes=[], models=[]):
        self.pipes, self.models = pipes, models
        
        return None
        
    
def Pipeline(df):
    return df

pipe = GAMS_Space(Pipeline)
pipe.Integer('years', 1, 7)
pipe.Real('lr', 1, 100, 'log-uniform')
pipe.Category('Bucket', ['Current', '1-30', '31+'], [0.5, 0.25, 0.25])