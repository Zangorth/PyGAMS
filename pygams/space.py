###########
# Imports #
###########
from scipy.stats import loguniform
from numpy import random, arange
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\PyGAMS\pygams')

import rng

########################
# Passthrough Pipeline #
########################
class PassPipe():
    '''
    Description - Passthrough pipeline that returns the same dataframe it was fed
    
    Arguments:
        None
    '''
    def __init__(self):
        return None
    
    def fit(self, df):
        return None
    
    def transform(self, df):
        return df


##################
# Pipeline Adder #
##################
class Space():
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
        Description - Creates a search space that returns an integer
        
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
        Description - Creates a search space that returns a real valued number
        
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
        
        return None
        
    def Category(self, parameter: str, choices: list, p=None):
        '''
        Description - Creates a search space that returns a category
        
        Arguments:
            parameter    - The name of the parameter to be optimized, for example the learning rate of n_estimators
                           parameter names should match with what they are called in the model/pipeline function
            choices      - A list of options that define the search space for this parameter
            p            - A list of probability (0, 1) corresponding to the initial likelihood of choosing each option
        '''
        def generator():
            return random.choice(choices, p=p)
        
        self.generators[parameter] = generator
        
        return None
    
    def Categories(self, parameter: str, choices: list, n=None, low=None, high=None, p=None):
        '''
        Description - Creates a search space that returns a list of categories
        
        Arguments:
            parameter    - The name of the parameter to be optimized, for example the learning rate of n_estimators
            choices      - A list of options that define the search space for this parameter
            n            - The number of items to be chosen from the list of choices
            low/high     - The lower and upper bounds (inclusive) on the number of items to be chosen from the list
                           For example, low=2, high=10 will return between 2 and 10 items from the list, with every
                           number between 2 and 10 being equally likely to be chosen
            p            - A list of probability (0, 1) corresponding to the initial likelihood of choosing each option
        '''
        def generator():
            if n is not None:
                size = n
                
            elif low is not None and high is not None:
                size = random.choice(arange(low, high+1))
                
            else:
                print('Either n or both high and low must be specified')
                return None
            
            return list(random.choice(choices, size=size, p=p))
        
        self.generators[parameter] = generator

    def generate(self):
        output = {key: self.generators[key]() for key in self.generators.keys()}
        
        return output