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
    def __init__(self, space_object, name=None):
        '''
        Description - Defines the search space for a model or pipeline object in GAMS    

        Parameters
        ----------
        space_object : callable
            A pipeline or model function. 
            Pipelines should have a fit and transform method
            Models should have a fit and predict method
        name : TYPE, optional
            The name to assign the creature (model/pipeline). The default is None.

        Returns
        -------
        None.
        '''
        self.space_object = space_object
        self.name = name
        self.types = {}
        self.generators = {}
        
        return None

    def Integer(self, parameter: str, low: int, high: int, distribution='uniform'):
        '''
        Description - Creates a search space that returns an integer

        Parameters
        ----------
        parameter : str
            The name of the parameter to be optimized, for example the learning rate of n_estimators
            parameter names should match with what they are called in the model/pipeline function
        low : int
            The lower bound of the search space
        high : int
            The upper bound of the search space
        distribution : TYPE, optional
            The method of drawing samples from the search space
            options include 'uniform', 'log-uniform', and 'exponential-decay'

        Returns
        -------
        TYPE
            An integer between the low and high values
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
        
        self.types[parameter] = int
        self.generators[parameter] = generator
        
        return None
        
    def Real(self, parameter: str, low: float, high: float, distribution='uniform'):
        '''
        Description - Creates a search space that returns a real valued number

        Parameters
        ----------
        parameter : str
            The name of the parameter to be optimized, for example the learning rate of n_estimators
            parameter names should match with what they are called in the model/pipeline function
        low : int
            The lower bound of the search space
        high : int
            The upper bound of the search space
        distribution : TYPE, optional
            The method of drawing samples from the search space
            options include 'uniform', 'log-uniform', and 'exponential-decay'

        Returns
        -------
        TYPE
            A float between the low and high values
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
        
        self.types[parameter] = float
        self.generators[parameter] = generator
        
        return None
        
    def Category(self, parameter: str, choices: list, p=None):
        '''
        Description - Creates a search space that returns a category

        Parameters
        ----------
        parameter : str
            The name of the parameter to be optimized, for example the learning rate of n_estimators
            parameter names should match with what they are called in the model/pipeline function
        choices : list
            A list of options that define the search space for this parameter
        p : TYPE, optional
            A list of probability (0, 1) corresponding to the initial likelihood of choosing each option. 
            The default is None, which assigns each choice an equal probability.

        Returns
        -------
        TYPE
            One element in the set of choices
        '''
        def generator():
            return random.choice(choices, p=p)
        
        self.types[parameter] = 'cat'
        self.generators[parameter] = generator
        
        return None
    
    def Categories(self, parameter: str, choices: list, n=None, low=1, high=None, p=None):
        '''
        Description - Creates a search space that returns a list of categories

        Parameters
        ----------
        parameter : str
            The name of the parameter to be optimized, for example the learning rate of n_estimators
            parameter names should match with what they are called in the model/pipeline function
        choices : list
            A list of options that define the search space for this parameter
        n : int, optional
            DESCRIPTION. The default is None.
        low : int, optional
            The lower bound on the number of items to be chosen from the list. 
            For example, low=2, high=10 will return between 2 and 10 items from the list, with every
            number between 2 and 10 being equally likely to be chosen
            The default is 1
        high : int, optional
            The upper bound on the number of items to be chosen from the list
            For example, low=2, high=10 will return between 2 and 10 items from the list, with every
            number between 2 and 10 being equally likely to be chosen
            The default is equal to the length of the list of choices.
        p : TYPE, optional
            A list of probability (0, 1) corresponding to the initial likelihood of choosing each option. 
            The default is None, which assigns each choice an equal probability.

        Returns
        -------
        TYPE
            N elements in the set of choices
        '''
        def generator():
            if n is not None:
                size = n
                
            elif low is not None and high is not None:
                size = random.choice(arange(low, high+1))
                
            else:
                print('Either n or both high and low must be specified')
                return None
            
            return list(random.choice(choices, size=size, p=p, replace=False))
        
        self.types[parameter] = 'cats'
        self.generators[parameter] = generator

    def generate(self):
        '''
        Description - Generates a possible set of parameters for model/pipeline that makes up the space
        
        Returns
        -------
        output : dict
            A dictionary of parameters for the model/pipeline that falls within the parameter space

        '''
        output = {key: self.generators[key]() for key in self.generators.keys()}
        
        return output