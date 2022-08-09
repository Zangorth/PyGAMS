###########
# Imports #
###########
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\PyGAMS\pygams')

from mate import choose_parents, rescuer, breed
from fitness import assess_fitness
from space import Space

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
    Arguments:
        space - the space object to be converted into list of space objects
    '''
    if type(space) == Space:
        return [space]
    
    elif type(space) == list and all(type(item) == Space for item in space):
        return space
    
    else:
        return None


###################
# Name that Space #
###################
def speciation(space: Space):
    '''
    Description - Ensures that each space has a name
    
    Parameters
    ----------
    space : Space
        A list of space objects

    Returns
    -------
    space : TYPE
        A list of named spaced objects
    '''
    for i in range(len(space)):
        if space[i].name is None:
            space[i].name = f'species_{i}'

    return space


#######################
# Generate Population #
#######################
def population_generator(models: list, pipes: list, population_size: int):
    '''
    Description - Generates the initial population for the genetic algorithm

    Parameters
    ----------
    models : list
        A list of space objects including the model options and their parameters
    pipes : list
        A list of space objects including the pipeline options and their parameters
    population_size : int
        The number of creatures to be included in the initial population

    Returns
    -------
    population : dict
        A dictionary of models, pipelines, and their parameters representing the 
        initial population to be used in the genetic algorithm
    '''
    population = []
    for i in range(population_size):
        model_space = np.random.choice(models)
        pipe_space = np.random.choice(pipes)
        
        creature = {'model_species': model_space.name.replace('species', 'model'),
                    'model_space': model_space,
                    'model': model_space.space_object,
                    'model_params': model_space.generate(),
                    'model_types': model_space.types,
                    'pipe_species': pipe_space.name.replace('species', 'pipe'),
                    'pipe_space': pipe_space,
                    'pipe': pipe_space.space_object,
                    'pipe_params': pipe_space.generate(),
                    'pipe_types': pipe_space.types,
                    'fitness': []}
        
        population.append(creature)
        
    return population


##########
# PyGAMS #
##########
class PyGAMS():
    def __init__(self, models, pipes=None, metric=roc_auc_score, cv=ShuffleSplit, 
                 generations=100, population_size=100, survivors=10, mutation_rate=0.1):
        '''
        Description - Class of functions uses a genetic algorithm to select the optimal model specification

        Parameters
        ----------
        models : Space or list of spaces
            The model space(s) to be optimized over
        pipes : Space or list of spaces, optional
            The pipeline space to be optimized over
        metric : callable, optional
            The scoring metric to use when evaluating the model. 
            Must be provided as a function/callable that takes y_true and y_score as arguments. 
            The default is roc_auc_score.
        cv : callable, optional
            The method of cross-validation to be used when evaluating the model. 
            Must be provided as a function/callable that takes x and y as arguments.
            The default is ShuffleSplit.
        generations : int, optional
            The number of iterations/generations to use in the genetic algorithm. 
            The default is 100.
        population_size : int, optional
            The number of creatures (models/pipelines) to evaluate per generation. 
            The default is 100.
        survivors : int, optional
            The top n creatures (models/pipelines) to keep between generations. 
            The default is 10.
        mutation_rate : float, optional
            The probability of a parameter to mutate (randomly change) during the mating process. 
            The default is 0.1.

        Returns
        -------
        None.

        '''
        if pipes is None:
           pipes = [Space(PassPipe)] 
        
        models, pipes = space_converter(models), space_converter(pipes)
        
        if models is None or pipes is None:
            print('Models and Pipes must be either space object or list of space objects')
            return None
        
        self.models, self.pipes = speciation(models), speciation(pipes)
        
        self.metric = metric
        self.cv = cv
        self.generations = generations
        self.population_size=population_size
        self.survivors = survivors        
        self.mutation_rate = mutation_rate
        
        return None    
    
    def run(self, x, y, n_jobs=1, proba=True):
        population = population_generator(self.models, self.pipes, self.population_size)
        
        for i in range(self.generations):
            fitness = [assess_fitness(x, y, 
                                      pipe=creature['pipe'], pipe_params=creature['pipe_params'],
                                      model=creature['model'], model_params=creature['model_params'],
                                      metric=self.metric, cv=self.cv, proba=proba)
                       for creature in population]
            
            for i in range(len(fitness)):
                population[i]['fitness'].append(fitness[i])
                
            survival_population = rescuer(fitness, population, self.survivors)
            parent_population = choose_parents(population, num_children=self.population_size-self.survivors)
            child_population = [breed(population, parents, self.mutation_rate) for parents in parent_population]
            population = survival_population + child_population
        
        return survival_population
    