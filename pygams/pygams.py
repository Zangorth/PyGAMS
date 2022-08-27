###########
# Imports #
###########
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from multiprocessing import Pool
from collections import deque
import seaborn as sea
import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\PyGAMS')

from pygams.mate import choose_parents, rescuer, breed
from pygams.fitness import assess_fitness
from pygams.space import Space
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
        
        creature = {'model_species': model_space.name,
                    'model_space': model_space,
                    'model': model_space.space_object,
                    'model_params': model_space.generate(),
                    'model_types': model_space.types,
                    'pipe_species': pipe_space.name,
                    'pipe_space': pipe_space,
                    'pipe': pipe_space.space_object,
                    'pipe_params': pipe_space.generate(),
                    'pipe_types': pipe_space.types,
                    'fitness': []}
        
        population.append(creature)
        
    return population

####################
# Population to DF #
####################
def population_to_df(models: list, pipes: list, population: dict):
    '''
    Description - Converts a population of creatures from a dictionary format to a dataframe format

    Parameters
    ----------
    models : list
        A list of space objects including the model options and their parameters
    pipes : list
        A list of space objects including the pipeline options and their parameters
    population : dict
        A dictionary of models, pipelines, and their parameters 

    Returns
    -------
    pd.DataFrame
        A dataframe of models, pipelines, and their parameters
    '''
    def creature_to_row(creature: dict, columns: list, kind: str, index: int):
        '''
        Description - Takes and individual creature and converts it into a row for the dataframe

        Parameters
        ----------
        creature : dict
            An individual member of the population or a dictionary collection of models, pipelines, and parameters
        columns : list
            The names of the columns to be used in the row
        kind : str
            'model' or 'pipe'
        index : int
            The index number to use for the row

        Returns
        -------
        pd.DataFrame
            A row of a dataframe for the population to df function

        '''
        data = deque([creature[f'{kind}_species']])
        for key in creature[f'{kind}_types'].keys():
            if creature[f'{kind}_types'][key] == 'cats':
                new_value = np.array(creature[f'{kind}_params'][key], dtype=object)
                
            else:
                new_value = creature[f'{kind}_params'][key]
            
            data.append(new_value)
            
        data = np.array(data, dtype=object).reshape(1, len(creature[f'{kind}_params'].keys())+1)
        
        return pd.DataFrame(data=data, index=[index], columns=columns)
    
    species_dict = {}
    for i in range(len(population)):
        species = population[i]['model_species'] + ' x ' + population[i]['pipe_species']
        
        model_columns = ['model_species'] + [species + ' | ' + key for key in population[i]['model_params'].keys()]
        pipe_columns = ['pipe_species'] + [species + ' | ' + key for key in population[i]['pipe_params'].keys()]
        
        model_merge = creature_to_row(population[i], model_columns, 'model', i)
        pipe_merge = creature_to_row(population[i], pipe_columns, 'pipe', i)
        
        if species not in species_dict:
            species_dict[species] = pd.concat([
                                               pd.DataFrame({'species': [species]}, index=[i]),
                                               pd.DataFrame({'fitness': [population[i]['fitness'][-1]]}, index=[i]),
                                               model_merge,
                                               pipe_merge
                                               ], axis=1)
        
        else:
            to_append = pd.concat([
                                   pd.DataFrame({'species': [species]}, index=[i]),
                                   pd.DataFrame({'fitness': [population[i]['fitness'][-1]]}, index=[i]),
                                   model_merge,
                                   pipe_merge
                                   ], axis=1)
            
            species_dict[species] = pd.concat([species_dict[species], to_append], axis=0)
            
    out_cols = ['fitness', 'species', 'model_species', 'pipe_species']
    for model in models:
        for pipe in pipes:
            out_cols += [f'{model.name} x {pipe.name} | {param}' for param in model.types.keys()]
            out_cols += [f'{model.name} x {pipe.name} | {param}' for param in pipe.types.keys()]
        
    out = pd.DataFrame(index=np.arange(0, len(population)), columns=out_cols)
    
    for species in species_dict:
        out = out.fillna(species_dict[species])
        
    return out

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
        
        self.models, self.pipes = speciation(models, 'model'), speciation(pipes, 'pipe')
        
        self.metric = metric
        self.cv = cv
        self.generations = generations
        self.population_size=population_size
        self.survivors = survivors
        self.mutation_rate = mutation_rate
        self.population_tracker = None
                
        return None    
    
    def run(self, x: pd.DataFrame, y: pd.Series, n_jobs=1, proba=True, verbose=False):
        '''
        Description - Runs the genetic algorithm and outputs a population of optimal models

        Parameters
        ----------
        x : pd.DataFrame
            DESCRIPTION.
        y : pd.Series
            DESCRIPTION.
        n_jobs : TYPE, optional
            DESCRIPTION. The default is 1.
        proba : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        survival_population : TYPE
            DESCRIPTION.

        '''
        population = population_generator(self.models, self.pipes, self.population_size)
        
        for generation in range(self.generations):
            if verbose:
                print(f'Generations Completed: {generation} / {self.generations}')
            fitness = [assess_fitness(x, y, 
                                      pipe=creature['pipe'], pipe_params=creature['pipe_params'],
                                      model=creature['model'], model_params=creature['model_params'],
                                      metric=self.metric, cv=self.cv, proba=proba)
                       for creature in population]
            
            for i in range(len(fitness)):
                population[i]['fitness'].append(fitness[i])
            
            if self.population_tracker is None:
                self.population_tracker = population_to_df(self.models, self.pipes, population)
                self.population_tracker['generation'] = generation
                
            else:
                to_append = population_to_df(self.models, self.pipes, population)
                to_append['generation'] = generation
                
                self.population_tracker = pd.concat([self.population_tracker, to_append], axis=0)
            
            survival_population = rescuer(fitness, population, self.survivors)
            parent_population = choose_parents(population, num_children=self.population_size-self.survivors)
            child_population = [breed(population, parents, self.mutation_rate) for parents in parent_population]
            population = survival_population + child_population
        
        return survival_population
    
    def plot_scores(self, kind='max', title=None, ylim=None):
        '''
        Description - Plots the scores gennerated by the genetic algorithm by generation

        Parameters
        ----------
        kind : str, optional
            Function to use when assessing the genetic algorithm. 
            By default the function will plot the 'max' score in each generation, 
                against the baseline of the average score for the first generation
            Other options include 'min' score in each generation, 'mean' score in 
                each generation, or any other function that would reasonably work with a pandas groupby
        title : str, optional
            The title of the graph to be used. 
            The default is None, which equates to {kind} {self.metric.__name__} by generation.
                An example of this would be 'Max Roc_Auc_Score By Generation'
        ylim : tuple, optional
            The ylim to be used for the plot. The default is None, which will automatically determine appropriate limits.

        Returns
        -------
        None.
        '''
        sea.set(style='whitegrid', rc={'figure.dpi': 300})
        
        pt = self.population_tracker.copy().reset_index(drop=True)
        
        plot_points = pt.groupby('generation')['fitness'].aggregate(kind)
        baseline = pt.loc[pt['generation'] == 0, 'fitness'].mean()
        
        if title is None:
            title = f'{kind} {self.metric.__name__} by generation'.title()
            
        if ylim is None:
            bot = min([min(plot_points), baseline]) - 0.5*pt.loc[pt['generation'] == 0, 'fitness'].std()
            top = max([max(plot_points), baseline]) + 0.5*pt.loc[pt['generation'] == 0, 'fitness'].std()
            ylim = (bot, top)
        
        fig, ax = plt.subplots(figsize=(16, 9))
        sea.lineplot(x=np.arange(0, self.generations), y=plot_points,
                     label=self.metric.__name__, ax=ax)
        sea.lineplot(x=np.arange(0, self.generations), y=baseline,
                     color='red', ls='dashed', label='Baseline Score')
        ax.set_ylim(ylim)
        ax.set_title(title)
        
        return None
    
    def distribution_options(self):
        '''
        Description - Returns a list of possible parameters that can be displayed by the plot_parameter function

        Returns
        -------
        list
            A list of possible parameters
        '''
        return [col for col in self.population_tracker.columns if col not in ['fitness', 'generation']]
    
    def plot_parameter(self, param: str, title=None, ylim=None):
        '''
        Description - Plot which shows how the distribution of a parameter is changing over time (by generation)

        Parameters
        ----------
        param : str
            The name of the parameter to be plotted. 
            A list of parameter options can be found using the distribution options function
        title : str, optional
            The title of the graph to be used. 
            The default is None, which equates to Distribution Of {param} By Generation.
        ylim : tuple, optional
            The ylim to be used for the plot. The default is None, which will automatically determine appropriate limits.

        Returns
        -------
        None.

        '''
        sea.set(style='whitegrid', rc={'figure.dpi': 300})
        
        pt = self.population_tracker.copy().reset_index(drop=True)
        
        param_type = (list if type(pt[param].dropna().iloc[0]) == np.ndarray else
                      float if str(pt[param].dropna().iloc[0]).replace('.', '', 1).isnumeric()
                      else str)
        
        if param_type is str:
            size = pt.loc[pt[param].notnull()].groupby([param, 'generation']).size().rename('count').reset_index()
            size['percent'] = size['count'] / size.groupby('generation')['count'].transform('sum')
            
            if title is None:
                title = f'Distribution of {param} by generation'.title()
            
            if ylim is None:
                ylim = (-0.01, 1.01)
            
            fig, ax = plt.subplots(figsize=(16, 9))
            sea.lineplot(x='generation', y='percent', hue=param, data=size)
            ax.set_title(title)
            ax.set_ylim(ylim)
            ax.set_ylabel('Percentage')
            plt.legend(title='')
            
        elif param_type is float:
            avg = pt.groupby('generation')[param].aggregate(['mean', 'sem']).reset_index()
            avg['sem'] = avg['sem'].fillna(0)
            
            if title is None:
                title = f'Mean value of {param} by generation'.title()
                
            if ylim is None:
                bot = min(avg['mean'] - 2.5*avg['sem'])
                top = max(avg['mean'] + 2.5*avg['sem'])
                ylim = (bot, top)
            
            fig, ax = plt.subplots(figsize=(16, 9))
            sea.scatterplot(x='generation', y='mean', data=avg,
                            ax=ax)
            ax.errorbar(x=avg['generation'], y=avg['mean'], yerr=1.96*avg['sem'],
                        color='black', ls='')
            ax.set_title(title)
            ax.set_ylabel('Mean Value')
            ax.set_ylim(ylim)
        
        else:
            # dummy_frame = pd.get_dummies(pt[param].dropna().apply(pd.Series).stack()).groupby(level=0).sum()
            # dummy_frame['generation'] = pt.loc[pt[param].notnull(), 'generation']
            
            # avg = pd.DataFrame(columns=['generation', 'feature', 'percentage'])
            # for generation in sorted(dummy_frame['generation'].unique()):
            #     generation_frame = dummy_frame.loc[dummy_frame['generation'] == generation].sum(axis=0)/self.population_size
            #     generation_frame = generation_frame.rename('percentage').reset_index()
            #     generation_frame['generation'] = generation
            #     generation_frame.columns = ['feature', 'percentage', 'generation']
            #     generation_frame = generation_frame[list(avg.columns)]
                
            #     avg = pd.concat([avg, generation_frame], axis=0).reset_index(drop=True)
            
            print('Parameter plots not yet implemented for lists of categories')
            print('These can be viewed manually by looking at PyGAMS().population_tracker')
            return None
