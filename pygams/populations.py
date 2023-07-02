###########
# Imports #
###########
from collections import deque
import pandas as pd
import numpy as np

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