###########
# Imports #
###########
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


#############
# Survivors #
#############
def rescuer(fitness: list, population: list, survivors: int):
    '''
    Description - Saves the top performing creatures in the population to inject 
                  into the next generation
    
    Parameters
    ----------
    fitness : list
        List of values indicating how well the model performed, with higher numbers being better
    population : list
        List of dictionaries containing the parameters for training a GA model (see population_generator function)
    survivors : int
        Number of creatures from the population to save

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    population = np.array(population)
    fitness = np.array(fitness)
    
    return list(population[(-fitness).argsort()][0:survivors])

#########################
# Selection Probability #
#########################
def select_p(df: pd.DataFrame, fit_col='fitness'):
    '''
    Description - Normalizes the fitness scores and ensures they sum to 1 to indicate 
                  the probability the creature is selected to mate

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe including the fitness of each creature as well as its species and index number
    fit_col : str, optional
        The column to be normalized. The default is 'fitness'.

    Returns
    -------
    p : pd.Series
        Series of values indicating the probability that the creature will be selected to mate

    '''
    normalized_fitness = MinMaxScaler().fit_transform(df[fit_col].to_numpy().reshape(-1, 1))
    p = normalized_fitness / normalized_fitness.sum()
    
    return p

#################
# Mate Selector #
#################
def mate_selector(df: pd.DataFrame):
    '''
    Description - Takes as input a dataframe with a randomly selected (based on fitness) "dad" creature 
                  and a list of potential mates and chooses a mate for that creature based upon the "mom" fitness

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with list of "dad" creatures and potential mates

    Returns
    -------
    np.array
        The index for the selected mate

    '''
    return np.random.choice(df['mom_index'], p=df['mom_p'])

########
# Mate #
########
def choose_parents(population: list, num_children: int):
    '''
    Description - Takes the population and chooses creatures to mate based upon their fitness score

    Parameters
    ----------
    population : list
        List of dictionaries containing the parameters for training a GA model (see population_generator function)
    num_children : int
        Number of child creatures to be generated, or (in other words) number of parents to choose

    Returns
    -------
    list
        List of dictionaries indicating which creatures to pair off for mating

    '''
    fitness_frame = pd.DataFrame({
        'index': np.arange(len(population)),
        'species': [creature['pipe_species'] + ' x ' + creature['model_species'] for creature in population],
        'fitness': [creature['fitness'][-1] for creature in population]
        })
    
    fitness_frame['p'] = select_p(fitness_frame)
    
    dads = np.random.choice(fitness_frame['index'], p=fitness_frame['p'], size=num_children)
    
    mates = fitness_frame.iloc[dads].copy()
    mates = mates[['index', 'species']]
    mates.columns = ['dad_index', 'species']
    mates['marriage'] = mates.groupby('dad_index').cumcount()+1
    mates['unique_id'] = 'Creature: ' + mates['dad_index'].astype(str) + ' | Partner: ' + mates['marriage'].astype(str)
    mates = mates[['species', 'unique_id', 'dad_index', 'marriage']]
    
    moms = fitness_frame[['index', 'species', 'fitness']].copy()
    moms.columns = ['mom_index', 'species', 'mom_fitness']
    
    dating_pool = mates.merge(moms, how='left', on='species')
    dating_pool = dating_pool.loc[dating_pool['dad_index'] != dating_pool['mom_index']]
    
    dating_pool['mom_p'] = np.nan
    for idx in dating_pool['unique_id'].unique():
        dating_pool.loc[dating_pool['unique_id'] == idx, 'mom_p'] = select_p(dating_pool.loc[dating_pool['unique_id'] == idx], 'mom_fitness')
    
    moms = dating_pool.groupby(['species', 'unique_id']).apply(mate_selector).rename('mom_index').reset_index()
    
    mates = mates.merge(moms, on=['species', 'unique_id'])
    mates = mates[['dad_index', 'mom_index']].copy()
    
    return [{'dad': mates.iloc[i, 0], 'mom': mates.iloc[i, 1]} for i in range(len(mates))]
    
###########
# Breeder #
###########
def breeder(mom: dict, dad: dict, param_set: str, mutation_rate: float):
    '''
    Description - Takes two models/pipelines and combines them to return a new model/pipeline

    Parameters
    ----------
    mom : dict
        The parameter set for the first creature
    dad : dict
        The parameter set for the second creature
    param_set : str
        Whether the set of parameters represents a model or a pipeline
    mutation_rate : float
        What percentage of the time to mutate the child creature

    Returns
    -------
    child_params : dict
        A dictionary of new parameters generated based on the parent models/pipelines

    '''
    child_params = {}
    
    for param in dad[f'{param_set}_params'].keys():
        child_space = [dad[f'{param_set}_params'][param], mom[f'{param_set}_params'][param]]
        
        if np.random.uniform() < mutation_rate:
            child_params[param] = dad[f'{param_set}_space'].generate()[param]
            
        elif mom[f'{param_set}_types'][param] is int:
            child_params[param] = int(round(np.mean(child_space)))
            
        elif dad[f'{param_set}_types'][param] is float:
            child_params[param] = np.mean(child_space)
            
        elif mom[f'{param_set}_types'][param] == 'cat':
            child_params[param] = np.random.choice(child_space)
            
        elif dad[f'{param_set}_types'][param] == 'cats':
            size = int(round((len(child_space[0]) + len(child_space[1]))/2))
            
            child_space = list(set(dad[f'{param_set}_params'][param] + mom[f'{param_set}_params'][param]))
            
            child_params[param] = list(np.random.choice(child_space, size=size, replace=False))
            
    return child_params

#########
# Breed #
#########
def breed(population: list, parents: dict, mutation_rate: float):
    '''
    Description - Takes as input to parent creatures and returns a child creature

    Parameters
    ----------
    population : list
        List of dictionaries containing the parameters for training a GA model (see population_generator function)
    parents : dict
        Dictionarys indicating which creatures to pair off for mating
    mutation_rate : float
        What percentage of the time to mutate the child creature

    Returns
    -------
    child : dict
        Dictionary representing a child creature
    '''
    mom = population[parents['mom']]
    dad = population[parents['dad']]
    
    child = {
        'model_species': dad['model_species'],
        'model_space': mom['model_space'],
        'model': dad['model'],
        'model_params': breeder(mom, dad, 'model', mutation_rate),
        'model_types': mom['model_types'],
        'pipe_species': dad['pipe_species'],
        'pipe_space': mom['pipe_space'],
        'pipe': dad['pipe'],
        'pipe_params': breeder(mom, dad, 'pipe', mutation_rate),
        'pipe_types': mom['pipe_types'],
        'fitness': []
        }
    
    return child
    