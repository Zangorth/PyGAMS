###########
# Imports #
###########
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


#############
# Survivors #
#############
def rescuer(fitness, population, survivors):
    population = np.array(population)
    fitness = np.array(fitness)
    
    return list(population[(-fitness).argsort()][0:survivors])

#########################
# Selection Probability #
#########################
def select_p(df, fit_col='fitness'):
    normalized_fitness = MinMaxScaler().fit_transform(df[fit_col].to_numpy().reshape(-1, 1))
    p = normalized_fitness / normalized_fitness.sum()
    
    return p

#################
# Mate Selector #
#################
def mate_selector(df):
    return np.random.choice(df['mom_index'], p=df['mom_p'])

########
# Mate #
########
def choose_parents(population, num_children):
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
def breeder(mom, dad, param_set, mutation_rate):
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
def breed(population, parents, mutation_rate):   
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
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    