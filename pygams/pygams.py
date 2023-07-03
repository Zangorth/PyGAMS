###########
# Imports #
###########
from pygams.helpers import PassPipe, space_converter, speciation, kwarg_gen
from pygams.populations import population_generator, population_to_df
from pygams.mate import choose_parents, rescuer, breed
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from pygams.fitness import assess_fitness
from matplotlib import pyplot as plt
from multiprocessing import Pool
from pygams.space import Space
import seaborn as sea
import pandas as pd
import numpy as np

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
                print(f'Generations Completed: {generation+1} / {self.generations}')
            
            kwargs = kwarg_gen(x, y, population, self.metric, self.cv, proba, n_jobs)

            if n_jobs == 1:
                fitness = [assess_fitness(**kwarg) for kwarg in kwargs]

            else:
                with Pool(n_jobs) as pool:
                    fitness = pool.starmap(assess_fitness, kwargs)

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
