###########
# Imports #
###########
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