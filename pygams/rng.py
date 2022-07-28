###########
# Imports #
###########
from scipy.stats import loguniform
from numpy import random

######################
# Loguniform Integer #
######################
def lu_int(low, high):
    if low <= 0:
        print('Error:')
        print('Lower Bound of Log Uniform Distribution must be >= 0')
        print('')
        return None
        
    return(round(loguniform.rvs(low, high)))

#######################
# Exponential Integer #
#######################
def exp_int(low, high):
    scale = (high-low)/2
    
    output = low - 1
    while output < low or output > high:
        output = round(random.exponential(scale))
        output = output + low
    
    return output

####################
# Exponential Real #
####################
def exp_real(low, high):
    scale = (high-low)/2
    
    output = low - 1
    while output < low or output > high:
        output = random.exponential(scale)
        output = output + low
    
    return output
    
    

