###########
# Imports #
###########
from scipy.stats import loguniform
from numpy import random, abs

######################
# Loguniform Integer #
######################
def lu_int(low, high):
    if low <= 0:
        print('Error:\nLower Bound of Log Uniform Distribution must be >= 0\n')
        return None
        
    return(round(loguniform.rvs(low, high)))

#######################
# Exponential Integer #
#######################
def exp_int(low, high, decay=True):
    if high <= low:
        print('Error:\nHigher bound must be greater than Lower bound\n')
        return None

    scale = (high-low)/2
    
    output = low - 1
    while output < low or output > high:
        output = round(random.exponential(scale))
        output = output + low
    
    if not decay:
        output = abs((high-output)+low)

    return output


####################
# Exponential Real #
####################
def exp_real(low, high, decay=True):
    if high <= low:
        print('Error:\nHigher bound must be greater than Lower bound\n')
        return None
    
    scale = (high-low)/2
    
    output = low - 1
    while output < low or output > high:
        output = random.exponential(scale)
        output = output + low
    
    if not decay:
        output = abs((high-output)+low)

    return output
    
    

