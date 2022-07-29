###########
# Imports #
###########
from pygams import Space, PassPipe
import pandas as pd

##############
# Space Test #
##############
def space_test():
    pipe = Space(PassPipe)
    
    pipe.Integer('years1', 1, 5)
    pipe.Integer('years2', 1, 10, 'log-uniform')
    pipe.Integer('years3', 1, 100, 'exponential-decay')
    
    pipe.Real('lr1', 0, 1)
    pipe.Real('lr2', 0.01, 1, 'log-uniform')
    pipe.Real('lr3', 1, 5, 'exponential-decay')
    
    pipe.Category('bucket', ['a', 'b', 'c', 'd'])
    
    test_frame = pd.DataFrame({
        'years1': [pipe.generators['years1']() for i in range(100000)],
        'years2': [pipe.generators['years2']() for i in range(100000)],
        'years3': [pipe.generators['years3']() for i in range(100000)],
        'lr1': [pipe.generators['lr1']() for i in range(100000)],
        'lr2': [pipe.generators['lr2']() for i in range(100000)],
        'lr3': [pipe.generators['lr3']() for i in range(100000)],
        'bucket': [pipe.generators['bucket']() for i in range(100000)]
        })
    
    print(('Success: ' if round(test_frame['years1'].mean(), 1) == 2.5 else 'Failure: ') + 'Integer Uniform')
    print(('Success: ' if round(test_frame['years2'].mean(), 1) == 3.9 else 'Failure: ') + 'Integer Log-Uniform')
    print(('Success: ' if round(test_frame['years3'].mean()) == 35 else 'Failure: ') + 'Integer Exponential Decay')
    print(('Success: ' if round(test_frame['lr1'].mean(), 1) == 0.5 else 'Failure: ') + 'Real Uniform')
    print(('Success: ' if round(test_frame['lr2'].mean(), 1) == 0.2 else 'Failure: ') + 'Real Log-Uniform')
    print(('Success: ' if round(test_frame['lr3'].mean(), 1) == 2.4 else 'Failure: ') + 'Real Exponential Decay')
    print(('Success: ' if round((test_frame['bucket'].value_counts()/len(test_frame)).mean(), 2) == 0.25 else 'Failure: ') + 'Categorical')
    
    
    
    
