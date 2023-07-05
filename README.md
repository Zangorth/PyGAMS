# PyGAMS
The Genetic Algorithm for Model Selection (GAMS) package is designed to do what the name would imply: implement a genetic algorithm that will assist in choosing a model. More broadly, the goal of the package is to help users select between different competing models and everything encapsulating what it means to be a data science model. For example, you can use this method to select between two candidate algorithms (e.g. a random forest vs. an SVM), select which features that algorithm should be trained on, select what data pre-processing methods should be used, conduct hyper-parameter tuning, and really help you decide between any other competing factors you might consider to when designing your model. All of these decisions will be optimized over whatever metric you choose simultaneously as it tests out different combinations of input parameters and searches the model space for the optimal set of parameters. 

With the general introductory blurb out of the way, I want to briefly discuss why I built this and some of the benefits of the package. I've had a personal interest in genetic algorithms for years, which came about when I was trying to conduct feature selection for a model I was building. There are a lot of ways you can go about selecting features for a model, but they all have different goals and can tell you different, contradictory things. For example, a basic method might tell you which features are most significant, which features minimize node impurity, or which features contribute the most to the output result. But, as a person who builds predictive models I don't know how much I should, or do, care about any of those things. I'd rather have a method that selects features that maximize AUROC, minimize RMSE, or in some other way optimize for a metric I care about rather than minimizing node impurity and hoping that correlates with a good predictive model. Genetic algorithms fit that bill, in that it is just a general optimization tool that you can use to optimize for any metric you want (AUROC included). 

There are, of course, other methods that allow you to do this. Bayesian optimization is a popular method that is probably better supported with pre-existing python packages such as scikit-optimize. However, in my experience, bayesian optimization tends to be a very slow process, particularly as the number of parameters you are optimizing over increases, due to the fact that Bayesian optimization runs sequentially. You test one model, do some calculations, learn some things, test a new model, and repeat one after the other. By contrast, with a genetic algorithm, you can test N models simultaneously, do some calculations, learn some things, then test N more models in batches. Because you can batch them (and the larger the batch, the faster you learn), you can set the process up to run in parallel, which (assuming a sufficiently powerful computer) can drastically reduce run times, so, personally, I've always been more of a fan of the genetic algorithm. 

Unfortunately when I started this project (many years ago) there were not many packages available that allowed you to plug and play a genetic algorithm in python. The few that did exist were poorly documented and certainly were not optimized for a Data Science use case. Even today, it seems to be more of a computer science tool that doesn't get applied to this domain very often. So, I decided to build one myself, in spaghetti code for that project, and slowly, over time, that original spaghetti has turned this package you see here. I'm sure there's still a long ways to go before I have the perfect package, but it has come a long way since its inception. 

## Installation Guide

## Example
In my opinion, the best form of documentation comes in the form of a good example. So, that is what we will have here. 

The first step in working this package is to have some data. Because I want this to be a reproducible example, we'll start by generating some data. To see the full code, visit the python package in the example, but I will snip the discussion pieces and post them here. 

### Generate data
The code below will create data, x and y, in which y is a binary variable based on the features x. 100 x features are generated, however, only 10 of those features are informative (predictive), so the goal of the algorithm will be to pick out those 10. I then convert x and y into a pandas format, just because I am more comfortable working in that format rather than using numpy arrays. 

```
x, y = make_classification(n_samples=1000, n_features=100, n_informative=10)

panda = pd.DataFrame(y, columns=['y'])
panda = panda.merge(pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])]),
                    how='outer', left_index=True, right_index=True)

x = panda.drop('y', axis=1).copy()
y = panda['y']
``` 

### Build a Pipeline
Once you have your data, the next step will be to develop a pipeline (or pipelines if you are testing between multiple pipelines) that can process the data in whatever way you want. This step is, strictly speaking, optional. If you do not have any pre-processing you wish to do to the data (or if you have already pre-processed it) you can skip this step. The package has a built in pre-processer which it can use if you do not provide one, and all that pre-processer does is pass the data through as is. 

But, assuming you do want to select between different pre-processing methods, I've included an example of one below. The first thing you will note is that this pre-processor is a class with fit and transform methods: this is the only requirement for the pipeline, it must be a class with fit and transform method that only take x as an argument. 

The decision points will come in through the init portion of the function, so in this case this pipeline will be choosing what features to include, what scaler to use, and what imputer to use. In the case of features, we want to use the genetic algorithm to select which features to include in the final model, so the pipeline needs to be able to take a list of features as an argument, and then subset the full data set to only include those selected features. Similarly, we want the genetic algorithm to select what scaler / imputer to use, so the pipeline needs to be able to take a scaler / imputer as an argument, implement those processes, and return the transformed output. Obviously this is just a small subset of things a pipeline can possibly do, but the general idea will be the same in every case, the pipeline just needs to be able to take the choice, set of choices, or number as an input and then use that input in some way to transform the data. 

```
class FeaturePipeline():
    def __init__(self, features, scaler, imputer):
		self.features = features
        self.scaler = scaler
        self.imputer = imputer
        
        return None
    
    def fit(self, x):
        df = x[self.features]

        self.scaler.fit(df)
        self.imputer.fit(df)

        return None
    
    def transform(self, x):
        df = x[self.features]

        output = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
        output = pd.DataFrame(self.scaler.transform(output), columns=df.columns)

        return output
```

### Establish Search space
Once you have your pipeline built, you will need to define the search space, or the set of options that you want to genetic algorithm to choose between. In principle this is a fairly straightforward process, but as the search space grows so too will the amount of code required to define that space, so keep that in mind as you are building your space. 

In the example below we define the search space for the pipeline we built above. In order to do so, you first need to create a space, using the Space class, and feed in the function associated with that space. Optionally, you can also name your space. This will help with readability on the back end (when examining results), especially if you are testing between two or more pipelines. If you do not give them a name, the function will just automatically name them sequentially, so it would come out as "pipeline_1", "pipeline_2", and so on. 

After creating the space, you then range for each of the inputs in the pipeline. In the pipeline above, we determined we wanted to search a space for three parameters (features, scaler, and imputer) so we need to define the options for each of those three inputs. In the case of features, we want it to choose a subset of options from a list, so we will use the categories method. This will take a list of potential options (in this case a list of features provided through x.columns) and return a subset of that list. The size of that subset is determined by the low and high parameters, so in this case we are asking it to choose between 4 and 15 features for the final model. You also need to provide the name of the pipeline parameter associated with this space, and it the names must match. 

Next, we want the pipeline to choose a scaler for us, and the options under consideration are the MinMaxScaler, the MaxAbsScaler, and the Standard scaler. In order to do this we use the category method, which takes a list of potential options and returns one of them. As above, you will also need to provide the name of the pipeline parameter that these options are associated with. The same logic will follow for the imputer, as it is just another category space.  
```
pipes = Space(FeaturePipline, name='FeaturePipeline')
pipes.Categories(parameter='features', 
                 choices=x.columns, low=4, high=15)
pipes.Category(parameter='scaler', choices=[MinMaxScaler(), MaxAbsScaler(), StandardScaler()])
pipes.Category(parameter='imputer', choices=[SimpleImputer(), SimpleImputer(strategy='median')])
```

Finally, you will also need to establish the search space for whichever models you intend to implement. For the purposes of this example, we want the genetic algorithm to select between two model algorithms, a traditional random forest and an extra trees random forest, and select some hyperparameters associated with each model algorithm. As in the case of the pipeline, the first step is to create a space for each model algorithm, provide the class associated with the model algorithm, and optionally provide a name (otherwise it will output as "model_1", "model_2", et cetera). The only requirement for a model algorithm is that it have fit and predict (or predict_proba) methods, otherwise you can pass any class you want to the space. 

After creating the space, you then need to give the options for each hyperparameter you want want to test. In this case, we want the genetic algorithm to select a number of estimators as an integer between 10 and 1000, and we want those options to have an exponential-decay distribution, so that it is more likely to select smaller numbers than larger numbers. We also want it to choose the max number of features using either the sqrt or log2 options, so we pass those in as a category. And finally we want the genetic algorithm to select a min impurity decrease that is between 0.0001 and 1, selecting between those numbers at random with a log-uniform distribution. This process is then repeated for the ExtraTreesClassifier, which has similar hyperparameter options. 

```
rf = Space(RandomForestClassifier, name='RandomForest')
rf.Integer('n_estimators', low=10, high=1000, distribution='exponential-decay')
rf.Category('max_features', choices=['sqrt', 'log2'])
rf.Real('min_impurity_decrease', low=0.0001, high=1, distribution='log-uniform')

et = Space(ExtraTreesClassifier, name='ExtraTrees')
et.Integer('n_estimators', low=10, high=1000, distribution='exponential-decay')
et.Category('max_features', choices=['sqrt', 'log2'])
et.Real('min_impurity_decrease', low=0.0001, high=1, distribution='log-uniform')
```

### Optimize the model
With your search spaces defined, the next step is to run the genetic algorithm. This is done by using the PyGAMS class. The parameters for this class are all listed below and we will discuss each of them in turn. 

The first two parameters are fairly straightforward. With the models parameter you just need to pass the model search spaces you created above to the class. If you are selecting between more than one model you should pass them as a list, however, if you are only choosing between hyperparameters for one model, then you can just pass it by itself. The same logic applies for the pipes parameter. Pass all of the pipelines you created above to the class through this parameter, as a list if you are choosing between multiple pipelines or (as in our case) by itself if you only have one pipeline. The pipes parameter is optional, if you do not wish to any data pre-processing you can just leave this field empty. 

Next you need to determine what metric you want the genetic algorithm to optimize for. This can be any function/class you want to pass, the only requirement being that it takes as options y_true and y_pred, and that it returns a single value as a score. I primarily work on classification projects, so I set the default to roc_auc_score, but you can use RMSE for continuous outcomes, or any other metric that you feel is appropriate. 

The rest of the parameters all relate to the genetic algorithm specifically and how you want it to perform. The population size indicates how many "creatures" you want to evaluate before moving on to the next generation. A creature here being defined as a pipeline with some set of parameters (a subset of the features, a scaler and an imputer) and a model with some set of hyperparameters (e.g. a random forest with 20 estimators and a min_impurity_decrease of 0.12). Each creature in the population is going to be evaluated and given a score based on the metric you chose before. A higher population will give you a better jumping off point. If you can find a creature, or a couple of creatures, with very high fitness in your first round, then the algorithm will have a good idea of where to start looking for the optimum. However, larger populations also mean more run time, since it has to evaluate each creature in the population. So it's going to be a push and pull of how fast do you want the algorithm to run versus how thorough do you want to be. 

The generations parameter then determines how many populations you want to evaluate. After each creature has been evaluated, the genetic algorithm will construct a new population based on the creatures with the based scores in the old population. If the genetic algorithm finds two features that are consistently associated with better fitness scores, then those features should become more prevalent in the next generation. And, on the other side of the coin, if the algorithm finds two features that are consistently associated with worse fitness scores, then those features should become less common in subsequent generations. From each generation to the next, the overall fitness of each population should increase approaching the optimal fitness, as such more generations will tend to lead to better final scores. However, again, more generations also means more models to test, so it will be a balance between how much time you have and how optimal you want the score to be. 

The survivors parameter indicates how many creatures you want to persist from one generation to the next. By default, we have the top 5 creatures survive to the next generation. Having some number of survivors can be beneficial for a number of reasons, perhaps the most prominent one being that you are guaranteed to have at least N good creatures in the new generation. Because the genetic algorithm is a stochastic process, it's theoretically possible that due to bad luck, all the creatures in the new generation are just worse than the creatures in the previous generation (particularly if you have a smaller population size), so including survivors will prevent you from going backwards for any extended period of time. A second benefit, however, is that you can include some battle tested creatures in your final generation. Cross validation is also a stochastic process and it is possible that you get a creature with a high fitness just by chance. If that creature becomes a surivor, it will be re-tested in the next generation and get a new fitness score. If it was just a lucky creature, then it should fall out of the population and not make it to the next generation, however, if it is a genuinely high performing creature, they you will see a consistent track record of high scores from generation to generation. 

The final parameter is the mutation rate. This parameter prevents the algorithm from getting stuck in local optima. Say, for example, in our example, we are trying to choose between the MinMaxScaler (MMS), MaxAbsScaler (MAS) and the StandardScaler (SS), and the MMS is the best scaler, however the MAS does fine and the SS does poorly. Over the generations the genetic algorithm identifies that the MAS performs well and the SS performs badly, but due to bad luck (e.g. getting paired with bad features, getting paired with bad hyperparameters, or just honest bad luck) the MMS has also been identified as a bad scaler. Therefore, after 2-3 generations, the genetic algorithm has completely knocked MMS out of contention and is no longer even testing it as a viable option. The algorithm is stuck in a local optima because it is no longer even testing potentially better combinations. 

The mutation rate then comes in to fix this problem. It essentially says how often do you want to just throw out the prior results and try something new. In this example, even though the genetic algorithm has "learned" that MMS is a bad scaler, how often do you just want to ignore that and try it anyways. This parameter is interpreted just as a percent, so 0.1 would imply that 10% of the time the algorithm will just throw out the prior results and try something new. This mutation rate applies to each parameter independently, so on any given creature there will be a 10% chance it throws out the scaler and tries a new scaler, a 10% chance it throws out the imputer and tries a new imputer, and (calculating the joint probability) a 1% chance it throws out both and tries something new for both. 

```
gams = PyGAMS(models=[rf, et], pipes=pipes, metric=roc_auc_score, population_size=20, generations=40, survivors=5, mutation_rate=0.1)
```

Once you have set these parameters, the final step in the process is to run the genetic algorithm. In order to do so, you need to provide your standard x and y, which just tells the algorithm what your features are and what your target is. The x should include all potential features, however, those features may be later subset by the pipeline you created. The n_jobs parameter allows you to run the algorithm in parallel. The only portion of the algorithm that is run in parallel is the population. So if you have a population_size of 20 and an n_jobs of 20, multi-processing will spin up 20 cores and evaluate each creature simultaneously, rather than doing them sequentially. However, if you increase your n_jobs over 20, then you will just be wasting cores and time as there are not more than 20 creatures to evaluate. If the verbose parameter is set to True, then after each generation you will receive a progress report regarding the fitness of that generation (based on your specified metric). 

```
model_selection = gams.run(x, y, n_jobs=20, verbose=True)
```

## File Guide
