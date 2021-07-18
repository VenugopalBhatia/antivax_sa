from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import utils
# from .. import utils

####### Methods for minimization and logging each time step taken by hyperopt ####


class optimize_model:

    def __init__(self,model,algo,max_evals,df,message_tokenized,x,y,lib = 'sklearn'):
        self.trials = Trials()
        self.fspace = self.get_fspace(model)
        self.algo = algo
        self.max_evals = max_evals
        self.df = df
        self.message_tokenized = message_tokenized
        self.x = x
        self.y = y
        self.model = model
        self.lib = lib
    

    # Method to return stratified cross validation scores 

    def train_model(self,global_params,get_model=False):

        params = global_params.copy()
        numFeatures = params['numFeatures']   # Treat number of features as variable, fspace must contain numFeatures
        del params['numFeatures']
        
        if(self.lib == 'nltk'):
            utils.get_features(df = df,n = numFeatures,message_tokenized = self.message_tokenized,features = self.x,lib = self.lib)
            X = self.df[self.x].values
            
            
        else:
            X,x_feature_names = utils.get_features(df = self.df,n = numFeatures,message_tokenized = self.message_tokenized)
        Y = self.df[self.y].values
        
        if(self.model == 'svm'):
            clf = SVC(**params)
        elif(self.model == 'naive_bayes'):
            clf = BernoulliNB(**params)
        elif(self.model == 'random_forest'):
            clf = RandomForestClassifier(**params)
        elif(self.model == 'decision_tree'):
            clf = DecisionTreeClassifier(**params)
        elif(self.model == 'knn'):
            clf = KNeighborsClassifier(**params)
        else:
            return 0
        
        if(get_model):
            return cross_val_score(clf,X,Y),clf
        
        return cross_val_score(clf,X,Y).mean()  #Default is StratifiedKFold for sklearn based classifiers
       


    
    def f(self,params):

        acc = self.train_model(params)

        return {'loss':-acc,'status':STATUS_OK}


    
    def get_optimum_hyperparameters(self):

        best = fmin(self.f,space = self.fspace,algo = self.algo, max_evals = self.max_evals,trials = self.trials)

        return best,self.trials
    


    def get_fspace(self,model):
        fspace_dict = {
            'naive_bayes':{
                'numFeatures': hp.choice('numFeatures',range(100,2500)),
                'alpha': hp.uniform('alpha', 0.0, 2.0)

            },
            'svm':{
                'numFeatures': hp.choice('numFeatures',range(100,2500)),
                'C': hp.uniform('C', 0, 10.0),
                'kernel': hp.choice('kernel', ['linear', 'rbf']),
                'gamma': hp.uniform('gamma', 0, 20.0)
            },
            'knn':{
                'numFeatures': hp.choice('numFeatures',range(100,2500)),
                'n_neighbors': hp.choice('knn_n_neighbors', range(1,50))
            },
            'random_forest':{
                'numFeatures': hp.choice('numFeatures',range(100,2500)),
                'max_depth': hp.choice('max_depth', range(1,20)),
                'max_features': hp.choice('max_features', range(1,5)),
                'n_estimators': hp.choice('n_estimators', range(1,20)),
                'criterion': hp.choice('criterion', ["gini", "entropy"])
            },
            'decision_tree':{
                'numFeatures': hp.choice('numFeatures',range(100,2500)),
                'max_depth': hp.choice('max_depth', range(1,20)),
                'criterion': hp.choice('criterion', ["gini", "entropy"])

            },
        }


        return fspace_dict[model]

    


    

   

