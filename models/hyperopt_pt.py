from hyperopt import fmin, tpe,hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from .. import utils

####### Methods for minimization and logging each time step taken by hyperopt ####


class optimize_model:

    def __init__(self,fspace,algo,max_evals,df,message_tokenized,x,y):
        self.trials = Trials()
        self.fspace = fspace
        self.algo = algo
        self.max_evals = max_evals
        self.df = df
        self.message_tokenized = message_tokenized
        self.x = x
        self.y = y
        self.model_dict = {
            'svm': SVC(**params),
            'naive_bayes': MultinomialNB(**params),
            'random_forest': RandomForestClassifier(**params),
            'decision_tree': DecisionTreeClassifier(**params),
            'knn': KNeighborsClassifier(**params)

       }
    

    # Method to return stratified cross validation scores 

    def train_model(self):

        params = self.fspace.copy()
        numFeatures = params['numFeatures']   # Treat number of features as variable, fspace must contain numFeatures
        utils.get_features(df,numFeatures,self.message_tokenized,self.x)
        X = self.df[self.x].values
        del params['numFeatures']
        
        Y = self.df[self.y].values

        clf = self.model_dict['model']
        del params['model']

        return cross_val_score(clf,X,Y).mean()  #Default is StratifiedKFold for sklearn based classifiers
       


    
    def f(self):

        acc = self.train_model()

        return {'loss':-acc,'status':STATUS_OK}


    
    def get_optimum_hyperparameters(self):

        best = fmin(f,space = self.fspace,algo = self.algo, max_evals = self.max_evals,trials = self.trials)

        return best
    

   

