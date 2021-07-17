
def get_model(model):
    model_dict = {
            'svm': SVC(**params),
            'naive_bayes': MultinomialNB(**params),
            'random_forest': RandomForestClassifier(**params),
            'decision_tree': DecisionTreeClassifier(**params),
            'knn': KNeighborsClassifier(**params)

       }

    return model_dict[model]


def get_fspace(model):
    fspace_dict = {
        'naive_bayes':{
            'numFeatures': hp.uniform('numFeatures',100,2500),
            'alpha': hp.uniform('alpha', 0.0, 2.0)

        },
        'svm':{
            'numFeatures': hp.uniform('numFeatures',100,2500),
            'C': hp.uniform('C', 0, 10.0),
            'kernel': hp.choice('kernel', ['linear', 'rbf']),
            'gamma': hp.uniform('gamma', 0, 20.0)
        },
        'knn':{
            'numFeatures': hp.uniform('numFeatures',100,2500),
            'n_neighbors': hp.choice('knn_n_neighbors', range(1,50))
        },
        'random_forest':{
            'numFeatures': hp.uniform('numFeatures',100,2500),
            'max_depth': hp.choice('max_depth', range(1,20)),
            'max_features': hp.choice('max_features', range(1,5)),
            'n_estimators': hp.choice('n_estimators', range(1,20)),
            'criterion': hp.choice('criterion', ["gini", "entropy"])
        },
        'decision_tree':{
            'numFeatures': hp.uniform('numFeatures',100,2500),
            'max_depth': hp.choice('max_depth', range(1,20)),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'min_samples_split' : hp.choice('min_samples_split',range(1,20))

        },
    }


    return fspace_dict(model)

