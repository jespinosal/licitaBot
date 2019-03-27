from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas as pd
from sklearn.metrics import confusion_matrix
from itemsAnalyze import dfPreprocesing

from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.linear_model import naive_bayes
import pickle
import os
from utils import plot_confusion_matrix
from utils import getTime
from utils import readLastModel
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import sys


def generateLog(cm,cm_normalized,accuracy,balancedAccuracy,modelName,encoder, time):
    encoder = pickle.load(open('./models/encoder/encoderLicitabot', 'rb'))
    logPath = './models/modelsLog'
    logName = os.path.join(logPath,modelName)

    accuracyDict = {}
    labels = encoder.classes_
    for n,line in enumerate(cm_normalized):
        accuracyDict[labels[n]] = line[n]


    with open(logName+'.txt', 'w') as data:
        data.write(str(accuracyDict))
        print('AccuracyDict saved')

    dfCm = pd.DataFrame(cm_normalized).set_index(labels)
    dfCm.columns = encoder.classes_
    dfCm.to_csv(os.path.join(logPath, modelName + '.csv'), sep=',')
    print('ConfMatrix saved')

    logLines = ['accuracy:'+str(accuracy),'trainingTime:'+str(time),'balancedAccuracy:'+str(balancedAccuracy)]
    #@todo hacer esto con train voy aqui
    logPath = './models/modelsLog'
    with open(os.path.join(logPath,modelName+'Summary.txt'),'w') as file:
        file.writelines(logLines)



def modelMetrics(labels,outputPredictions,encoder):
    accuracy = metrics.accuracy_score(labels, outputPredictions)
    balancedAccuracy =  metrics.balanced_accuracy_score(labels, outputPredictions)
    print('Training Accuracy', accuracy)
    print('Training Balanced Accuracy', balancedAccuracy)

    cm = confusion_matrix(labels, outputPredictions, encoder.transform((encoder.classes_)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)  # [:np.newaxis]
    print([(n, item) for n, item in enumerate(encoder.classes_)])

    #plt.figure()
    #plot_confusion_matrix(cm, classes=encoder.classes_,
                          #title='Confusion matrix, without normalization', normalize=True)
    #plt.show()
    return cm,cm_normalized,accuracy,balancedAccuracy



def tuningModel():
    ########## Read Data
    trainDF = dfPreprocesing()
    ########## Tuning
    encoder = preprocessing.LabelEncoder()
    encoder.fit(trainDF['ITEM']) ; print(encoder.classes_)
    labels = encoder.transform(trainDF['ITEM'])
    categories = encoder.classes_
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', naive_bayes.MultinomialNB()),
    ])


    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
         'tfidf__norm': ('l1', 'l2'),
        #'clf__max_iter': (5,),
        'clf__alpha': (0.00001, 0.000001),
        #'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__max_iter': (10, 50, 80),
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=10,
                               n_jobs=-1, verbose=1)
    t0 = time()
    grid_search.fit(trainDF['PARRAFO'], labels)
    tuningTime = time() - t0
    print("done in %0.3fs" % (tuningTime))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")


    best_parameters = grid_search.best_estimator_.get_params()
    bestModelPipeline = grid_search.best_estimator_
    outputPredictions = bestModelPipeline.predict(trainDF['PARRAFO'])

    #metrics
    cm, cm_normalized, accuracy, balancedAccuracy = modelMetrics(labels, outputPredictions, encoder)
    encoderName = 'encoderLicitabot'
    fileName = 'tuningModelLicitaBot'
    modelName = fileName+'_'+getTime()
    modelPath = os.path.join('./models/tuningModels/',modelName)
    encoderPath = os.path.join('./models/encoder/',encoderName)
    # Storage pipeline
    pickle.dump(bestModelPipeline, open(modelPath, 'wb'))
    pickle.dump(encoder, open(encoderPath, 'wb'))

    # Write log
    generateLog(cm, cm_normalized, accuracy, balancedAccuracy, modelName, encoder, tuningTime)






def loadModel(model='tuning'):
    loaded_model = pickle.load(open(readLastModel(model), 'rb'))
    return loaded_model

def trainBestModel():

    encoder = pickle.load(open('./models/encoder/encoderLicitabot','rb'))
    trainDF = dfPreprocesing()
    t0 = time()
    #encoder = preprocessing.LabelEncoder()
    #encoder.fit(trainDF['ITEM'])
    print('Categories:', encoder.classes_)

    text_clf = loadModel(model='tuning')#grid_search.best_estimator_
    # text_clf = Pipeline(steps = [('vect', CountVectorizer()),
    #('tfidf', TfidfTransformer( norm=best_parameters['tfidf'].get_params()['norm'],
    #                             )),
    # ('clf', MultinomialNB(best_parameters['clf'].get_params()['alpha']))])
    #
    # text_clf.set_params(**grid_search.best_params_)

    labels = encoder.transform(trainDF['ITEM'])
    print("Train Samples:", len(labels))
    text_clf.fit(trainDF['PARRAFO'], labels)

    outputPredictions = text_clf.predict(trainDF['PARRAFO'])

    #metrics
    cm, cm_normalized, accuracy, balancedAccuracy = modelMetrics(labels, outputPredictions, encoder)

    fileName = 'productionModelLicitaBot'
    modelName = fileName + '_' + getTime()
    modelPath = os.path.join('./models/productionModels/', modelName)

    pickle.dump(text_clf, open(modelPath, 'wb'))
    trainingTime = time() - t0
    generateLog(cm, cm_normalized, accuracy, balancedAccuracy, modelName, encoder, trainingTime)

def predictGenerator(input):
    encoder = pickle.load(open('./models/encoder/encoderLicitabot', 'rb'))
    #Grabar log datos performance
    #Incluir args para llamar script
    productionModel = loadModel(model='production')
    predictions = productionModel.predict(input)
    #@todo llamar Ngrams candidatos con esta func 16 feb
    return encoder.inverse_transform(predictions)



if __name__ == "__main__":

    if sys.argv[1] == 'True':
        tuningModel()
        print('Develop model its DONE')
    else:
        trainBestModel()
        print('Production model its DONE')

