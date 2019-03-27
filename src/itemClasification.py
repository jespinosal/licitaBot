from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas
from sklearn.metrics import confusion_matrix
from itemsAnalyze import dfPreprocesing


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



trainDF = dfPreprocesing()
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['PARRAFO'], trainDF['ITEM'])
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


#Count Vectors as features
# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['PARRAFO'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)

# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print ("NB, Count Vectors: ", accuracy)


# HACER PARTICIONES POR CLASE
# OBTENER MODELO
# HACER PREDICCIONES POR CLASE (particion clases balanceadas)
# ENTRENAR OTROS MODELOS
# GENERAR CLASE OTROS TEXTOS RANDOM

classifier = naive_bayes.MultinomialNB()
classifier.fit( xtrain_count, train_y)
predictions = classifier.predict(xvalid_count)

encoder.inverse_transform(predictions)



# 1. ###########################################################################################
#@todo acc por clase matriz confusión
#@todo cross validation
#@todo tuning model
#@todo reply with pipeline
trainDF = dfPreprocesing()
y_true= np.empty(0)
y_predict = np.empty(0)
acc = []
encoder = preprocessing.LabelEncoder()
encoder.fit(trainDF['ITEM']) ; print(encoder.classes_)
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
#Tfidf_vect = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

count_vect.fit(trainDF['PARRAFO'])
#Tfidf_vect.fit(trainDF['PARRAFO'])


for i in range(0,1000):
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['PARRAFO'], trainDF['ITEM'], stratify=trainDF['ITEM'], test_size = 0.7)
    train_y = encoder.transform(train_y)
    valid_y = encoder.transform(valid_y)
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    classifier = naive_bayes.MultinomialNB()
    classifier.fit( xtrain_count, train_y)
    predictions = classifier.predict(xvalid_count)
    acc.append(metrics.accuracy_score(predictions, valid_y))
    y_predict = np.append(y_predict,predictions)
    y_true = np.append(y_true,valid_y)
    print(i,acc[-1])
print(np.mean(acc))

# Metrics
# Confusion matrix
totalPredictions = len(y_true)
cm = confusion_matrix(y_true,y_predict, labels = encoder.transform((encoder.classes_)))
cm_normalized = cm.astype('float')/cm.sum(axis=1)#[:np.newaxis]
#cmRatio = cm/ @todo aqui voy
print([(n,item)for n,item in enumerate(encoder.classes_)])
metrics.accuracy_score(y_true, y_predict)
metrics.balanced_accuracy_score(y_true, y_predict)

plt.figure()
plot_confusion_matrix(cm, classes=encoder.classes_,
                      title='Confusion matrix, without normalization', normalize=True)
plt.show()

# 2. FLUJO PREDICCIÓN #########################################################################################
############################################################################################
# @todo opcion leer o entrenar modelo 27 enero de 2018, generar log(train info), guardar modelos
# poner paramtros features y modelos ---> grabar y leer
# leer ultimo modelo de path
from sklearn.naive_bayes import MultinomialNB
modelBestParameters = {'alpha': 1e-05}


trainDF = dfPreprocesing() # load trainingData

encoder = preprocessing.LabelEncoder()
encoder.fit(trainDF['ITEM'])
print('Categories:',encoder.classes_)

text_clf = grid_search.best_estimator_
# text_clf = Pipeline(steps = [('vect', CountVectorizer()),
# ('tfidf', TfidfTransformer( norm=best_parameters['tfidf'].get_params()['norm'],
#                             )),
# ('clf', MultinomialNB(best_parameters['clf'].get_params()['alpha']))])
#
# text_clf.set_params(**grid_search.best_params_)

labels = encoder.transform(trainDF['ITEM'])
print("Train Samples:",len(labels))
text_clf.fit(trainDF['PARRAFO'] , labels  )

outputPredictions = text_clf.predict(trainDF['PARRAFO'])
print('Training Accuracy',metrics.accuracy_score(labels, outputPredictions))
print('Training Balanced Accuracy',metrics.balanced_accuracy_score(labels, outputPredictions))


cm = confusion_matrix(labels,outputPredictions, encoder.transform((encoder.classes_)))
cm_normalized = cm.astype('float')/cm.sum(axis=1)#[:np.newaxis]
print([(n,item)for n,item in enumerate(encoder.classes_)])


plt.figure()
plot_confusion_matrix(cm, classes=encoder.classes_,
                      title='Confusion matrix, without normalization', normalize=True)
plt.show()


# 3 ##########################################################################################

# Tuning CV and PIPELINE and combine Count TFidf
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

trainDF = dfPreprocesing()
encoder = preprocessing.LabelEncoder()
encoder.fit(trainDF['ITEM']) ; print(encoder.classes_)
categories = encoder.classes_
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(),
    ('clf', SGDClassifier()),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
     'tfidf__norm': ('l1', 'l2'),
    'clf__max_iter': (5,),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__max_iter': (10, 50, 80),
}

grid_search = GridSearchCV(pipeline, parameters, cv=10,
                           n_jobs=-1, verbose=1)
t0 = time()
grid_search.fit(trainDF['PARRAFO'], encoder.transform(trainDF['ITEM']))
print("done in %0.3fs" % (time() - t0))

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()



##################################################################################

# Tuning CV and PIPELINE and combine Count TFidf
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import naive_bayes
import pickle
#import time
import os
from utils import getTime()
from utils import readLastModel()


trainDF = dfPreprocesing()
encoder = preprocessing.LabelEncoder()
encoder.fit(trainDF['ITEM']) ; print(encoder.classes_)
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
grid_search.fit(trainDF['PARRAFO'], encoder.transform(trainDF['ITEM']))
print("done in %0.3fs" % (time() - t0))

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
bestModelPipeline = grid_search.best_estimator_
fileName = 'modelLicitaBot'
modelName = fileName+'_'+getTime()
modelPath = os.path.join('./models',modelName)
pickle.dump(bestModelPipeline, open(modelPath, 'wb'))






loaded_model = pickle.load(open(readLastModel(), 'rb'))