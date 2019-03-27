from time import gmtime, strftime
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import itertools
import logging

def loggerMng():

    listaA = []
    logging.basicConfig(filename='./systemLog/logE.log', level=logging.ERROR,
                        format='[%(asctime)s] {%(filename)s:%(lineno)d} [%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    return logger

def natural_sort(key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    return sort_key



def readLastModel(model='tuning'):
    if model=='tuning':
        modelsPaths = glob.glob('./models/tuningModels/tuningModelLicitaBot*')
    elif model=='production':
        modelsPaths = glob.glob('./models/productionModels/productionModelLicitaBot*')
        modelsPaths.sort(key=natural_sort())
    return modelsPaths[-1]


def getTime():
    return strftime("%Y_%m_%d_%H_%M_%S", gmtime())


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

