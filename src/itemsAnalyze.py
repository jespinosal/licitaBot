

import pandas as pd
import os
from collections import Counter
import spacy
from readInput import textNormalize
import numpy as np
from nltk import ngrams
from itertools import chain
import re
from nltk.corpus import stopwords
import nltk.stem
from utils import getTime
import sys
import json



def readTextInput():
    validColumns = ['ACCIÓN', 'DOCUMENTO', 'ITEM', 'NUMERAL', 'NUMERO DE PROCESO', 'PAGINA', 'PARRAFO', 'TIPO DE VALOR', 'TITULO', 'VALOR']
    files = ['licitacionezItemz.xlsx','licitacionezItemz2.xlsx']
    path = '/home/espinosa/Escritorio/licitaBot/pliegosCondiciones'
    sheetCounter = 0
    for n,file in enumerate(files):
        #file = 'licitacionezItemz2.xlsx'
        # Load all excel Sheets
        xls = pd.ExcelFile(os.path.join(path,file))
        #df1 = pd.read_excel(xls, 'IDIGER-SA-SI-010-2018')
        #df2 = pd.read_excel(xls, 'IO-087-2018')


        # Get all the sheets in a unique dataFrame
        for sheetName in xls.sheet_names:
            if sheetName != 'Hoja1': # because of it is a example file
                if (sheetCounter == 0) and (n==0):
                    df = pd.read_excel(xls, sheetName)
                    sheetCounter +=1
                elif sheetCounter>0:
                    sheetCounter += 1
                    df = df[validColumns]
                    df = df.append(pd.read_excel(xls, sheetName))

                df[['PARRAFO']] = df[['PARRAFO']].astype(str)

        itemsBalance = Counter(df['ITEM'])

    return df



def stopWordsAnalysis(df,minRatio=0.1,maxRatio=0.9):
    allText = " ".join(dfClean['PARRAFO']).split()
    wordFrecuency = Counter(allText)
    wordFrecuency = sorted(wordFrecuency.items(), key=lambda kv: kv[1])


def porterStemmer(text):
    spanish_stemmer = nltk.stem.SnowballStemmer('spanish')
    text = [spanish_stemmer.stem(word) for word in text.split()]
    return " ".join(text)

def removeStopWords(text):
    stopWords = set(stopwords.words('spanish'))
    textList = [word for word in text.split() if word not in stopWords]
    text = " ".join(textList)
    return text

def dfDropNan(df):
    dfClean = df[df['PARRAFO'] != 'nan']
    return dfClean


def getWordsAnalysis(dfClean):
    # Use dfClean or df
    #analizar con porterStemmer tambien
    allText = " ".join(dfClean['PARRAFO']).split()
    wordFrecuency = Counter(allText)
    #Counter(np.array(list(wordFrecuency.values()))==1)
    wordFrecuency = sorted(wordFrecuency.items(), key=lambda kv: kv[1])
    # 9 enero decidir si tomar stop words de aqui o crear dict fijo
    return wordFrecuency

def getStopWordsFromText(text):
    df = readTextInput()
    df = dfDropNan(df)
    max = 0.03
    manualWords = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','x','y','w','z']# poner todos los elementos que consideren basura
    wordFrecuency = getWordsAnalysis(df)
    wordFrecuencyIndex = [frecuency for word, frecuency in wordFrecuency]
    invalidIndex = np.array(wordFrecuencyIndex) > sum(wordFrecuencyIndex) * max
    stopWords = [wordFrec[0] for wordFrec in (np.array(wordFrecuency)[invalidIndex])]
    print('filter high frecuency stopWords:', stopWords)
    stopWords = stopWords + manualWords
    textList = [word for word in text.split() if word not in stopWords]
    text = " ".join(textList)
    return text


def wordCount(text):
    return len(text.split())

def charCount(text):
    return len(text)

def wordFreq(text):
    wordFrecuency = Counter(text.split())
    wordFrecuency = sorted(wordFrecuency.items(), key=lambda kv: kv[1])

    return wordFrecuency

def getItemAnalysis():
    logPath = './analysisLog/'
    df = dfPreprocesing()
    #Items frecuency
    itemsBalance = Counter(df['ITEM'])
    df.groupby('ITEM').agg({'ITEM': ['count']})
    #Number frecuency
    #Number of words
    df['numeroPalabras']=df['PARRAFO'].apply(wordCount)
    #Number of Punctuation characters
    df['numeroCaracteres'] = df['PARRAFO'].apply(charCount)
    #Word frecuency by item
    dfItem = df.groupby('ITEM').agg({'PARRAFO': 'sum', 'numeroPalabras':'mean', 'numeroCaracteres':'mean'})
    dfItem = dfItem.reset_index()
    dfItem['frecPalabras'] = dfItem['PARRAFO'].apply(wordFreq)

    dfItem.to_csv(os.path.join(logPath, 'analysisItem_'+ getTime() + '.csv'), sep=',')
    with open (os.path.join(logPath,'itemsBalance_'+getTime()+'.json'),'w')  as outPutFile:
        json.dump(dict(itemsBalance),outPutFile)
    #Mean Page, Max - Min
    return getItemAnalysis



def buildItemCorpusDict(df):
    #Print a build python dictionary from dataFrame (no mandatory)
    items = set(df['ITEM'])
    itemsText = {}
    itemsCorpus = {}
    for item in items:
        itemTextList = list(df[df.ITEM == item][['PARRAFO']]['PARRAFO'])
        #itemsText[item] = [re.sub('\s\s+', ' ', parrafo.replace('\n',' ')) for parrafo in itemTextList if not parrafo=='nan']
        itemsText[item] = [textNormalize(parrafo.replace('\n', ' ')) for parrafo in itemTextList if not parrafo == 'nan']


    for item in itemsText.keys():
        print(item)
        for n, sample in enumerate(itemsText[item]):
            print('sample:', n, 'text:', sample)

    return itemsText




def textPreprocesing(text):
    #Nota: En este punto se construye el pipeline del procesamiento de texto. Se debe aplicar el mismo criterio en entrenamiento y predicción
    localStopWords,libraryStopWords,porterStemmer,normalizeNumber = (True, True, False, False)

    text = textNormalize(text.replace('\n', ' '))

    if normalizeNumber:
        print('True0')

    if porterStemmer:
        print('True1')

    if localStopWords:
        text = getStopWordsFromText(text)
        print('True3')

    if libraryStopWords:
        text = removeStopWords(text)
        print('True4')
    return text



def dfPreprocesing():
    df = readTextInput()
    df = df.filter(items=['PARRAFO','ITEM','PAGINA'])
    df = dfDropNan(df)
    #axisParrafo = list(df.columns).index('PARRAFO')
    df['PARRAFO'] = df['PARRAFO'].apply(textPreprocesing)

    return df



if __name__ == "__main__":

    getItemAnalysis = getItemAnalysis()

#in shell: python itemsAnalyze.py



# Corregir csv

# 1. Completar estadisticas ---> Done
# 2. Entrenar clasificador -----> NExt
# 3. Incorporar estadisticas en readInput
# 4. Agergar metodos para extraer en un df los candidatos (considerando estadisticas)
# 5. Calcular estadisticas
# 7. Realizar barrido sobre candidatos
