from itemsModel import predictGenerator
from collections import defaultdict
from collections import Counter
import time
from readInput import readSeveralDocuments
from itemsAnalyze import dfPreprocesing
import pandas as pd
import os
import math
import numpy as np
import json
from utils import getTime
import sys
#@todo recuperar posiciones
# evaluar por item si se detecta  o no
# Reotrnar posiciones y string ganador!
# Encontrar regiones mayor densidad





def indexToPage(characterMap,indexStartEnd):
    '''
    Función para determinar la/las páginas que un token tiene asociado
    :param characterMap:
    :param indexList:
    :return:
    '''
    #start,end = indexList
    #start = 600
    #end = 800
    tupleStarEndPage=[]
    for start,end in indexStartEnd:
        startPage = np.where(np.array(list(characterMap[document].values()))[:,0] <= start)[0][-1]
        endPage = np.where(np.array(list(characterMap[document].values()))[:,0] <= end)[0][-1]
        tupleStarEndPage.append(list(set([startPage,endPage])))
    return tupleStarEndPage


def readTextInput2():
    validColumns = ['ACCIÓN', 'DOCUMENTO', 'ITEM', 'NUMERAL', 'NUMERO DE PROCESO', 'PAGINA', 'PARRAFO', 'TIPO DE VALOR', 'TITULO', 'VALOR','FILE']
    files = ['licitacionezItemKPI1.xlsx']
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
    df = df.filter(items=['ITEM','PAGINA','FILE'])
    return df



def kpiOutput(documentsSummary):
    dfKpi = readTextInput2()


    documents =  documentsSummary.keys()
    items = list(documentsSummary[list(documents)[0]].keys())

    itemsSummaryT= {}
    itemsSummaryP = {}
    itemsSummaryPU = {}
    for item in items:
        countItemTeoricoPos=0
        countItemTeoricoNeg = 0
        CountTP = 0
        CountTN = 0
        CountFN = 0
        CountFP = 0
        Other = 0
        CountTPU = 0
        CountTNU = 0
        CountFNU = 0
        CountFPU = 0


        for document in documents:
            documentTempTPU = 'None'
            documentTempTNU = 'None'
            paTeorica = dfKpi[(dfKpi.ITEM == item) & (dfKpi.FILE == document)]['PAGINA'].iloc[0]
            ngrams = documentsSummary[document][item]
            if not math.isnan(paTeorica):
                countItemTeoricoPos += 1
            else:
                countItemTeoricoNeg += 1

            sizePredictionsItem = len(ngrams)
            countItemTeorico=0
            if sizePredictionsItem==0:
                if not math.isnan(paTeorica):
                    CountFN += 1
                    CountFNU += 1
                elif math.isnan(paTeorica):
                    CountTN += 1
                    CountTNU += 1
                else:
                    Other += 1
                    print(item)
            else:
                for ngram in ngrams:
                    paPractica = ngram[-1]
                    if (paTeorica in paPractica) and (ngram[0]==item):
                        CountTP+=1
                    elif  (ngram[0]!=item) or (paTeorica not in paPractica):#(paPractica!=[]) & (math.isnan(paTeorica)) or
                        CountFP+=1
                    elif (paPractica==[]) & (not math.isnan(paTeorica)):
                        CountFN += 1
                    elif (paPractica==[]) & (math.isnan(paTeorica)):
                        CountTN +=1
                    else:
                        Other+=1

                for ngram in ngrams:
                    paPractica = ngram[-1]
                    if (paTeorica in paPractica) & (ngram[0]==item) & (documentTempTPU != document):
                        CountTPU+=1
                        documentTempTPU = document
                        #break
                    elif  (ngram[0]!=item) or (paTeorica not in paPractica):#(paPractica!=[]) & (math.isnan(paTeorica)) or
                        CountFPU+=1
                    elif (paPractica==[]) & (not math.isnan(paTeorica)):
                        CountFNU += 1
                    elif (paPractica==[]) & (math.isnan(paTeorica)) & (documentTempTNU != document):
                        CountTNU +=1
                        documentTempTNU = document
                    else:
                        Other+=1

        itemsSummaryP[item] = [CountTP,CountTN,CountFP,CountFN,Other,sizePredictionsItem]
        itemsSummaryPU[item] = [CountTPU, CountTNU, CountFPU, CountFNU] # Detecciones prácticas
        itemsSummaryT[item] = [countItemTeoricoPos,countItemTeoricoNeg] # Detecciones teóricas
            #voy aqui: REVISAR RESULTADOS!! Y SEPARA FP DE ITEM Y POSICION, COMENTAR CODIGO... COMPLEMENTAR CON COUNT POR ITEM
            #Hacer diccionario que cuente apariciones por item pada dividir el practivo
            #todo normalizar diviendo por documento practico, ejemplo poner break por documento para contar solo uno!!!
            # tener dos metricas practivas una en totales y otra por documento que solo cuente 1
    return itemsSummaryPU,itemsSummaryT





def choiceMostFrec(documentsSummary):

    def getNgramsIndexConsecutive(listaIndices):
        indexRanges=[]
        indicesTotales = []
        indicesParciales = []
        for n, li in enumerate(listaIndices[0:-2]): # Se recorren todas las tuplas menos la última
            diferencia = listaIndices[n][1] - listaIndices[n + 1][0] # calculo de diferencias de
            if diferencia > 0: # Si la resta del inicio de la tupla actual con el fin de la futura es mayor que cero, los Ngrams son consecutivos
                if indicesParciales==[]: # En el primer Ngram consecutivo se guardan el index actual y siguiente
                    indicesParciales.append(n)
                    indicesParciales.append(n+1)
                else: # despues que se guarde el actual y siguiente se sigue guardando solo siguiente
                    indicesParciales.append(n + 1)

            else:
                if indicesParciales != [] : # Si hay un indiceParcial se grabam la lista de indices consecutivos encontrad ... pte n==len(listaIndices)-3
                    indicesTotales.append(indicesParciales)
                    indexRanges.append((listaIndices[indicesParciales[0]][0],listaIndices[indicesParciales[-1]][1]))
                indicesParciales = [] # indicesParciales se inicializa cuando no se detecten indices consecutivos

        return indexRanges


    documentsSummaryItemIndex = defaultdict(lambda: defaultdict(list))
    documentsSummaryItemIndexRanges =  defaultdict(lambda: defaultdict(list))
    documentsSummaryFilter ={}
    for document,items in documentsSummary.items():
        for item,itemLists in items.items():
            if len(itemLists)>0:
                for itemList in itemLists:
                    if item == itemList[0]:
                        print(item)
                        documentsSummaryItemIndex[document][item].append(itemList[1])

                documentsSummaryItemIndex[document][item] = sorted(documentsSummaryItemIndex[document][item], key= lambda x:x[1])
                documentsSummaryItemIndexRanges[document][item] = getNgramsIndexConsecutive(documentsSummaryItemIndex[document][item])

    return documentsSummaryItemIndexRanges





if __name__ == "__main__":
    # predictionFlow
    initTime = time.time()
    inPath = '/home/espinosa/Escritorio/licitaBot/src/inputFiles'
    predictDocumentDict = defaultdict(lambda: defaultdict(dict))
    predictDocumentDictFrequencies = defaultdict(lambda: defaultdict(dict))
    tokensDocumentDict, characterMap, indexTokensDocumentDict = readSeveralDocuments(inPath)
    readTime = time.time() - initTime;
    #print("time to Read %fs" % readTime)
    for document, items in tokensDocumentDict.items():
        for item, tokens in items.items():
            if tokens == []:
                itemTokens = ['empty']
            else:
                itemTokens = predictGenerator(tokens)
            predictDocumentDict[document][item] = itemTokens  # Counter(itemTokens).most_common() # considerar grabar etiqueta al mismo dict entrada
            predictDocumentDictFrequencies[document][item] = Counter(itemTokens).most_common() # Contiene la frenciencia de items de las predicciones
    predictionTime = time.time() - readTime;
    #print("time to Predict %fs" % predictionTime)

    '''
    It method build a dicctionary with key:document and values where each value is:[label,(indexStart,indexEnd),page,text
    '''

    documentsSummary = defaultdict(lambda: defaultdict(dict))
    documentsSummaryCopy = defaultdict(lambda: defaultdict(dict))

    for document, items in predictDocumentDict.items():
        for item in items.keys():
            # @todo poner condicion si no existe item!
            indexStartEnd = indexTokensDocumentDict[document][item]
            documentsSummary[document][item] = list(
                zip(predictDocumentDict[document][item], indexStartEnd, tokensDocumentDict[document][item],
                    str(indexToPage(characterMap, indexStartEnd)))) #str bucause of 64bit json problems
            if sys.argv[1]=='True':
                documentsSummaryCopy[document][item] = list(
                    zip(predictDocumentDict[document][item], indexStartEnd, tokensDocumentDict[document][item],
                        indexToPage(characterMap, indexStartEnd)))  # str bucause of 64bit json problems


    # return documentsSummary
    documentsSummaryItemIndexRanges = choiceMostFrec(documentsSummary)

    path = './outPutFiles'
    fileNames =  {'predictDocumentDictFrequencies':predictDocumentDictFrequencies,'documentsSummary':documentsSummary, 'documentsSummaryItemIndexRanges':documentsSummaryItemIndexRanges}
    for fileName,outputData in fileNames.items():
        outputName = os.path.join( path ,fileName + '_' + getTime()+'.json')
        #outputDataJson = json.dumps(outputData)
        with open(outputName, 'w') as outfile:
            json.dump(outputData, outfile)
    print("time to Predict %fs" % predictionTime)

    if sys.argv[1]=='True':
        path = './outPutFilesTest'
        itemsSummaryPU, itemsSummaryT = kpiOutput(documentsSummaryCopy)
        outputNamePU = os.path.join(path, 'itemPredictionCounter' + '_' + getTime() + '.json')
        with open(outputNamePU, 'w') as outfile:
            json.dump(itemsSummaryPU, outfile)
        outputNameT = os.path.join(path, 'itemLabeledCounter'  + '_' + getTime() + '.json')
        with open(outputNameT, 'w') as outfile:
            json.dump(itemsSummaryT, outfile)


#in shell: predictionFlow.py True or predictionFlow.py False
#@todo voy aqui 20 marzo, poner agrs aqui y en train.


'''
A cada token asignar su tupla de indices
A cada tupla de indices asignar pagina a que pertenece

Generar diccioario documento,item, pagina de los .xls
Implementar KPI item practivo vs item teorico

'''

#Corto plazo entrega:
# 1. Mapa tokens posiciones.
# 2. Mapa items pagina
# 3. Mapa posiciones pagina


# Ideas futuro
# Filtrar ventana de pagina
# Crear Clase de basura
# Ignorar elementos consecutivos
# Al extraer numeros por reglas se pueden filtar muchos FP


'''
@todo penalizar por esar dentro del rango de paginas!!! y quitar pdf basura
document = 'ESTUDIOS PREVIOS - RUTA REVISADO ... DR. RAFAEL.pdf'
document = 'PLIEGO DE CONDICIONES 2018.pdf'
#document = 'DOCUMENTO COMPLEMENTARIO TRANSPORTE ESPECIAL 2018 .pdf'
#- document = 'estudios previos.pdf' no esta es imagen
#- document = 'ESTUDIOS PREVIOS TRANSPORTE DE FUNCIONARIOS 2018 y ANEXO TÉCNICO JUNIO 25 FIRMADOS.pdf' # es una mierda mirar CUMPLIMIENTO, MULTAS Y CLAUSULA PENAL PECUNIARIA imagen sobre texto
#- document = 'EP TRANSPORTE ESPECIAL.PDF' # imagen de mierda con OCR
#R document = 'PLIEGO DE CONDICIONES DEFINITIVOS PR. 0144-18.pdf'
# - document = 'ESTUDIOS PREVIOS 4 DIAS-1.pdf' # puras tablas
# - document = 'ESTUDIO Def initivo 2018-01-009866.pdf' # Regular etiqueras cortas sin contexto
# - document = 'Estudio Previo 900_1.PDF' # Imagen escaneada

for item,indices in documentsSummaryItemIndexRanges[document].items():
    for indice in indices:
        print(item,indice,indice[0]-indice[1], [k for k,v in characterMap[document].items() if ((v[0]<indice[0]) and (v[1]>indice[0])) or ((v[0]<indice[1]) and (v[1]>indice[1]))])

'''