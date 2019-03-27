
# Author: Jonathan Espinosa
# Description: Methods to get text segments with a regular pattern from text data in pdf format.
# Enviroment name: envLB (conda)
# Python 3.6
# pip install PyPDF2, pdfminer.six, pandas, num2words, -U spacy, nltk, scikit-learn, matplotlib==2.1.2
# download and install spanish models: python -m spacy download es_core_news_md


############################################ Read input #######################################
import re
import os
import PyPDF2
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
import pandas as pd
from collections import Counter
import os
import fnmatch
import unicodedata
import re
from num2words import num2words
import spacy
import glob
from nltk import ngrams
import nltk
from utils import getTime, loggerMng
########################################### Normalizer ##########################
def textNormalize(text):
    '''
    This function apply a text normalizarion getting only valid characters. The valid characters are defined in the regex function.
    :param text: Text is a string chain to proccess
    :return: It return the same text formated in a normal shape
    '''
    trans_tab = str.maketrans({chr(key): None for key in range(768, 879)})
    del trans_tab[771]
    text = text.lower()
    # Normalization deleting diacritics
    nfkd = unicodedata.normalize('NFKD', text).translate(trans_tab)
    # To delete '~' from all letters, e.g. 'õ', except from the 'ñ'
    # https://es.stackoverflow.com/questions/135707/c%C3%B3mo-puedo-reemplazar-las-letras-con-tildes-por-las-mismas-sin-tilde-pero-no-l
    new_text = re.sub(r'([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+', r'\1', nfkd, 0, re.I)
    # Join both characters again (n and '~')
    new_text = unicodedata.normalize('NFKC', new_text)
    regex = '[^a-zA-Z0-9 ñÑ %]+'
    # Return only the allowed chars defined by the regex
    new_text = re.sub(regex, '', new_text)
    # Delete double whitespaces, if any
    new_text = re.sub('\s\s+', ' ', new_text)

    return new_text

############################################ pdminer ###########################################

def extract_text_by_page(pdf_path):
    '''
    This fuction read a PDF document page by page using pdfmine library
    :param pdf_path: directory that contains a pdf file
    :return: It return a iterator, to read page by page the data
    '''
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle)
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)

            text = fake_file_handle.getvalue()
            yield text

            # close open handles
            converter.close()
            fake_file_handle.close()


def extract_text(pdf_path):
    '''
    This function build the data flow to read all the page in a pdf with pdfMiner using a iterable element
    :param pdf_path: directory that contains a pdf file
    :return: The function return a python dicctionary with key = page Number and value = text in the 'key' page
    '''
    pages={}
    for n,page in enumerate(extract_text_by_page(pdf_path)):
        pageText = re.sub("\s\s+", " ", page) # re.sub("\s\s+", " ", page.replace('\n', ' '))
        pageText = textNormalize(pageText)
        pages[n]={'text':pageText,'len':len(pageText.split())}
    return pages

######################################### pypdf2 ##############################################


def extractTextByPage(pdf_path):
    '''
    This function allow to get data in text formar from a pdf file, getting text page by page. Result is storage in a python dict.
    :param pdf_path: directory that contains a pdf file
    :return: Python dict with pdf text, each page have a key, and text is the value. Len data is storage to compare multiple pdf libraries
    '''
    pdfFileObj = open(pdf_path, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    pagesLen = pdfReader.numPages
    #@todo lectura con dos librerias pdf comparar pagina a pagina
    #ojo son simbolos pueden ser pista de extraer formato ítems
    corpus =[]
    pages = {}
    lenLastPage = 0
    for page in range(0,pagesLen):
        objPage = pdfReader.getPage(page)
        pageText = re.sub("\s\s+" , " ", objPage.extractText().replace('\n',' ')) # re.sub("\s\s+" , " ", objPage.extractText().replace('\n',' '))
        pageText = textNormalize(pageText)
        pages[page] = {'text':pageText,'len':len(pageText.split())}#+lenLastPage
        #lenLastPage = len(pageText.split())
        #corpus.extend([pageText])
    return pages



def readPdfPages(pdf_path):
    '''
    This function build a new text dictionary that involves as much data as is posible, comparing page by page from diferent sources.
    :param pdf_path: pdf file directory
    :return: Python dictionari key: page Numbre, Value: Text in the 'key' page
    '''
    pagesA = extractTextByPage(pdf_path)
    pagesB = extract_text(os.path.join(pdf_path))
    keysPageA = pagesA.keys(); keysPageB = pagesB.keys()
    maxPagesDetected = max(len(keysPageA),len(keysPageB))

    pages = {}
    for page in range(0,maxPagesDetected):
        if (page in pagesA) and (page in pagesB):
            pages[page] = (pagesA[page],pagesB[page])[(pagesA[page]['len'],pagesB[page]['len']).index(max(pagesA[page]['len'],pagesB[page]['len']))]

            #print((pagesA[page]['len'],pagesB[page]['len']).index(max(pagesA[page]['len'],pagesB[page]['len'])))
            #print((pagesA[page]['len'],pagesB[page]['len']))
        else:
            if (page not in pagesA) and (page not in pagesB):
                print('Error 1 lectura en Página:', page)
                pages[page] = 'None'
            elif (page not in pagesA):
                print('Error 1A lectura en Página:', page)
                pages[page] = pagesB[page]
            elif (page not in pagesB):
                print('Error 1B lectura en Página:', page)
                pages[page] = pagesA[page]
            else:
                print('Error 2 lectura en Página:', page)
                pages[page] = 'None'

        pageCorpus = re.sub('\s\s+', ' ', pages[page]['text'])
        tokens = nltk.word_tokenize(pageCorpus)
        pageCorpus = ' '.join(tokens)
        pages[page]['text'] = ' ' + pageCorpus + ' '

    return pages


def getTextCorpus(pages):
    '''
    This method get the text from a python dicctionary in a string format. Combining all the pages in the same string
    :param pages: python dicctionary defined as: ... key: page Numbre, Value: Text in the 'key' page
    :return: String with all the pdf text
    '''
    pageCorpus = ''
    for page in pages.values():
        pageCorpus+=page['text']
    return pageCorpus

def recoverPageNumber(pages):
    '''
    It was developed in order to recover the page given a character index
    :param pages: python dict key:page Number, value: page text
    :return: python dict with key:page numbrer, value: range characters
    '''
    pageRanges = {}
    countInf=0
    countSup=0
    for pageNumber,pageData in pages.items():
        countSup += pageData['len']
        if pageNumber>0:
            countInf += pages[pageNumber-1]['len']
            pageRanges[pageNumber] = (countInf+1, countSup) # plus 1 to take close interval
        else:
            pageRanges[pageNumber] = (countInf,countSup)
    return pageRanges

def getPotencialItems(text):
    '''
    Given a item features from a previous analysis is posible to capture text segments with similar features by each ítem in a document. The analysis must to include words, size, kind of numbres
    :param text: String that contain all the text of a procesed pdf file
    :return: by each item it return the text index where the searched items can appear.
    '''
    #getNumbersIndex=[]
    #getPorcentageIndex=[]
    #getKeyWordIndex =[]
    #getMoneyIndex=[]
    #getTimeIndex=[]

    #Define key vocabulary to get text cursors
    #@todo usar palabras como ' dia ' para evitar strings internos como en inmediatamente.... igual con numeros..
    # algunas palabras pueden estar contenidas dentro de otros str ej: 'wwwcontratosgovco'
    moneyDict = {'unidades' : [' millones ',' miles '], 'siglas': [' smlv ',' cop ',' smmlv '],'moneda':[' pesos ',' dolares ',' euros ']}
    tiempoDict = {'periodo':[' dia ',' mes ',' año ',' semestre ',' trimestre ',' bimestre ',' dias ',' meses ',' años ',' semestres ',' trimestres ',' bimestres ']}
    numerosDict = {'numeros':[num2words(i, lang= 'es') for i in range(0,20)]}

    keyFrecWord = { 'EXPERIENCIA DEL PROPONENTE' :  [ ' valor ' ,  ' objeto ' , ' proponente ' , ' contrato ' , ' contratos ' , ' experiencia ' ],
                 'GARANTIA DE SERIEDAD DE LA OFERTA' : [ ' oferta ' ,  ' seriedad ' ,  ' garantia ' ],
                 'LIQUIDEZ' :[ ' activo ' , ' pasivo ' , ' liquidez ' , ' corriente ' ],
                 'NIVEL DE ENDEUDAMIENTO' : [ ' pasivo ' ,  ' activo ' ,  ' total ' ,  ' endeudamiento ' ],
                 'RAZÓN DE COBERTURA DE INTERESES' : [ ' utilidad ' , ' operacional ' , ' gastos ' , ' cobertura ' , ' intereses ' ],
                 'RENTABILIDAD SOBRE ACTIVOS' : [ ' operacional ' ,  ' total ' , ' rentabilidad ' , ' activo ' ],
                 'CUMPLIMIENTO, MULTAS Y CLAUSULA PENAL PECUNIARIA' :  [ ' cumplimiento ' ,  ' incumplimiento ' , ' clausula ' , ' penal ' , ' pecuniaria ' , ' plazo ' , ' meses ' , ' contratista ' , ' valor ' , ' contrato ' ],
                 'CALIDAD DE LOS SERVICIOS PRESTADOS' : [ ' vigencia ' ,  ' plazo ' ,  ' meses ' ,  ' valor ' ,  ' servicio ' , ' contrato ' ],
                 'PAGO DE SALARIOS, PRESTACIONES SOCIALES E INDEMNIZACIONES LABORALES' : [ ' laborales ' , ' años ' ,  ' salarios ' , ' valor ' , ' indemnizaciones ' , ' prestaciones ' , ' sociales ' , ' contrato ' ],
                 'GARANTIA DE RESPONSABILIDAD CIVIL EXTRACONTRACTUAL' : [ ' responsabilidad ' , ' civil ' , ' valor ' , ' contrato ' , ' extracontractual ' , ' smmlv ' ],
                 'RENTABILIDAD SOBRE PATRIMONIO' : [ ' utilidad ' , ' operacional ' , ' rentabilidad ' , ' patrimonio ' ]
                }


    KeyFrecWord_Re = {}
    for k,v in keyFrecWord.items():
        KeyFrecWord_Re[k] = [re.finditer(word, text) for word in v]

    KeyFrecWord_Index = {}
    for k, v in KeyFrecWord_Re.items():
        KeyFrecWord_Index[k] = [objRe.span() for reIter in v for objRe in reIter]


    # #Fine Main regular expresion
    # moneyRe = [re.finditer(word,text) for words in moneyDict.values() for word in words]#indexNumericPattern = re.finditer(r'\d+',text)
    # tiempoRe = [re.finditer(word, text) for words in tiempoDict.values() for word in words]
    # numerosRe = [re.finditer(word, text) for words in numerosDict.values() for word in words]
    #
    # #Define main positions and get cursors tuples
    # moneyIndex = [objRe.span() for reIter in moneyRe for objRe in reIter]
    # tiempoIndex = [objRe.span() for reIter in tiempoRe for objRe in reIter]
    # numerosIndex = [objRe.span() for reIter in numerosRe for objRe in reIter]
    # #@todo leer de PDF analysis wordCount/ítem 17 feb 2019
    # #Buid index candidates by item (usion item data <len,page>)
    # itemData = {'EXPERIENCIA DEL PROPONENTE': {'index':moneyIndex, 'windowSize':120}, #  ('acreditar', 39), ('proceso', 40), ('fecha', 41), ('debera', 43), ('certificaciones', 44), ('valor', 47), ('objeto', 48), ('proponente', 53), ('contrato', 76), ('contratos', 86), ('experiencia', 99)]
    #             'GARANTIA DE SERIEDAD DE LA OFERTA':{'index':moneyIndex+numerosIndex, 'windowSize':35}, #  ('debera', 29), ('contrato', 29), ('oferta', 40), ('seriedad', 42), ('garantia', 73)]
    #             'LIQUIDEZ':{'index':tiempoIndex, 'windowSize':46},# ('activo', 31), ('pasivo', 31), ('liquidez', 42), ('corriente', 62)]
    #             'NIVEL DE ENDEUDAMIENTO': {'index': moneyIndex, 'windowSize': 44},#  ('pasivo', 19), ('activo', 24), ('total', 39), ('endeudamiento', 43)]
    #             'RAZÓN DE COBERTURA DE INTERESES': {'index': moneyIndex + numerosIndex, 'windowSize': 46}, # ('utilidad', 22), ('operacional', 24), ('gastos', 28), ('cobertura', 32), ('intereses', 66)]
    #             'RENTABILIDAD SOBRE ACTIVOS': {'index': tiempoIndex, 'windowSize': 44},#  ('operacional', 20), ('total', 21), ('rentabilidad', 25), ('activo', 51)]
    #             'CUMPLIMIENTO, MULTAS Y CLAUSULA PENAL PECUNIARIA': {'index': tiempoIndex, 'windowSize': 31}, # ('cumplimiento', 16), ('incumplimiento', 17), ('clausula', 18), ('penal', 18), ('pecuniaria', 18), ('plazo', 18), ('meses', 21), ('contratista', 25), ('valor', 40), ('contrato', 79)]
    #             'CALIDAD DE LOS SERVICIOS PRESTADOS': {'index': tiempoIndex, 'windowSize': 19}, # ('vigencia', 16), ('plazo', 17), ('meses', 20), ('valor', 23), ('servicio', 31), ('contrato', 43)]
    #             'PAGO DE SALARIOS, PRESTACIONES SOCIALES E INDEMNIZACIONES LABORALES': {'index': tiempoIndex, 'windowSize': 23},# ('laborales', 27), ('años', 28), ('3', 29), ('salarios', 30), ('valor', 30), ('indemnizaciones', 31), ('prestaciones', 32), ('sociales', 32), ('contrato', 54)]
    #             'GARANTIA DE RESPONSABILIDAD CIVIL EXTRACONTRACTUAL': {'index': tiempoIndex, 'windowSize': 35}, #, ('responsabilidad', 25), ('igual', 28), ('civil', 36), ('valor', 36), ('contrato', 36), ('extracontractual', 38), ('smmlv', 61)]
    #             'RENTABILIDAD SOBRE PATRIMONIO': {'index': tiempoIndex, 'windowSize': 38}#('utilidad', 20), ('operacional', 20), ('rentabilidad', 28), ('patrimonio', 58)]
    #             }
    #


    itemData = {'EXPERIENCIA DEL PROPONENTE': {'index':'', 'windowSize':120}, #  ('acreditar', 39), ('proceso', 40), ('fecha', 41), ('debera', 43), ('certificaciones', 44), ('valor', 47), ('objeto', 48), ('proponente', 53), ('contrato', 76), ('contratos', 86), ('experiencia', 99)]
                'GARANTIA DE SERIEDAD DE LA OFERTA':{'index':'', 'windowSize':35}, #  ('debera', 29), ('contrato', 29), ('oferta', 40), ('seriedad', 42), ('garantia', 73)]
                'LIQUIDEZ':{'index':'', 'windowSize':46},# ('activo', 31), ('pasivo', 31), ('liquidez', 42), ('corriente', 62)]
                'NIVEL DE ENDEUDAMIENTO': {'index': '', 'windowSize': 44},#  ('pasivo', 19), ('activo', 24), ('total', 39), ('endeudamiento', 43)]
                'RAZÓN DE COBERTURA DE INTERESES': {'index': '', 'windowSize': 46}, # ('utilidad', 22), ('operacional', 24), ('gastos', 28), ('cobertura', 32), ('intereses', 66)]
                'RENTABILIDAD SOBRE ACTIVOS': {'index': '', 'windowSize': 44},#  ('operacional', 20), ('total', 21), ('rentabilidad', 25), ('activo', 51)]
                'CUMPLIMIENTO, MULTAS Y CLAUSULA PENAL PECUNIARIA': {'index': '', 'windowSize': 31}, # ('cumplimiento', 16), ('incumplimiento', 17), ('clausula', 18), ('penal', 18), ('pecuniaria', 18), ('plazo', 18), ('meses', 21), ('contratista', 25), ('valor', 40), ('contrato', 79)]
                'CALIDAD DE LOS SERVICIOS PRESTADOS': {'index': '', 'windowSize': 19}, # ('vigencia', 16), ('plazo', 17), ('meses', 20), ('valor', 23), ('servicio', 31), ('contrato', 43)]
                'PAGO DE SALARIOS, PRESTACIONES SOCIALES E INDEMNIZACIONES LABORALES': {'index': '', 'windowSize': 23},# ('laborales', 27), ('años', 28), ('3', 29), ('salarios', 30), ('valor', 30), ('indemnizaciones', 31), ('prestaciones', 32), ('sociales', 32), ('contrato', 54)]
                'GARANTIA DE RESPONSABILIDAD CIVIL EXTRACONTRACTUAL': {'index': '', 'windowSize': 35}, #, ('responsabilidad', 25), ('igual', 28), ('civil', 36), ('valor', 36), ('contrato', 36), ('extracontractual', 38), ('smmlv', 61)]
                'RENTABILIDAD SOBRE PATRIMONIO': {'index': '', 'windowSize': 38}#('utilidad', 20), ('operacional', 20), ('rentabilidad', 28), ('patrimonio', 58)]
                }

    for key in itemData.keys():
        itemData[key]['index']= KeyFrecWord_Index[key]


    return itemData



def mapaCharacterToWord(pageCorpus):
    tokens = nltk.word_tokenize(pageCorpus)
    offSet = 0 # palabras consecutivas index repite, si es relevante poner condición if (ex Clara lo es)
    lastWord = ''
    mapa = dict()
    for n,token in enumerate(tokens):
        if lastWord == token:
            offSet = pageCorpus.find(' '+token+' ',offSet+len(lastWord))
        else:
            offSet = pageCorpus.find(' '+token+' ', offSet)
        lastWord = token
        #mapa[offSet]=n # character to word, otherwise ---> mapa[n]=lastWord wort to character
        for m,__ in enumerate(token):
            print(offSet+m,token)
            mapa[offSet+m] = n
    return mapa

def getTokensFromIndex(pageCorpus,potencialItems,item):
    '''
    Este metodo toma las posiciones de los Ngrams candidatos y extra la secuencia de texto en "tokens"
    :param pageCorpus:
    :param potencialItems:
    :param item:
    :return:
    '''
    listaIndicesTokens = []
    #texts = pageCorpus # ' estudios previos transporte pagina 1 de 5 version fecha de aprobacion codigo ....
    indexes = potencialItems[item]['index'] # [(2350, 2353), (2360, 2363), (5333, 5336)....
    window = potencialItems[item]['windowSize'] # 36
    #mapa character to word
    #mapa = [n for n, word in enumerate(texts.split()) for i in word + ' ']
    mapa = mapaCharacterToWord(pageCorpus)
    tokens=[]
    tokensCorpus=nltk.word_tokenize(pageCorpus)
    #21 feb voy aqui poner condicion window en extremos
    for index in indexes:
        indexInf = mapa[index[0]] - window
        indexSup = mapa[index[0]] + window
        # because Ngram method doenst work with text files witn len > n
        if indexInf < 0:
            indexInf = 0
            indexSup = window
        lenghtCorpus = len(tokensCorpus)
        if indexSup >= lenghtCorpus:
            indexInf = lenghtCorpus - window
            indexSup = lenghtCorpus

        #indexInf = indexInf if indexInf > 0 else 0
        #indexSup = indexSup if indexSup <  len(pageCorpus.split()) else len(pageCorpus.split())- window
        windowSentence = tokensCorpus[indexInf : indexSup]
        #stringSentence = pageCorpus[indexInf : indexSup]
        #windowSentence = nltk.word_tokenize(stringSentence)
        if bool(re.search(r'\d'," ".join(windowSentence))):  # Reject Ngrams without numbers
            nGramWindow = list(ngrams(windowSentence, window))
            tokens.extend(nGramWindow)
            listaIndicesTokens.append((indexInf,indexSup))
        # feb19 poner solo num a items y generar tabla de items candidatos por cada indice. Diccionario por index en lugar items
    listaTokens = [" ".join(toke) for toke in tokens]


    return listaTokens, listaIndicesTokens


def checkNumbers(text):
    return bool(re.search(r'\d', text))

def textPreprocesing(text,startIndex, endIndex, ventana):
    '''
    Apply a window in the index found above, to get the text context by each cursor by each item
    :param text:  String that contain all the text of a procesed pdf file
    :param startIndex: Cursor as item candidate start index
    :param endIndex: Cursor as item candidate end index
    :param ventana: Size of text window item
    :return: Python dict with text of each item candidate
    '''
    #startIndex = 34074; endIndex= 34082
    #ventana = 50
    # @ todo 17 feb 1. Realizar el barrido de todos lo indices no solo el primero. 2. HAcerlo sobre 50 palabras no caracteres, si no tomar numero caracteres.
    #okensItem[item] = textPreprocesing(pageCorpus,potencialItems[item]['index'][0][0] , potencialItems[item]['index'][0][1],potencialItems[item]['windowSize']) #textPreprocesing(text,startIndex, endIndex, ventana)
    #above exaple to get the first token from the first index in a item
    if startIndex<ventana:
        startDelta = ventana
    elif endIndex> len(text)-ventana:
        endIndex = len(text)- ventana

    frase = nlp(text[startIndex-ventana:endIndex+ventana])
    tokens = []
    for token in frase:
        tokens.append(token.text)

    return tokens

#def getTextNgramsCandidates
#def cleanText
#add try and catch
def readSeveralDocuments(inPath):
    '''

    :param inPath:
    :return: A python dictionary where each item has their own key and value has the item candidates text segments (it will be analized by a text clasificator in next step). Othe dicctionary storage the same result by each pdf file, using pdf name as key
    '''
    #inPath ='./inputFiles/'
    logger = loggerMng()
    ratioMaximoPaginasNullPermitidas = 0.5
    characterMap = {}
    nlp = spacy.load('es_core_news_md') # get Spacy spanish posTag, three dependencies and lemma
    tokensDocumentDict = {}
    indexTokensDocumentDict ={}
    licitaciones = glob.glob(os.path.join(inPath,'*.pdf'))#os.listdir(inPath)
    for licitacion in licitaciones:
        try:
            tokensItem = {}
            indexTokensItem= {}
            #pdf_path = os.path.join(inPath,licitacion)
            licitacionName = licitacion.split('/')[-1]  # to get only fileName
            pages = readPdfPages(licitacion) # read pdf text (clean & normalized) ---> Lectura de PDFs
            if sum([1 for page in pages.values() if page['len']==0]) > len(pages.keys())*ratioMaximoPaginasNullPermitidas:
                raise Warning('Documento %s Vacio' % licitacion)
                #break
            else:
                characterMap[licitacionName]=recoverPageNumber(pages) # to get characters page index
                pageCorpus =  getTextCorpus(pages) # get a unique text corpus from PDF (without pages index)
                potencialItems = getPotencialItems(pageCorpus) # to get character index if it contains keywords from item dictionaries
                for item in potencialItems.keys():
                    #tokensItem[item] = textPreprocesing(pageCorpus,potencialItems[item]['index'][0][0] , potencialItems[item]['index'][0][1],potencialItems[item]['windowSize']) #textPreprocesing(text,startIndex, endIndex, ventana)
                    tokensItem[item], indexTokensItem[item] = getTokensFromIndex(pageCorpus, potencialItems, item)
                tokensDocumentDict[licitacionName] = tokensItem
                indexTokensDocumentDict[licitacionName]=indexTokensItem

        except Warning as w:
            print('warning tipo por %s en documento' % w)
            logger.exception(w)
        except ValueError as e:
            print('error tipo ValueError por %s en documento' % e)
            logger.error(e)
        except Exception as e:
            print('error %s de tipo %s' % (e, type(e)))
            logger.error(e)

    return tokensDocumentDict,characterMap,indexTokensDocumentDict

def intexToPage(indexMap,index):
    '''

    :param indexMap: map of len characters by page
    :param index: index of character to check
    :return: page where de index is
    '''
    for key,value in indexMap.items():
        if value[0]< index < value[1]:
            page = key; print(key)
    return page




if __name__ == "__main__":
    inPath = '/home/espinosa/Escritorio/licitaBot/src/inputFiles'
    tokensDocumentDict, characterMap, indexTokensDocumentDict = readSeveralDocuments(inPath)

# 20 marzo voy aqui tryCatch con log y sale!

# @todo 2 marzo:
# aplicar mismo tokemizer y page corpus para extraer el %
# Sistema de votacion por clasificacion
# Filtrado si no hay numero
# Poner KPI para saber si se identifica en misma pagina el item





