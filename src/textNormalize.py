
import os
import codecs
from data_manipulation import pred_utils
import re

# 1) .replace()/.translate() of this dictionary
# With first and last whitespaces to avoid join first or last words with other possible words
# (except 'll' which should be deleted everywhere). If these keys substring


#cualquier palabra hay que ponerla con un espacio a cada lado
#si la clave del diccionario es compuesta hay que separar las palabras por espacio doble
#si la palabra acaba en y se acaba sustituyendo por i latina


_itemsToReplaceDict = {
    'll': 'y',
    ' tp  link ': ' tepelink ',
    ' aig  europe  limited ': ' ai europ limite ',
    ' cade  box ': ' cadebos ',
    #' giga  herzios ': ' jiga ercios ',
    ' giga ': 'jiga', #clara ya lo pone como j el sonido
    ' gigas ': ' jigas ',
    ' bq  aquaris ': ' becu acuaris ',
    #' smart  tv ': ' esmartivi ',
    #' smar  tv ': 'esmartivi',
    #' smar  tivi': 'esmartivi',
    ' tv ': ' tivi ',
    ' smart': 'esmar,', #creo que esto es importante, no considerar smart + nombre si no es tv
    ' pack  futbol ': ' pak fubol ',
    ' bein  sports ': ' beinespor ',
    ' zte  blade ': ' zetate bleid ',
    ' play  station ': ' plei esteishon ',
    #' smart  watch ': ' smarwach ',
    ' watch ': ' wach ',
    ' wash ': ' wach ',
    ' go  play ': ' gou plei ',
    ' por  ciento ': ' porciento ',
    ' pepe  phone ': ' pepefon ',
    ' phone ': 'fon', #importante debajo de la anterior
    ' stok ': ' estok ',
    ' orange ': ' oranch ',
    ' oran ': ' oranch ',
    ' cannon ': ' canon ',
    ' tablet ': ' table ',
    ' mediapad ': 'mediapa',
    ' amperios ': 'anperios',
    ' pantaia ': 'pantaya'
}

_itemsToReplaceDictExpansionOrange = {
    ' cuatro  ge ': ' cuatroje ',
    ' cuatro  g ': ' cuatroje ',
    ' cuatro  je ': ' cuatroje ',
    ' 4g ': ' cuatroje ',
    ' u  dos ': ' udos ',
    ' u2 ': ' udos ',
    #' equis  ele ': ' equisele ',
    #' xl ': ' equisele ',
    ' quis ': ' equis ',
    ' x ': ' equis ',
    ' ele  uno ': ' eleuno ',
    ' l  uno ': ' eleuno ',
    ' l1 ': ' eleuno ',
    #' equis  a ': ' equisa ',
    #' xa ': ' equisa ',
    #' equis  zeta ': ' equiszeta ',
    #' xz ': ' equiszeta ',
    ' a  tres ': 'atres',
    ' ge  cinco ': ' gecinco ',
    ' g  cinco ': ' gecinco ',
    ' g5 ': ' gecinco ',
    ' e  cuatro ': ' ecuatro ',
    ' e4 ': ' ecuatro ',
    ' ca  cuatro ': ' cacuatro ',
    ' ka  cuatro ': ' cacuatro ',
    ' k  cuatro ': ' cacuatro ',
    ' k4 ': ' cacuatro ',
    ' ca  ocho ': ' caocho ',
    ' k  ocho ': ' caocho ',
    ' k8 ': ' caocho ',
    ' ca  diez ': ' cadiez ',
    ' k  diez ': ' cadiez ',
    ' k10 ': ' cadiez ',
    ' cu  seis ': ' cuseis ',
    ' q  seis ': ' cuseis ',
    ' q6 ': ' cuseis ',
    # ' a  cinco ': '  ',  # 'a seis, a...N': too much risk, very frequent in other places
    ' pe  veinte ': ' peveinte ',
    ' p  veinte ': ' peviente ',
    ' p20 ': ' peveinte ',
    ' pe  ocho ': ' peocho ',
    ' p  ocho ': ' peocho ',
    ' p8 ': ' peocho ',
    ' pe  nueve ': ' penueve ',
    ' p  nueve ': ' penueve ',
    ' p9 ': ' penueve ',
    ' pe  diez ': ' pediez ',
    ' p  diez ': ' pediez ',
    ' p10 ': ' pediez ',
    #' pe  smart ': ' pesmar ',
    #' p  smart ': ' pesmar ',
    ' pesmar ': ' pe esmar ',
    ' i  griega ': ' igriega ',
    ' y3 ': ' igriega tres ',
    ' y4 ': ' igriega cuatro ',
    ' y5 ': ' igriega cinco ',
    ' y6 ': ' igriega seis ',
    ' y7 ': ' igriega siete ',
    ' i  siete ': ' isiete ',
    ' seis  ese ': ' seisese ',
    ' seis  s ': ' seisese ',
    ' 6s ': ' seisese ',
    ' cuatro  ese ': ' cuatroese ',
    ' cuatro  s ': ' cuatroese ',
    ' 4s ': ' cuatroese ',
    ' cinco  ese ': ' cincoese ',
    ' cinco  s ': ' cincoese ',
    ' 5s ': ' cincoese ',
    #' iphone  se ': ' aifon esee ',
    #' iphone  x ': ' aifon equis ',
    ' iphone ': 'aifon',
    ' a  cinco ': 'acinco',
    ' jota  cinco ': ' jotacinco ',
    ' j  cinco ': ' jotacinco ',
    ' j5 ': ' jotacinco ',
    ' jota  tres ': ' jotatres ',
    ' j  tres ': ' jotatres ',
    ' j3 ': ' jotatres ',
    ' jota  siete ': ' jotasiete ',
    ' j  siete ': ' jotasiete ',
    ' j7 ': ' jotasiete ',
    ' ese  tres ': ' esetres ',
    ' s  tres ': ' esetres ',
    ' s3 ': ' esetres ',
    ' ese  seis ': ' eseseis ',
    ' s  seis ': ' eseseis ',
    ' s6 ': ' eseseis ',
    ' ese  siete ': ' esesiete ',
    ' s  siete ': ' esesiete ',
    ' s7 ': ' esesiete ',
    ' ese  ocho ': ' eseocho ',
    ' s  ocho ': ' eseocho ',
    ' s8 ': ' eseocho ',
    ' ese  nueve ': ' esenueve ',
    ' s  nueve ': ' esenueve ',
    ' s9 ': ' esenueve ',
    ' active ': ' actif ',
    ' ache  erre ': ' acheerre ',
    ' hr ': ' acheerre ',
    #' smart  band ': ' smarban ',
    #' smartband ': ' smarban ',
    ' forerunner ': ' forerraner ',
    ' airbox ': ' erbox ',
    ' te  tres ': ' tetres ',
    ' t  tres ': ' tetres ',
    ' t3 ': ' tetres ',
    ' snes ': ' esenes ',
    ' dos  de  ese ': ' dosdeese ',
    ' dos  ds ': ' dosdeese ',
    ' 2ds ': ' dosdeese ',
    ' pe  ese  cuatro ': ' pesecuatro ',
    ' ps  cuatro ': ' pesecuatro ',
    ' ps4 ': ' pesecuatro '
}

# 2) Substitute these words
_wordsToReplaceDict = {
    ' meflife ': ' metlaif ', #@todo: para trabajar con el item tratamiento_datos (pensar en otra solución)
    ' metlife ': ' metlaif ',
    ' europe ': ' europ ',
    ' samsung ': ' samsun ',
    ' cdp ': ' cedepe ',
    ' tplink ': ' tepelink ',
    ' wanadoo ': ' wanadu ',
    ' tdi ': ' tedei ',
    ' youtube ': ' yutube ',
    ' track ': ' trak ',
    ' twitter ': ' tuiter ',
    ' facebook ': ' feisbuk ',
    ' whatsapp ': ' wasap ',
    ' hitachi ': ' itachi ',
    ' bytes ': ' baits ',
    ' pc ': ' pece ',
    ' adsl ': ' adesele ',
    ' ok ': ' okei ',
    ' outlook ': ' outluk ',
    ' yahoo ': ' yaju ',
    ' netflix ': ' nefli ',
    ' xperia ': ' experia ',
    ' lg ': ' elege ',
    ' huawei ': ' juawei ',
    ' apple ': ' apel ',
    ' love ': ' lof ',
    ' holiday ': ' jolidei ',
    ' roaming ': ' romin ',
    #' megapixeles ': ' megapiseles ',
    ' gamer ': ' gueimer ',
    ' jazztel ': ' yastel ',
    ' online ': ' onlain ',
    ' router ': ' ruter ',
    ' sms ': ' eseemeese ',
    ' lopd ': ' eleopede ',
    ' email ': ' imeil ',
    ' finance ': ' fainans ',
    ' gmail ': ' jemeil ',
    ' hotmail ': ' jotmeil ',
    ' dni ': ' denei ',
    ' de  ene  i ': 'denei',
    ' hbo ': ' achebeo ',
    ' bluetooth ': ' blutuz ',
    ' xiaomi ': ' xiomi ',
    ' play ': ' plei ',
    ' galaxy ': ' galaxi ',
    ' vodafone ': ' vodafon ',
    ' champions ': ' champion ',
    ' hd ': ' achede ',
    ' tv ': ' tivi ',
    ' ebook ': ' ibook ',
    ' x ': ' equis ',
    ' equisa ': ' equis  a',
    ' google ': ' guguel ',
    ' bq ': ' becu ',
    ' nike ': ' naik ',
    ' ipad ': ' aipad ',
    ' one ': ' guan ',
    ' hauk ': ' jauk ',
    ' drone ': ' dron ',
    ' chromecast ': ' croumcas ',
    ' schneider ': ' esneider ',
    ' kindle ': ' kindel ',
    ' paperwhite ': ' peiperguait '
}

# Add '_itemsToReplaceDictExpansionOrange' words to the '_itemsToReplaceDict' dictionary
_itemsToReplaceDict.update(_itemsToReplaceDictExpansionOrange)
_itemsToReplaceDict.update(_wordsToReplaceDict )

def normalize_clara_words(text):

    """
    Normalize the 'text' also replacing 'words' and 'items' with the above dictionaries for Clara.

    :param text: Text to normalize.
    :return: Normalized text.
    """

    # Introduce a leading and trailing whitespaces to always match with '_itemsToReplaceDict' dictionary items if any of
    # the items are just at the beginning or ending of 'text'.
    text = ' ' + text + ' '

    # Call to general normalize text from 'pred_utils'
    #textNorm = pred_utils.normalize_text(text, onlyLetters=True)
    textNorm = pred_utils.normalize_text(text)

    # Replaces items (multiple words including leading and trailing whitespaces are never going to be contained in individual words),
    # therefore we can use replace. Additionally, we also replace 'll' since it always must be replaced wherever it appears.

    textNorm = textNorm.replace(' ', '  ')

    clavesOrdenadas = sorted(_itemsToReplaceDict.keys())
    valoresOrdenados = [_itemsToReplaceDict[clave] for clave in clavesOrdenadas]

    #for k, v in _itemsToReplaceDict.items():
    for k, v in zip(clavesOrdenadas, valoresOrdenados):
        textNorm = textNorm.replace(k, v)
    textNorm = textNorm.replace('  ', ' ')

    textNorm= textNorm.replace('y ', 'i ')

    textNorm=textNorm[1:-1] #sólo si termina y empieza con espacio

    textNorm= pred_utils.normalize_text(textNorm, onlyLetters=True)

    return textNorm


def normalize_clara_words_folder(path, pattern, recursively=True):
    """
    Normalize all 'path' files following 'pattern' 'recursively' or not also replacing words and items
    with the above dictionaries.

    :param path: Folder to find the files to normalize.
    :param pattern: The pattern the files must match.
    :param recursively: Do it recursively (default) or not.
    :return: Nothing. The files are overwritten already normalized.
    """

    # Get all the files to normalize
    files = pred_utils.get_files(path, pattern, recursively)

    for file in files:
        # Ensure file is read as utf-8 (important)
        with codecs.open(os.path.join(path, file), 'r+', 'utf-8') as fio:
            # Read full file content
            fileContent = fio.read()
            fileContent = fileContent.replace('\n', ' ')
            fileContent = re.sub('\s\s+', ' ', fileContent)


            # Write to a file the final content, .seek(0) and .truncate() to overwrite the file
            fio.seek(0)
            fio.write(normalize_clara_words(fileContent))
            fio.truncate()
