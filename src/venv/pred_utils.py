
import os
import fnmatch
import unicodedata
import re


def get_files(path, pattern, recursively=True):
    """
    Get all files matching `pattern` from `path`, being recursively by default.

    :param path: The path to read files from.
    :param pattern: The pattern the files must match. Patterns are Unix shell style.
                    Check this url for more info: https://docs.python.org/3.6/library/fnmatch.html
    :param recursively: Do it recursively (default) or not.
    :return: The list of matching files.
    """
    matches = []

    if recursively:
        for root, dirs, files in os.walk(path):
            for filename in fnmatch.filter(files, pattern):
                matches.append(os.path.join(root, filename))
    else:
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, pattern):
                matches.append(os.path.join(path, file))

    return matches


def normalize_text(text, onlyLetters=False, lower=True, includeNTilde=True):
    """
    Normalize text deleting diacritics and following the parameter options.

    :param text: Text to normalize.
    :param onlyLetters: Allow only letters (`True`) or letters and numbers (`False`: default).
    :param lower: Lower case (`True`: default).
    :param includeNTilde: Include (do not delete) `ñ` character (default).
    :return: Normalized text without diacritics following the parameter options.

    >>> text = 'Aõs3á2{ vùñD"s k!Ñ+• Agentes. • Campañas. ◦ Items: ▪ Reglas. ▪  Parámetros.Items: ▪ \t\tReglas. ▪ \tParámetros.'
    >>> normalize_text(text)
    'aos3a2 vuñds kñ agentes campañas items reglas parametrositems reglas parametros'
    >>> normalize_text(text, onlyLetters=True)
    'aosa vuñds kñ agentes campañas items reglas parametrositems reglas parametros'
    >>> normalize_text(text, lower=False)
    'Aos3a2 vuñDs kÑ Agentes Campañas Items Reglas ParametrosItems Reglas Parametros'
    >>> normalize_text(text, onlyLetters=True, includeNTilde=False)
    'aosa vunds kn agentes campanas items reglas parametrositems reglas parametros'
    """

    # Example text to test
    # text = 'Aõs3á2{ vùñD"s k!Ñ+'
    # extended test with special chars, double whitespace, tabs, etc.
    # text = 'Aõs3á2{ vùñD"s k!Ñ+• Agentes. • Campañas. ◦ Items: ▪ Reglas. ▪ Parámetros.Items: ▪ \t\tReglas. ▪ \tParámetros.'

    # Check maketrans: 1st: unicode map, 2nd (&1st): strings=length, 3rd: string to None
    # Combining diacritical marks: \u0300-\u036F (768 to 879) -> https://en.wikipedia.org/wiki/Combining_Diacritical_Marks
    trans_tab = str.maketrans({chr(key): None for key in range(768, 879)})

    # If 'includeNTilde', do not delete '~' which is 771 (\u0303)
    if includeNTilde:
        del trans_tab[771]

    # Lower is needed?
    if lower:
        text = text.lower()

    # Normalization deleting diacritics
    nfkd = unicodedata.normalize('NFKD', text).translate(trans_tab)

    # To delete '~' from all letters, e.g. 'õ', except from the 'ñ'
    # https://es.stackoverflow.com/questions/135707/c%C3%B3mo-puedo-reemplazar-las-letras-con-tildes-por-las-mismas-sin-tilde-pero-no-l
    new_text = re.sub(r'([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+', r'\1', nfkd, 0, re.I)
    # Join both characters again (n and '~')
    new_text = unicodedata.normalize('NFKC', new_text)

    # Select the regex based on the parameters
    if onlyLetters:
        regex = '[^a-zA-Z ñÑ]+'
    else:
        regex = '[^a-zA-Z0-9 ñÑ]+'

    # Return only the allowed chars defined by the regex
    new_text = re.sub(regex, '', new_text)
    # Delete double whitespaces, if any
    new_text = re.sub('\s\s+', ' ', new_text)

    return new_text


# Do not executed on imports
if __name__ == "__main__":
    import doctest
    # For each modification of this 'pred_utils.py' run the following line to test everything works as expected
    doctest.testmod()
