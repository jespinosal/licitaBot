# licitaBot
Text sentence level classificator to extract contract information from public tenders in .pdf source. It is a text clasification system that includes all the flow, sice data extraction (data in pdf format) to text clasification and EER.

Librearies:

Code description:

readInput: this code includes the following components:

    Methods to read pdf text through an assembly that involves the pdfMiner and pyPdf2 libraries.
    Text Normalizer: This adjusts the text format in a universe of caraceres normalized according to the UTF-8 standard.
    Analyzer regular expressions: Contains methods that look for cursors in the text to identify numeric patterns that reference dates, monetary values, economic and financial indicators, quotas, deadlines, etc.
    Segment extractor: This filters segments of text that can potentially be the searched items, according to the regular expressions above.
    Tokenized: Each candidate segment is pre-processed to generate tokens, which represent the segment in question. Taking into account punctuation and spelling of the Castilian language.
    Construction of dictionaries: For each file in pdf format of the reference path, a dictionary containing the candidate segments for each item is generated. In the end this is stored in a document dictionary, where each document is the key.

