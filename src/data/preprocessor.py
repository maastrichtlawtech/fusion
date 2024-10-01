import re
import spacy
from tqdm import tqdm


class TextPreprocessor:
    """ 
    Preprocess text data using spaCy.

    :param spacy_model: The spaCy model to use for pre-processing.
    """
    def __init__(self, spacy_model):
        self.nlp = spacy.load(spacy_model)

    def preprocess(
        self, 
        texts: list[str], 
        lowercase: bool = True, 
        remove_punct: bool = True, 
        remove_num: bool = True, 
        remove_stop: bool = True, 
        lemmatize: bool = True,
    ) -> list[str]:
        """
        Preprocess a list of texts.

        :param texts: A list of texts to pre-process.
        :param lowercase: Whether or not to lowercase the text.
        :param remove_punct: Whether or not to remove punctuation.
        :param remove_num: Whether or not to remove numbers.
        :param remove_stop: Whether or not to remove stopwords.
        :param lemmatize: Whether or not to lemmatize the text.
        :returns: A list of pre-processed texts.
        """
        processed_texts = []
        for doc in tqdm(self.nlp.pipe(texts, n_process=-1), desc='Pre-processing', total=len(texts)):
            out = self.preprocess_document(doc, lowercase, remove_punct, remove_num, remove_stop, lemmatize)
            processed_texts.append(out)
        return processed_texts

    def preprocess_document(
        self, 
        doc: spacy.tokens.Doc,
        lowercase: bool, 
        remove_punct: bool, 
        remove_num: bool, 
        remove_stop: bool, 
        lemmatize: bool,
    ) -> str:
        """
        Preprocess a single document.

        :param doc: A spaCy Doc object.
        :param lowercase: Whether or not to lowercase the text.
        :param remove_punct: Whether or not to remove punctuation.
        :param remove_num: Whether or not to remove numbers.
        :param remove_stop: Whether or not to remove stopwords.
        :param lemmatize: Whether or not to lemmatize the text.
        :returns: A pre-processed text.
        """
        tokens = []
        for token in doc:
            if remove_punct and token.is_punct:
                continue
            if remove_num and (token.is_digit or token.like_num or re.match('.*\d+', token.text)):
                continue
            if remove_stop and token.is_stop:
                continue
            if lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
        text = ' '.join(tokens)
        if lowercase:
            text = text.lower()
        return text
