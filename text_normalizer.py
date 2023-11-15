import spacy
from num2words import num2words
import re
 

class WhiteSpaceTokenizer():
    PUNCTUATION = "?:!.,;"
    CUSTOMS_EXCLUDE = ["sound", 'music', "song","recording","features", 'feature', 'record', 'audio', 'contain', "play", 'consist' ]
    SPECIAL_TERM = {'\"4 on the floor\"': "spe_floor"}
    def __init__(self,language: str = "en_core_web_sm", lemma: bool = True, stopwords: bool = True) -> None:
        self.stopwords = stopwords
        self.lemma = lemma

        try:
            self.nlp = spacy.load(language)
        except IOError:
            spacy.cli.download(language)  # type: ignore
            self.nlp = spacy.load(language)

    def __call__(self, text, return_text: bool = False):
        for term in self.SPECIAL_TERM:
          text = text.replace(term, self.SPECIAL_TERM[term])
        text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)  # type: ignore
        # normalize text
        text = self.nlp(text)  # type: ignore
        # remove stopwords
        if self.stopwords:
            text = [w for w in text if not w.is_stop]  # type: ignore
        # remove punctuation
        text = [w for w in text if w.text not in self.PUNCTUATION]  # type: ignore
        
        text = [getattr(t, "lemma_" if self.lemma else "text") for t in text]
        # text = [w for w in text if w not in self.CUSTOMS_EXCLUDE] 

        return " ".join(text)