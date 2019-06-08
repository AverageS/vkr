"""Определение авторства публикации в онлайновой социальной сети

Модуль предоставляет интерфейс для определения авторства на основе текстовых публикаций данных из социальных сетей

Ситников Алексей
факультет Информатики и систем управления
кафедра Информационной безопасности
2019 г
Москва
"""

import typing as tp
import requests_html as requests
import html2text
import numpy as np
import nltk
import pickle

from pathlib import Path
from zope.cachedescriptors.property import CachedProperty
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.vq import whiten


CWD = Path(__file__).parent

word_stats = [('в', 320765), ('и', 281835), ('не', 187358), ('на', 161125), ('что', 129043), ('с', 100797),
              ('то', 73719), ('по', 65542), ('это', 64167), ('а', 62023), ('как', 57721), ('из', 49726), ('я', 48545),
              ('у', 44171), ('А', 43597), ('за', 41734), ('В', 38693), ('com', 38278), ('все', 36123), ('так', 36015),
              ('livejournal', 34938), ('но', 33760), ('к', 33724), ('о', 32455), ('от', 31827), ('https', 31569),
              ('вы', 30499), ('для', 30239), ('или', 29408), ('И', 29219), ('же', 29088), ('бы', 27160),
              ('можно', 26947), ('html', 26048), ('http', 26008), ('его', 25562), ('он', 25491), ('только', 24199),
              ('уже', 23411), ('есть', 22955), ('UTC', 22747), ('блога', 22448), ('Я', 21670), ('будет', 20655),
              ('было', 19571), ('ru', 19386), ('еще', 19043), ('если', 19006), ('всё', 18897), ('России', 18888)]
"""Статистика словоупотреблений в livejounral

Список кортежей содержащих слово и число словоупотрелений на миллион словоупотреблений во всем корпусе
"""

pos_stats = [('S', 34375), ('V', 14951), ('PR', 11257), ('A', 8441), ('CONJ', 7993), ('S-PRO', 7065), ('A-PRO', 5174),
             ('PART', 4997), ('NUM', 4349), ('ADV', 4294)]
"""Статиска использований различных частей речи в livejournal

Список кортежей содержащих часть речи и число употреблений на миллион словоупотреблений во всем корпусе

S       : существительное
V       : глагол
PR      : предлог
A       : прилагательное
CONJ    : союз
S-PRO   : местоимение существительное
A-PRO   : местоимение прилагательное
PART    : частика
NUM     : числительное
ADV     : наречие
"""

nltk.data.path.append(CWD / 'nltk_data')


def download(url: str) -> str:
    """
    Загрузка и парсинг публикации

    Args:
        url: ссылка на публикацию (напр. https://exler.livejournal.com/2238228.html)

    Returns:
        текст публикации
    """
    response = requests.HTMLSession().get(url)

    parsed_text = html2text.html2text(response.text, bodywidth=0)

    clear_paragraphs = []
    for line in parsed_text.splitlines():
        if not line.strip():
            continue

        if line.strip()[0] in ('*', '['):
            continue

        rus_symbols = len([x for x in line if 1072 <= ord(x.lower()) <= 1105])
        total_symbols = len([x for x in line if x.isalpha()])

        try:
            rus_proportion = rus_symbols / total_symbols
        except ArithmeticError:
            rus_proportion = 0

        if rus_proportion < 0.4:
            continue

        clear_paragraphs.append(line.strip())

    return '\n'.join(clear_paragraphs)


class Publication:
    """
    Класс для обработки публикации и выделении свойств
    """
    def __init__(self, text, *,
                 sentence_tokenizer: nltk.tokenize.punkt.PunktSentenceTokenizer
                 = nltk.data.load('tokenizers/punkt/russian.pickle'),
                 word_tokenizer: nltk.tokenize.RegexpTokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')):
        """
        Args:
            text: анализируемый текст
            sentence_tokenizer: разделитель текста на предложения
            word_tokenizer: разделитель текста на слова
        """
        self.text = text

        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer

    @CachedProperty('text')
    def sentences(self):
        """
        Returns:
            Список предложений текста
        """
        return self._sentence_tokenizer.tokenize(self.text)

    @CachedProperty('text')
    def words(self):
        """
        Returns:
            Список слов текста
        """
        return self._word_tokenizer.tokenize(self.text)

    @CachedProperty('text')
    def tokens(self):
        """
        Returns:
            Список токенов (элеменов парсинга)
        """
        return nltk.word_tokenize(self.text.lower())

    @CachedProperty('text')
    def words_features(self):
        """
        Returns:
            Вектор употрблений частотных слов в тексте
        """
        vectorizer = CountVectorizer(vocabulary=[t[0] for t in word_stats][:16],
                                     tokenizer=self._word_tokenizer.tokenize)

        counter = (
            vectorizer
            .fit_transform([self.text])
            .toarray()
            .astype(np.float64)
        )

        normalized_counter = counter / np.c_[np.apply_along_axis(np.linalg.norm, 1, counter)]
        normalized_counter[np.isnan(normalized_counter)] = 0

        return normalized_counter

    @CachedProperty('text')
    def lexical_features(self):
        """
        Returns:
            Вектор лексический характеристик, таких как: доля уникальных слов, средняя длина предлоежния,
            сркв откл длины предложения
        """
        words_per_sentence = np.array([
            len(self._word_tokenizer.tokenize(sentence))
            for sentence in self.sentences
        ])

        diversity = len(set(self.words)) / len(self.words)
        mean = words_per_sentence.mean()
        variation = words_per_sentence.std()

        features = whiten(np.array([mean, variation, diversity], np.float64))

        return features

    @CachedProperty('text')
    def punctuation_features(self):
        """
        Returns:
            Вектор употребления знаков пунктуации
        """
        return whiten(np.array([
            self.tokens.count('.') / len(self.sentences),
            self.tokens.count(',') / len(self.sentences),
            self.tokens.count('!') / len(self.sentences),
            self.tokens.count('?') / len(self.sentences),
            self.tokens.count(';') / len(self.sentences),
            self.tokens.count(':') / len(self.sentences)
        ]))

    @CachedProperty('text')
    def syntactic_features(self):
        """
        Returns:
            Вектор употребления частей речи
        """
        posed_words = [t[1].split('=')[0] for t in nltk.pos_tag(self.words, lang='rus')]

        counter = np.array([posed_words.count(pos[0]) for pos in pos_stats], np.float64)

        normalized_counter = counter / np.c_[np.apply_along_axis(np.linalg.norm, 0, counter)]

        return normalized_counter

    def get_features(self):
        """
        Returns:
            Вектор свойств текста
        """
        return np.concatenate((
            self.syntactic_features,
            self.punctuation_features,
            self.words_features,
            self.lexical_features
        ), axis=None)


def parse_csv_row(s: str):
    s = s.split(',')
    y, f1, f2 = bool(int(s[0])), np.array(s[1:36]).astype(np.float64), np.array(s[36:]).astype(np.float64)

    return y, f1, f2


def read_obj_from_csv(dump_file: tp.TextIO):
    """
    Args:
        dump_file: файл в формате сsv

    Returns:
        список элементов вида (bool, vector, vector) содержащий соответственно метку о том, были ли написаны тексты
        одним автором, вектор свойств первого текста, вектор свойств второго текста
    """
    return [parse_csv_row(line) for line in dump_file.readlines()]


def concatenate(feature_vector_1: np.ndarray, feature_vector_2: np.ndarray) -> np.ndarray:
    return np.concatenate((feature_vector_1, feature_vector_2), axis=None)


with open(CWD / 'data' / 'sample.csv') as f:
    SAMPLE_DATA = read_obj_from_csv(f)
    SAMPLE_DATA = [
        [t[0] for t in SAMPLE_DATA],
        [t[1] for t in SAMPLE_DATA],
        [t[2] for t in SAMPLE_DATA],
    ]

with open(CWD / 'classifiers' / 'svc.pickle', 'rb') as f:
    svc = pickle.load(f)

with open(CWD / 'classifiers' / 'mlp.pickle', 'rb') as f:
    mlp = pickle.load(f)

with open(CWD / 'classifiers' / 'rfc.pickle', 'rb') as f:
    rfc = pickle.load(f)


def predict(text_1: str, text_2: str, classifier=svc) -> bool:
    """Определить написаны ли два текста одним автором или разными

    Args:
        text_1: первый текст
        text_2: второй текст
        classifier: классификатор определяющий авторство

    Returns:
        True - если написаны одним автором
        False - если разными
    """
    publication_1, publication_2 = Publication(text_1), Publication(text_2)

    res = classifier.predict([concatenate(publication_1.get_features(), publication_2.get_features())]).tolist()[0]

    return res
