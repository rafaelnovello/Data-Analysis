# -*- coding: utf-8 -*-

import re
import os.path
import requests
import logging
import argparse
import unicodedata

from goose import Goose
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords

import sklearn
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


Editorials = (
    ('http://g1.globo.com/economia/agronegocios/', 'agro'),
    ('http://g1.globo.com/ciencia-e-saude/', 'ciencia-e-saude'),
    ('http://g1.globo.com/economia/', 'economia'),
    ('http://g1.globo.com/educacao/', 'educacao'),
    ('http://g1.globo.com/politica/', 'politica'),
    ('http://g1.globo.com/tecnologia/', 'tecnologia'),
    ('http://politica.estadao.com.br/', 'politica'),
    ('http://economia.estadao.com.br/', 'economia'),
    ('http://educacao.estadao.com.br/', 'educacao'),
    ('http://ciencia.estadao.com.br/', 'ciencia-e-saude'),
    ('http://saude.estadao.com.br/', 'ciencia-e-saude')
)

def train(df, fit_file):
    print "\nTraining..."
    df = df.dropna()
    train_size = 0.8
    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None
    )
    forest = RandomForestClassifier(n_estimators=100)
    pipe = Pipeline([('vect', vectorizer), ('forest', forest)])
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        df.texto, df.categoria, train_size=train_size
    )
    pipe.fit(X_train, Y_train)
    accuracy = pipe.score(X_test, Y_test)
    print "\nAccuracy with {:.0%} of training data: {:.1%}\n".format(train_size, accuracy)
    pipe.fit(df.texto, df.categoria)
    joblib.dump(pipe, fit_file)

def predict(url, fit_file):
    pipe = joblib.load(fit_file)
    words = pre_processor(url)
    resp = pipe.predict([words])
    print "\nCategory: %s \n" % resp[0]
    resp = zip(pipe.classes_, pipe.predict_proba([words])[0])
    resp.sort(key=lambda tup: tup[1], reverse=True)
    for cat, prob in resp:
        print "Category {:16s} with {:.1%} probab.".format(cat, prob)

def to_unicode(data):
    try:
        data = data.decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        try:
            data = data.decode('iso-8859-1')
        except (UnicodeDecodeError, UnicodeEncodeError):
            try:
                data = data.decode('latin-1')
            except (UnicodeDecodeError, UnicodeEncodeError):
                data = data
    return data

def remove_nonlatin(s): 
    ss = []
    for ch in s:
        if ch == '\n':
            ss.append(' ')
            continue
        try:
            if unicodedata.name(unicode(ch)).startswith(('LATIN', 'SPACE')):
                ss.append(ch)
        except:
            continue
    return ''.join(ss)

def pre_processor(link):
    stops = set(stopwords.words("portuguese"))
    goose = Goose()
    article = goose.extract(link)
    text = article.cleaned_text
    text = remove_nonlatin(to_unicode(text))
    words = text.lower().split()
    words = ' '.join([w for w in words if not w in stops])
    return words

def worker(link, categ, lines):
    words = pre_processor(link)
    print "{:6d} words in: \t {:.50}" % (len(words), link)
    lines.append((link, categ, words))

def main(links_file, fit_file, to_predict):
    if not os.path.isfile(fit_file):
        df = prepare_data(links_file)
        train(df, fit_file)

    if to_predict:
        predict(to_predict, fit_file)


def prepare_data(links_file):
    pool = ThreadPoolExecutor(6)
    lines = []

    if links_file:
        links = pd.read_csv(links_file, sep=';')
        links = ((r['link'], r['categ']) for i, r in links.iterrows())
    else:
        links = []
        for edit, categ in Editorials:
            resp = requests.get(edit)
            html = BeautifulSoup(resp.content, 'lxml')
            posts = html.find_all('a', 'feed-post-link')
            posts = html.find_all('a', 'link-title') if not posts else posts
            links += [(post.get('href'), categ) for post in posts]

    for link, categ in links:
        pool.submit(worker, link, categ, lines)
        #worker(link, categ, lines)

    pool.shutdown(wait=True)
    df = DataFrame(lines)
    df.columns = ['link', 'categoria', 'texto']
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='')

    parser.add_argument(
        '-t', '--train',
        help='Treinar e salvar para o arquivo.',
    )

    parser.add_argument(
        '-p', '--predict',
        help='Prever',
    )

    parser.add_argument(
        '-f', '--file',
        help='Arquivo de links',
    )

    args = parser.parse_args()
    links_file = args.file
    to_predict = args.predict
    fit_file = args.train

    try:
        main(links_file, fit_file, to_predict)
    except Exception as e:
        logging.exception(e)
