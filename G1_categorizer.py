# -*- coding: utf-8 -*-

import re
import requests
import logging
import argparse
import unicodedata

from goose import Goose
from pandas import DataFrame
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor

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
    # s = (ch for ch in s
    #      if unicodedata.name(unicode(ch)).startswith(('LATIN', 'SPACE')))
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

def worker(link, categ, lines):
    stops = set(stopwords.words("portuguese"))
    goose = Goose()
    article = goose.extract(link)
    text = article.cleaned_text
    text = remove_nonlatin(to_unicode(text))
    words = text.lower().split()
    print len(words)
    words = ' '.join([w for w in words if not w in stops])
    lines.append((link, categ, words))

def main():
    pool = ThreadPoolExecutor(6)
    
    lines = []
    for edit, categ in Editorials:
        resp = requests.get(edit)
        html = BeautifulSoup(resp.content, 'lxml')
        posts = html.find_all('a', 'feed-post-link')
        posts = html.find_all('a', 'link-title') if not posts else posts

        for post in posts:
            link = post.get('href')
            future = pool.submit(worker, link, categ, lines)

    pool.shutdown(wait=True)
    df = DataFrame(lines)
    df.to_csv('bag_words.csv', index=False, encoding='utf-8', sep=';')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='')

    # parser.add_argument(
    #     '-f', '--fit', required=True,
    #     help='Treinar',
    # )

    # parser.add_argument(
    #     '-p', '--predict', required=True,
    #     help='Prever',
    # )

    # args = parser.parse_args()
    # fit = args.fit
    # predict = args.predict

    try:
        main()
    except Exception as e:
        logging.exception(e)
