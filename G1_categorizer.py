# -*- coding: utf-8 -*-

import re
import requests
import logging
import argparse
import unicodedata

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
)

def remove_nonlatin(s): 
    s = (ch for ch in s
         if unicodedata.name(ch).startswith(('LATIN', 'SPACE')))
    return ''.join(s)

def worker(link, categ, lines):
    stops = set(stopwords.words("portuguese"))
    html = requests.get(link).content
    html = BeautifulSoup(html, 'lxml')
    matter = html.find_all('div', 'content-text')
    if not matter:
        return
    text = ' '.join([ch.get_text() for ch in matter])
    text = remove_nonlatin(text)
    words = text.lower().split()
    words = ' '.join([w for w in words if not w in stops])
    lines.append((link, categ, words))

def main():
    pool = ThreadPoolExecutor(6)
    
    lines = []
    for edit, categ in Editorials:
        resp = requests.get(edit)
        html = BeautifulSoup(resp.content, 'lxml')
        posts = html.find_all('a', 'feed-post-link')

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
