# -*- coding: utf-8 -*-

import json
from pprint import pprint
from collections import Counter, defaultdict

path = 'usagov_bitly_data2012-03-16-1331923249.txt'

# Pure Python examples


def get_records():
    records = [json.loads(line) for line in open(path)]
    return records


def get_time_zones(records):
    time_zones = [rec['tz'] for rec in records if 'tz' in rec]
    return time_zones


def get_counts(sequence):
    counts = defaultdict(int)
    for el in sequence:
        counts[el] += 1
    return counts


def top_counts(sequence, limit):
    counts = Counter(sequence)
    return counts.most_common(limit)


def get_counts_with_pandas(records, limit):
    import pandas as pd
    from pandas import DataFrame, Series

    frame = DataFrame(records)
    tz_counts = frame['tz'].value_counts()
    return tz_counts[:limit]


if __name__ == '__main__':
    recs = get_records()
    tzs = get_time_zones(recs)
    tz_counts = get_counts(tzs)
    tz_sp = 'America/Sao_Paulo'
    print '%s -> %d' % (tz_sp, tz_counts[tz_sp])
    top = top_counts(tzs, 10)
    pprint(top)
    print get_counts_with_pandas(recs, 10)

