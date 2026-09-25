"""Version 1: insert each appended batch after ten historical batches.

No batch is split; old and new internal orders and member/gain pairs survive.
The deterministic cumulative allocation ends with one new batch, not 100.
"""
from collections import Counter

VERSION = 'small-scale-interleave-v1'


def permutation():
    result = []
    old, new = 0, 1000
    for position in range(1100):
        if (position + 1) * 100 // 1100 > position * 100 // 1100:
            result.append(new); new += 1
        else:
            result.append(old); old += 1
    return result


def reorder(values, order):
    if len(values) != 6600 or sorted(order) != list(range(1100)):
        raise ValueError('Incomplete or duplicate batch permutation')
    return [x for i in order for x in values[6*i:6*i+6]]


def validate(before, gains, after, new_gains, order):
    if order != permutation(): raise ValueError('Unfrozen order')
    if after != reorder(before, order) or new_gains != reorder(gains, order):
        raise ValueError('Member or brightness order drift')
    if Counter(zip(before, gains)) != Counter(zip(after, new_gains)):
        raise ValueError('Member/augmentation combination drift')
    return True
