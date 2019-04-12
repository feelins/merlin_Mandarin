#!usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@author:shaopf
@file: gen_pinyin_combos.py 
@version:
@time: 2019/04/09 09:57:57
@email:feipengshao@163.com
@function： gen Mandarin Pinyin combos
"""

import IO


def getVowels():
    """gen vowel list"""
    vowel_path = r'docs/Vowel.lst'
    return IO.readList(vowel_path)


def getConsonant():
    """gen Consonant list"""
    consonant_path = r'docs/Consonant.lst'
    return IO.readList(consonant_path)


def getCombos(vowels, consonants):
    """gen pinyin combos"""
    result = []
    for vowel in vowels:
        result.append(vowel + ',' + 'XX' + '+' + vowel)
        for consonant in consonants:
            result.append(consonant + vowel + ',' + consonant + '+' + vowel)
    result.sort()
    return result


combo_file = r'docs/pinyin_combo.lst'
vowels = getVowels()
consonants = getConsonant()
IO.write(combo_file, getCombos(vowels, consonants))
print('Done')
