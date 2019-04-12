#!usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@author:shaopf
@file: vowel.py 
@version:
@time: 2019/04/09 10:04:35
@email:feipengshao@163.com
@function： vowel list of Mandarin
"""

import IO


vowels = []


def getVowels(vowel_path):
    """get the vowel list"""
    vowels = IO.readList(vowel_path)
