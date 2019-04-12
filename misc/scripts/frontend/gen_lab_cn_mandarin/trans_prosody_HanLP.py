#!usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@author:shaopf
@file: trans_prosody_HanLP.py 
@version:
@time: 2019/04/09 11:35:19
@email:feipengshao@163.com
@function： transfer HanLP result to prosody
input_pinyin_file:
0001|句子内容是什么
...
input_pinyin_file:
0001|ju4 zi5 # nei4 rong2 # shi4 # shen2 me5 %

output_prosody_file:
0001
句 子# 内 容# 是# 什 么
[j v4][z ii5]#[n ei4][r ong2]#[sh iii4]#[sh en2][m e5]%
...
"""

import IO


def createPYtoSY():
    """gen sheng yun from pinyin"""
    input_combo_list = IO.readList(r'docs/pinyin_combo.lst')
    py_maps = {}
    for line in input_combo_list:
        line_array = line.split(',')
        py_maps[line_array[0]] = line_array[1].replace('XX+', '')
    return py_maps


def createNewPY():
    """trans normal pinyin to TTS pinyin"""
    py_trans = {}
    input_pinyin_list = IO.readList(r'docs/transTTSPinyin.txt')
    for line in input_pinyin_list:
        line_array = line.split(',')
        py_trans[line_array[0]] = line_array[1]
    return py_trans


def createWords(word_str):
    """Cn Mandarin sentence to Cn Mandarin Words list"""
    pre_func = IO.readList(r'docs/pre_punctuation.txt')[1:]
    lat_func = IO.readList(r'docs/post_punctuation.txt')[1:]
    en_letters = IO.readList(r'docs/special_English_letters.txt')[1:]
    words = []
    j = 0
    tmp_word = ''
    while j < len(word_str):
        find_pre_func = 0
        while j < len(word_str) and word_str[j] in pre_func:
            tmp_word += word_str[j]
            find_pre_func = 1
            j += 1
        if (u'\u9fa5' >= word_str[j] >= u'\u4e00') or word_str[j] in en_letters:
            if find_pre_func:
                tmp_word += word_str[j]
            else:
                tmp_word = word_str[j]
            j = j + 1
            while j < len(word_str) and word_str[j] in lat_func:
                tmp_word += word_str[j]
                j = j + 1
            words.append(tmp_word)
            tmp_word = ''
    return words


def createPhons(phon_str):
    """create phons list, be aware of the prosody break #, *, $, %"""
    tts_pinyin = createNewPY()
    shengyuns = createPYtoSY()
    tmp_phon_str = phon_str
    tmp_phon_str = tmp_phon_str.replace(' #', '# ')
    tmp_phon_str = tmp_phon_str.replace(' *', '* ') # not used now
    tmp_phon_str = tmp_phon_str.replace(' $', '$ ')
    tmp_phon_str = tmp_phon_str.replace(' %', '% ')
    prosodys = ['#', '*', '$', '%']
    tones = ['1', '2', '3', '4', '5']
    punctions = IO.readList(r'docs/pre_punctuation.txt')[1:]
    punctions.extend(IO.readList(r'docs/post_punctuation.txt')[1:])
    old_pinyins = tmp_phon_str.split()
    old_pinyins = [item for item in filter(lambda x: x.strip() != '', old_pinyins)]
    old_pinyins = [item for item in map(lambda x: x.strip(), old_pinyins)]
    old_pinyins = [item for item in filter(lambda x: x not in punctions, old_pinyins)]
    new_pinyins = []
    for py in old_pinyins:
        cur_tone = ''
        cur_prosody = ''
        cur_py = py
        # check prosody
        if cur_py[-1] in prosodys:
            cur_prosody = cur_py[-1]
            cur_py = cur_py[:-1]
        # check tone
        if cur_py[-1] in tones:
            cur_tone = cur_py[-1]
            cur_py = cur_py[:-1]
        else:
            print('Error: tone' + phon_str + ',' + py)
            exit(0)
        trans_py = cur_py
        if trans_py in tts_pinyin:
            trans_py = tts_pinyin[trans_py]
        cur_shengyun = shengyuns[trans_py]
        new_pinyins.append(cur_shengyun + cur_tone + ']' + cur_prosody)
    return new_pinyins


def setWordsProdosy(words, phons):
    """set word prosody according to phons break"""
    prosodys = ['#', '*', '$', '%']
    new_words = []
    for i in range(len(words)):
        cur_prosody = ''
        if phons[i][-1] in prosodys:
            cur_prosody = phons[i][-1]
        new_words.append(words[i] + cur_prosody)
    return new_words


def genProsody(input_hanzi_file, input_pinyin_file, save_prosody_file):
    """gen prosody"""
    hanzi_lines = IO.readList(input_hanzi_file)
    pinyin_lines = IO.readList(input_pinyin_file)
    assert len(hanzi_lines) == len(pinyin_lines)
    results = []
    for i in range(len(hanzi_lines)):
        hz_line_array = hanzi_lines[i].split('|')
        py_line_array = pinyin_lines[i].split('|')
        print('Processing ' + hz_line_array[0])
        cur_words = createWords(hz_line_array[1])
        cur_phons = createPhons(py_line_array[1])
        assert len(cur_phons) == len(cur_words)
        words_prosody = setWordsProdosy(cur_words, cur_phons)

        # output
        results.append(hz_line_array[0])
        results.append(' '.join(words_prosody))
        results.append('[' + '['.join(cur_phons))
    IO.write(save_prosody_file, results)


if __name__ == '__main__':
    input_pinyin_file = r'data/sample_pinyin_file.txt'
    input_hanzi_file = r'data/sample_hanzi_file.txt'
    output_prosody_file = r'data/Prosody.txt'

    genProsody(input_hanzi_file, input_pinyin_file, output_prosody_file)
    print('Done!')
