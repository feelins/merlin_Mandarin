#!usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@author:shaopf
@file: prepare_data_for_Montreal_Align.py 
@version:
@time: 2019/04/09 14:20:45
@email:feipengshao@163.com
@function： prepare labels and dict for Montreal Alignment, need input pinyin file
"""

import IO
import os


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


def genDictLabWord(input_file, out_lab_path, out_dict_file):
    """gen data"""
    IO.createDir(out_lab_path)
    input_list = IO.readList(input_file)
    tmp_dict_list = {}
    tts_pinyin = createNewPY()
    shengyuns = createPYtoSY()
    punctions = IO.readList(r'docs/pre_punctuation.txt')[1:]
    punctions.extend(IO.readList(r'docs/post_punctuation.txt')[1:])
    for line in input_list:
        line_array = line.split('|')
        file_name = line_array[0]
        save_lab_file = os.path.join(out_lab_path, file_name + '.lab')
        print(file_name)
        out_lab_list = []
        init_line = line_array[1]
        init_line = init_line.replace(' #', '/ ')
        init_line = init_line.replace(' *', '/ ')  # not used now
        init_line = init_line.replace(' $', '/ ')
        init_line = init_line.replace(' %', '/ ')
        old_pinyins = init_line.split('/ ')
        new_pinyins = []
        for py in old_pinyins:
            tmp_py = py
            for func in punctions:
                tmp_py = tmp_py.replace(func, '')
            tmp_py = tmp_py.strip()
            if tmp_py != '':
                new_pinyins.append(tmp_py)

        # output the labels
        lab_pinyins = [item for item in map(lambda x: x.replace(' ', ''), new_pinyins)]
        out_lab_list.append(' '.join(lab_pinyins))
        IO.write(save_lab_file, out_lab_list)

        # gen dict
        for word_py in new_pinyins:
            dict_word_pinyins = []
            syllable_pinyins = word_py.split()
            syllable_pinyins = [item for item in filter(lambda x: x.strip() != '', syllable_pinyins)]
            for pinyin in syllable_pinyins:
                tmp_pinyin = pinyin[:-1]
                tmp_tone = pinyin[-1:]
                try:
                    trans_pinyin = tmp_pinyin
                    if trans_pinyin in tts_pinyin:
                        trans_pinyin = tts_pinyin[trans_pinyin]
                    cur_shengyun = shengyuns[trans_pinyin].replace('+', ' ')
                    dict_word_pinyins.append(cur_shengyun + tmp_tone)
                except KeyError as e:
                    print(str(e) + ',' + word_py + 'in:' + str(new_pinyins))
                    exit(0)
            word_py = word_py.replace(' ', '')
            if word_py not in tmp_dict_list:
                tmp_dict_list[word_py] = ' '.join(dict_word_pinyins)
    result_dict = []
    for k, v in tmp_dict_list.items():
        result_dict.append(str(k) + '\t' + str(v))
    IO.write(out_dict_file, result_dict)


if __name__ == '__main__':
    input_pinyin_file = r'data/sample_pinyin_file.txt'
    output_dict_file = r'data/montreal_align_dict.txt'
    output_lab_align_path = r'data/lab_align'

    genDictLabWord(input_pinyin_file, output_lab_align_path, output_dict_file)
    print('Done!')

