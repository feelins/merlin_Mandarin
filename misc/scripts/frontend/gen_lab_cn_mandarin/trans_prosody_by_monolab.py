#!usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@author:shaopf
@file: trans_prosody_by_monolab.py 
@version:
@time: 2019/04/09 19:11:51
@email:feipengshao@163.com
@function： transfer Prosody by new monolab
"""

import IO
import os
import re


def createMonos(monolab_file):
    """monolab to monos"""
    tmp_monos = IO.readList(monolab_file)
    input_monos = tmp_monos[1:len(tmp_monos) - 1] # igmore the sil of begin and end
    monos_result = {}
    k = 0
    j = 0
    while k < len(input_monos):
        mono_array = input_monos[k].split()
        with_sp_follow = ''
        if k != len(input_monos) - 1:
            mono_array_next = input_monos[k + 1].split()
            if mono_array_next[2] == 'sp':
                with_sp_follow = 'sp'
                k += 1
        monos_result[j] = [mono_array[2], with_sp_follow]
        k += 1
        j += 1
    return monos_result


def genNewPhonLine(input_word_line, input_prosody_line, monos):
    """transfer the pinyin line by monolab"""
    input_prosodys = input_prosody_line.replace('$', '#').split('[') # ignore the break of old text line
    input_prosodys = [item for item in filter(lambda x: x.strip() != '', input_prosodys)]
    check_input_prosodys = re.split('\+|\]', input_prosody_line)
    check_input_prosodys = [item for item in filter(lambda x: x.strip() != '', check_input_prosodys)]
    input_words = input_word_line.split()
    assert len(monos) == len(check_input_prosodys)
    k = 0
    for j in range(len(input_prosodys)):
        phons = input_prosodys[j].split('+')
        for i in range(len(phons)):
            try:
                if phons[i][-1] == '#' and monos[k][1] == 'sp':
                    phons[i] = phons[i].replace('#', '$')
                k += 1
            except KeyError as e:
                print(str(e))
                exit(0)
        input_prosodys[j] = '+'.join(phons)
        if input_prosodys[j].find('$') != -1:
            input_words[j] = input_words[j].replace('#', '$')
    return ' '.join(input_words), '[' + '['.join(input_prosodys)


def transProsody(input_prosody_file, mono_path, output_prosody_file):
    """gen new Prosody"""
    prosody_list = IO.readList(input_prosody_file)
    for i in range(0, len(prosody_list), 3):
        file_name = prosody_list[i]
        mono_file = os.path.join(mono_path, file_name + '.lab')
        if os.path.exists(mono_file):
            print(prosody_list[i])
            monos = createMonos(mono_file)
            prosody_list[i + 1], prosody_list[i + 2] = genNewPhonLine(prosody_list[i + 1], prosody_list[i + 2], monos)
    IO.write(output_prosody_file, prosody_list)


if __name__ == '__main__':
    input_prosody_file = r''
    input_monolab_path = r''
    output_prosody_file = r''

    transProsody(input_prosody_file, input_monolab_path, output_prosody_file)
    print('Done!')
