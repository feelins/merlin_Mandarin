#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:shaopf
@file: gen_pinyin_combos.py
@version:
@time: 2019/04/09 09:57:57
@email:feipengshao@163.com
@function： gen monolab from TextGrid
"""

import os
from TextGrid import *
import IO

plist = []
def cleanName(_in_phon_name, _strange_marks:list) -> str:
    """替换掉韵律标记符号，比如声调，比如重读"""
    cur_phon = _in_phon_name
    for suf in _strange_marks:
        cur_phon = cur_phon.replace(suf, "")
    return cur_phon

def gen(_in_path, _out_path, _strange_marks:list) -> None:
    """生成monolab，输入TextGrid目录，输出monolab目录，替换第一层其它任意符号"""
    if not os.path.exists(_out_path):
        os.mkdir(_out_path)
    
    for _fileName in os.listdir(_in_path):
        print(_fileName)
        simple_name = _fileName.split('.')[0]
        _oldName = os.path.join(_in_path, _fileName)
        _newName = os.path.join(_out_path, simple_name + ".lab")
        _tg = TextGrid(_oldName)
        _tg.read()
        phon_tier = 1
        cur_tiers = _tg.tiers[phon_tier]
        result = []
        for i in range(1, cur_tiers.intervals_num + 1):
            cur_interval = cur_tiers.intervals[i]
            cur_phon = cleanName(cur_interval.name, _strange_marks)
            cur_begin = str(int(float(cur_interval.begin) * 10000000))
            cur_end = str(int(float(cur_interval.end) * 10000000))
            if cur_phon not in plist:
                plist.append(cur_phon)
            result.append(cur_begin + " " + cur_end + " " + cur_phon)
        IO.write(_newName, result)

if __name__=='__main__':
    open_path = r'/home/shaopf/study/BiaoBeiData/BZNSYP/TextGrid4'
    save_path = r'/home/shaopf/study/BiaoBeiData/BZNSYP/monolab'
    save_phoneme_list = r'/home/shaopf/study/BiaoBeiData/BZNSYP/phoneme.lst'
    tones = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    gen(open_path, save_path, tones)
    plist.sort()
    IO.write(save_phoneme_list, plist)
    print('Done!')
