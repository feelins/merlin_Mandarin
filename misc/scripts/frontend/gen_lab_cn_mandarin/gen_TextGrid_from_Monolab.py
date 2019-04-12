#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:shaopf
@file: gen_TextGrid_from_Monolab.py
@version:
@time: 2019/04/09 17:20:45
@email:feipengshao@163.com
@function： gen TextGrid from monolab and Prosody
"""

import TextGrid
import wave
import os
import re
import IO


def getIntervalListFromMonolab(input_monolab_file):
    fr = open(input_monolab_file, 'r', encoding='utf-8')
    in_list = fr.readlines()
    phon_intervals = {}
    index = 1
    for i in range(len(in_list)):
        tmp_array = re.split(" |\t", in_list[i].strip())
        tmp_array = [item for item in filter(lambda x: x != '', tmp_array)]
        try:
            phon_intervals[index] = Interval(index, tmp_array[0], tmp_array[1], tmp_array[2])
            index += 1
        except IndexError as e:
            print(str(e) + in_list[i].strip() + "\t" + input_monolab_file)
            exit(0)
    return phon_intervals


def getIntervalListFromProsodyPhonLine(prosody_phon_line):
    """韵律文本的读音行，分隔成音节和读音的序列"""
    """返回带声调的，和不带声调的音节"""
    symbols = ["(L-L%)", "silv", "sp", "%"]
    # 增加正则表达式实现
    for sym in symbols:
        prosody_phon_line = prosody_phon_line.replace(sym, "")
    syllables = re.split("\#|\/|\[|\]|\*", prosody_phon_line)
    syllables = [item for item in filter(lambda x: x != '', syllables)]
    syllables = [item for item in map(lambda x: x.strip(), syllables)]
    syllables_noTone = []
    for syl in syllables:
        syllables_noTone.append(syl[0:-1])
    return syllables, syllables_noTone


def getIntervalListFromProsodyWordLine(prosody_word_line):
    sarray = re.split(" |\/|\[|\]", prosody_word_line)
    sarray = [item for item in filter(lambda x: x != '', sarray)]
    biaodian = ["，", "：", "。", "、", "？", "；", "！", "【", "】", "《", "》", "", "”", "“", '—', "-", "?", ";", "…", "(", ")"]
    newWord = []
    i = 0
    while i < len(sarray):
        next_word = ""
        if i != len(sarray) - 1:
            next_word = sarray[i + 1]
        cur_word = sarray[i]
        for j in range(len(cur_word)):
            tmp_cur_word = cur_word[j]
            if j == len(cur_word) - 1:
                if next_word in biaodian:
                    tmp_cur_word += next_word
                    i = i + 1
                tmp_cur_word += "1"
            newWord.append(tmp_cur_word)
        i = i + 1
    return newWord


def getNextLineArray(_in_list, i):
    """get the sArray of next line"""
    result = []
    if i != len(_in_list) - 1:
        result = re.split(' |\t', _in_list[i + 1].strip())
        result = [item for item in filter(lambda x: x != '', result)]
        result = [item for item in map(lambda x: x.strip(), result)]
    return result


def getCurLineArray(_in_list, i):
    """get the sArray of current line"""
    result = re.split(' |\t', _in_list[i].strip())
    result = [item for item in filter(lambda x: x != '', result)]
    result = [item for item in map(lambda x: x.strip(), result)]
    return result


def initialWordInterval(words, phons, phons_no_tone, mono_list):
    phon_intervals = []
    word_intervals = []
    break_intervals = []
    syll_index = 0
    mono_names = []
    break_name = ""
    tmp_word_begin = 0
    tmp_phon_end = 0
    i = 0
    while i < len(mono_list):
        tmp_array = getCurLineArray(mono_list, i)
        tmp_phon_end = tmp_array[1]
        tmp_next_array = getNextLineArray(mono_list, i)
        if tmp_array[2] not in ["sil", "silv", "sp"]:
            mono_names.append(tmp_array[2])
        if ' '.join(mono_names) == phons_no_tone[syll_index]:
            cur_syl_tone = phons[syll_index][-1]
            phon_intervals.append(Interval((i + 1), getFloat(tmp_array[0]), getFloat(tmp_phon_end),
                                           tmp_array[2] + cur_syl_tone))
            if tmp_next_array[2] in ["sil", "silv", "sp"]:
                tmp_phon_end = tmp_next_array[1]

                i += 1
                tmp_sil_array = getCurLineArray(mono_list, i)
                phon_intervals.append(
                    Interval((i + 1), getFloat(tmp_sil_array[0]), getFloat(tmp_sil_array[1]), tmp_sil_array[2]))
            if tmp_next_array[2] == 'sil':
                break_name = '4'
            if tmp_next_array[2] == 'silv' or tmp_next_array[2] == 'sp':
                break_name = '3'
            cur_word_name = words[syll_index]

            word_intervals.append(
                Interval((syll_index + 1), getFloat(tmp_word_begin), getFloat(tmp_phon_end),
                         cur_word_name + break_name))
            syll_index += 1
            tmp_word_begin = tmp_phon_end
            mono_names = []
            break_name = ""
        else:
            phon_intervals.append(Interval((i + 1), getFloat(tmp_array[0]), getFloat(tmp_phon_end), tmp_array[2]))
        i = i + 1
    phon_s = {}
    for i in range(len(phon_intervals)):
        phon_s[i + 1] = phon_intervals[i]
    word_s = {}
    for i in range(len(word_intervals)):
        word_s[i + 1] = word_intervals[i]
    break_s = {}
    for i in range(len(break_intervals)):
        break_s[i + 1] = break_intervals[i]
    return phon_s, word_s, break_s


def setToneInterval(phon_intervals):
    """根据第一层是否有停顿，增加边界调"""
    result_intervals = {}
    for i in range(1, len(phon_intervals) + 1):
        next_interval_name = ""
        if i != len(phon_intervals):
            next_interval_name = phon_intervals[i + 1].name
        if next_interval_name in ["sil", "silv", "sp"]:
            result_intervals[i] = Interval(phon_intervals[i].index, phon_intervals[i].begin, phon_intervals[i].end,
                                           "L-L%")
        else:
            result_intervals[i] = Interval(phon_intervals[i].index, phon_intervals[i].begin, phon_intervals[i].end, "")
    return result_intervals


def setBreakInterval(phon_intervals):
    """根据第一层是否有停顿，增加边界调"""
    result_intervals = []
    for i in range(len(phon_intervals)):
        next_interval_name = ""
        if i != len(phon_intervals) - 1:
            next_interval_name = phon_intervals[i + 1].name
        if next_interval_name in ["sil", "silv"]:
            result_intervals.append(
                Interval(phon_intervals[i].index, phon_intervals[i].begin, phon_intervals[i].end, "L-L%"))
        else:
            result_intervals.append(
                Interval(phon_intervals[i].index, phon_intervals[i].begin, phon_intervals[i].end, ""))
    return result_intervals


def getBlankTiers(input_intervals):
    result_intervals = []
    for i in range(len(input_intervals)):
        result_intervals.append(
            Interval(input_intervals[i].index, input_intervals[i].begin, input_intervals[i].end, ""))
    return result_intervals


def getFloat(input_str):
    return str(float(input_str) / 10000000)


input_file = r'/home/shaopf/study/BiaoBeiData/BZNSYP/Prosody1.txt'  # 为了
wav_path = r'data/wav'
monolab_path = r'data/monolab'
textgrid_path = r'data/TextGrid_Initial'  # 输出

# 如果目录不存在，创建输出目录
if not os.path.exists(textgrid_path):
    os.makedirs(textgrid_path)

f = open(input_file, 'r', encoding='utf-8')
lines = f.readlines()
lineNum = len(lines)
tones = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

for i in range(0, len(lines), 3):
    fileName = lines[i].strip()
    print('Processing ' + fileName)
    monolab_file = os.path.join(monolab_path, fileName + ".lab")
    wav_file = os.path.join(wav_path, fileName + ".wav")
    textgrid_file = os.path.join(textgrid_path, fileName + ".TextGrid")
    if os.path.exists(wav_file) and os.path.exists(monolab_file):
        mono_list = IO.readList(monolab_file)
        wavefile = wave.open(wav_file, 'r')
        framerate = wavefile.getframerate()
        numframes = wavefile.getnframes()
        duration = float(numframes) / float(framerate)
        word_list = lines[i + 1].strip().split()
        syllable_phons_list, syllable_phons_no_tone_list = getIntervalListFromProsodyPhonLine(lines[i + 2].strip())
        if len(word_list) != len(syllable_phons_list):
            print("hello")
            exit(0)
        phons, words, breaks = initialWordInterval(word_list, syllable_phons_list, syllable_phons_no_tone_list,
                                                   mono_list)
        phons[len(phons)].end = duration
        words[len(words)].end = duration
        result_tiers = {1: Tier(None, "Phon", phons), 2: Tier(None, "Word", words)}
        tg = TextGrid(textgrid_file)
        tg.write_new(2, duration, result_tiers)
f.close()
