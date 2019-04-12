#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:shaopf
@file: gen_pinyin_combos.py
@version:
@time: 2019/04/10 19:57:57
@email:feipengshao@163.com
@function： gen fulllab
"""

import re
import os
import IO

global vowels

global all_l4s
global all_l3s
global all_l1s
global all_lws
global all_l0s
global all_lps


class Unit:
    """means the phoneme, the smallest unit, corresponding with the intervals of Tier1"""
    ls = {'lp': '+|%|$|\*|#|\[|\/| |\]', 'l0': '%|$|\*|#|\[|\/', 'lw': '%|$|\*|#', 'l1': '%|$|\*', 'l3': '%|$', 'l4': '%'} # 所有分隔符号
    suffix = ["(L-L%)", "(L-H%)"]
    tones = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    accent_stress = ["(1)", "(2)", "(H*)"]

    def __init__(self, _index, _name):
        self.in_sen = _index
        self.name = self.clean_name(_name, self.suffix)
        
    def clean_name(self, _input_string, _in_symbols):
        out_string = _input_string
        for ss in _in_symbols:
            out_string = out_string.replace(ss, "")
        return out_string
    
    def splits(self, _splits_cat, _input_str):
        cur_list = re.split(self.ls[_splits_cat], _input_str)
        cur_list = [item for item in filter(lambda x: x != '', cur_list)]
        return cur_list


class Lp(Unit):
    """means the phoneme, the smallest unit, corresponding with the intervals of Tier1"""
    def __init__(self, _index, _name, _lp_index_l0, _lp_in_l0, _lp_in_l1, _lp_in_l3, _lp_in_l4):
        Unit.__init__(self, _index, _name)
        self.name = Unit.clean_name(self, self.name, Unit.tones)
        self.name = Unit.clean_name(self, self.name, Unit.accent_stress)
        self.in_l0_f = _lp_index_l0
        self.in_l0_b = _lp_in_l0.has_lps - _lp_index_l0 + 1
        self.atl4 = _lp_in_l4
        self.atl3 = _lp_in_l3
        self.atl2 = 0
        self.atl1 = _lp_in_l1
        self.atl0 = _lp_in_l0
        self.pos = "x"


class L0(Unit):
    """means the syllable, with only one vowel"""
    def __init__(self, _index, _index_in_l3, _index_in_l1, _name, _l0_in_l1, _l0_in_l3, _l0_in_l4):
        Unit.__init__(self, _index, _name)
        self.in_l3_f = _index_in_l3
        self.in_l3_b = _l0_in_l3.has_l0s - _index_in_l3 + 1
        self.in_l1_f = _index_in_l1
        self.in_l1_b = _l0_in_l1.has_l0s - _index_in_l1 + 1
        self.atl4 = _l0_in_l4
        self.atl3 = _l0_in_l3
        self.l0_in_lw = 0
        self.atl1 = _l0_in_l1
        self.has_lps = self.get_lps()
        self.vowel = self.get_vowel()
        self.stress = "x"
        self.accent = "x"
        self.tone = self.get_tone()
        
    def get_lps(self):
        cur_list = re.split(Unit.ls['lp'], self.name)
        cur_list = [item for item in filter(lambda x: x != '', cur_list)]
        sum_lps = 0
        for phon in cur_list:
            if phon.find('silv') == -1:
                sum_lps += 1
        return sum_lps
    
    def get_vowel(self):
        global vowels
        tmp_name = Unit.clean_name(self, self.name, Unit.accent_stress)
        tmp_name = Unit.clean_name(self, tmp_name, Unit.tones)
        cur_list = Unit.splits(self, "lp", tmp_name)
        vowel_num = 0
        cur_vowel = ""
        for ph in cur_list:
            if ph in vowels:
                vowel_num += 1
                cur_vowel = ph
        if vowel_num != 1 or cur_vowel not in vowels:
            print('Vowels number is not correct! ' + self.name)
            exit(0)
        return cur_vowel
    
    def get_stress(self):
        """stress of syllable, used as (1), (2)"""
        if self.info_name.find("(1)") != -1 or self.info_name.find("(2)") != -1:
            return "1"
        else:
            return "0"
        
    def get_accent(self):
        """stress of syllable, used as (1), (2)"""
        if self.info_name.find("(H*)") != -1:
            return "1"
        else:
            return "0"
        
    def get_tone(self):
        tmp_name = Unit.clean_name(self, self.name, Unit.accent_stress)
        cur_list = Unit.splits(self, "lp", tmp_name)
        if 'silv' in cur_list:
            cur_list.remove('silv')
        if 'sil' in cur_list:
            cur_list.remove('sil')
        if 'sp' in cur_list:
            cur_list.remove('sp')
        cur_tone = cur_list[-1][-1]
        if cur_tone not in Unit.tones:
            print('Error find tone ' + self.name)
            exit(0)
        return cur_tone
        
        
class Lw(Unit):
    """with the ][, normally connected with pos, means a lexcial word"""
    pass
    
    
class L1(Unit):
    """with the * in Tibetan, with ][ in other syllable languages"""
    def __init__(self, _index, _l1_index_l3, _l1_index_l4, _name, _l1_in_l3, _l1_in_l4):
        Unit.__init__(self, _index, _name)
        self.in_l3_f = _l1_index_l3
        self.in_l3_b = _l1_in_l3.has_l1s - _l1_index_l3 + 1
        self.in_l4 = _l1_index_l4
        self.has_lws = len(Unit.splits(self, "lw", self.name))
        self.has_l0s = len(Unit.splits(self, "l0", self.name))
        self.has_lps = len(Unit.splits(self, "lp", self.name))
        self.atl3 = _l1_in_l3
        self.atl4 = _l1_in_l4
    
    
class L3(Unit):
    """generally means a phrase, with L-L%, with sil or silence, also L4 is a L3"""
    def __init__(self, _index, _l3_index_l4, _name, _l3_in_l4):
        Unit.__init__(self, _index, _name)
        self.in_l4_f = _l3_index_l4
        self.in_l4_b = _l3_in_l4.has_l3s - _l3_index_l4 + 1
        self.has_l1s = len(Unit.splits(self, "l1", self.name))
        self.has_lws = len(Unit.splits(self, "lw", self.name))
        self.has_l0s = len(Unit.splits(self, "l0", self.name))
        self.has_lps = len(Unit.splits(self, "lp", self.name))
        self.atL4 = _l3_in_l4
        self.tone = "x"
        
        
class L4(Unit):
    """generally means the sentence, specially has several L4s in a sentence"""
    def __init__(self, _index, _name):
        Unit.__init__(self, _index, _name)
        self.has_l3s = len(Unit.splits(self, "l3", self.name))
        self.has_l1s = len(Unit.splits(self, "l1", self.name))
        self.has_lws = len(Unit.splits(self, "lw", self.name))
        self.has_l0s = len(Unit.splits(self, "l0", self.name))
        self.has_lps = len(Unit.splits(self, "lp", self.name))

  
def init_vowels(_vowel_path):
    global vowels
    vowels = IO.readList(_vowel_path)


def splits(_splits_chars, _input_str):
    cur_list = re.split(_splits_chars, _input_str)
    cur_list = [item for item in filter(lambda x: x != '', cur_list)]
    return cur_list


def read(input_line):
    global all_l4s
    global all_l3s
    global all_l1s
    global all_lws
    global all_l0s
    global all_lps
    all_l4s = []
    all_l3s = []
    all_l1s = []
    all_l0s = []
    all_lws = []
    all_lps = []
    cur_l4s = splits(Unit.ls["l4"], input_line)
    l3_index_sen = 1
    l1_index_sen = 1
    lp_index_sen = 1
    l0_index_sen = 1
    for l4_index in range(len(cur_l4s)):
        cur_l4_name = cur_l4s[l4_index]
        cur_l4 = L4(l4_index, cur_l4_name) # Initial L4
        
        cur_l3s = splits(Unit.ls["l3"], cur_l4_name)
        l1_index_l4 = 1
        for tmp_l1_index_l4 in range(0, l4_index):
            l1_index_l4 += len(splits(Unit.ls["l1"], cur_l4s[tmp_l1_index_l4]))

        for l3_index in range(len(cur_l3s)):
            cur_l3_name = cur_l3s[l3_index]
            cur_l3 = L3(l3_index_sen, (l3_index + 1), cur_l3_name, cur_l4) # Initial L3
            l3_index_sen += 1
            
            for tmp_l1_index_l3 in range(0, l3_index):
                l1_index_l4 += len(splits(Unit.ls["l1"], cur_l3s[tmp_l1_index_l3]))
            
            cur_l1s = splits(Unit.ls["l1"], cur_l3_name)
            
            for l1_index in range(len(cur_l1s)):
                cur_l1_name = cur_l1s[l1_index]
                cur_l1 = L1(l1_index_sen, (l1_index + 1), l1_index_l4, cur_l1_name, cur_l3, cur_l4) # Initial L1
                l1_index_sen += 1
                
                l0_index_l3 = 1
                for tmp_l0_index_l1 in range(0, l1_index):
                    l0_index_l3 += len(splits(Unit.ls["l0"], cur_l1s[tmp_l0_index_l1]))
                
                cur_l0s = splits(Unit.ls["l0"], cur_l1_name)
                for l0_index in range(len(cur_l0s)):
                    cur_l0_name = cur_l0s[l0_index]
                    cur_lps = splits(Unit.ls["lp"], cur_l0_name)
                    cur_l0 = L0(l0_index_sen, (l0_index_l3 + l0_index), (l0_index + 1), cur_l0_name, cur_l1, cur_l3, cur_l4)
                    l0_index_sen += 1
                    cur_lps = splits(Unit.ls["lp"], cur_l0_name)
                    for lp_index in range(len(cur_lps)):
                        cur_lp_name = cur_lps[lp_index]
                        cur_lp = Lp(lp_index_sen, cur_lp_name, (lp_index + 1), cur_l0, cur_l1, cur_l3, cur_l4) # Initial Lp
                        lp_index_sen += 1
                        if cur_lp.name != '':
                            all_lps.append(cur_lp)
                    all_l0s.append(cur_l0)
                all_l1s.append(cur_l1)
            all_l3s.append(cur_l3)
        all_l4s.append(cur_l4)


def write(_save_path, mono_list):
    results = []
    k = 0
    #### HTS need the duration info
    results.append(mono_list[k].split()[0] + " " + mono_list[k].split()[1] + " " + "xx^xx-sil+" + all_lps[0].name + "=" + all_lps[1].name + "@xx_xx/A:\n")
    k += 1
    for i in range(len(all_lps)):
        temp_str = ""
        temp_str += print_label_0(i)
        temp_str += print_label_a(i)
        temp_str += print_label_b(i)
        temp_str += print_label_c(i)
        temp_str += print_label_d(i)
        temp_str += print_label_e(i)
        temp_str += print_label_f(i)
        temp_str += print_label_g(i)
        temp_str += print_label_h(i)
        temp_str += print_label_i(i)
        temp_str += print_label_j(i)
        temp_str += print_label_t(i)        
        results.append(mono_list[k].split()[0] + " " + mono_list[k].split()[1] + " " + temp_str + "\n")
        k += 1
    results.append(mono_list[k].split()[0] + " " + mono_list[k].split()[1] + " " + all_lps[-2].name + "^" + all_lps[-1].name + "-sil+xx=xx@xx_xx/A:\n")
    save_file = open(_save_path, 'w', encoding = 'ascii')
    save_file.writelines(results)


def print_label_0(_i):
    result_str = ""
    if _i == 0:
        result_str += "xx^sil"
    elif _i == 1:
        result_str += "sil^" + all_lps[0].name
    else:
        result_str += all_lps[_i - 2].name + "^" + all_lps[_i - 1].name
    result_str += "-"
    result_str += all_lps[_i].name
    result_str += "+"
    if _i == len(all_lps) - 1:
        result_str += "sil=xx"
    elif _i == len(all_lps) - 2:
        result_str += all_lps[-1].name + "=sil"
    else:
        result_str += all_lps[_i + 1].name + "=" + all_lps[_i + 2].name
    result_str += "@"
    if all_lps[_i].name in ["silv", "sil", "sp"]:
        result_str += "xx_xx/"
    else:
        result_str += str(all_lps[_i].in_l0_f) + "_"
        result_str += str(all_lps[_i].in_l0_b) + "/"
    return result_str   


def print_label_a(_i):
    if all_lps[_i].atl0.in_l3_f == 1:
        return "A:xx_xx_xx/"
    else:
        cur_l0_index = all_lps[_i].atl0.in_sen
        prev_l0_index = cur_l0_index - 1
        result_index = -1
        for j in range(len(all_l0s)):
            if all_l0s[j].in_sen == prev_l0_index:
                result_index = j
                break
        prev_l0 = all_l0s[result_index]
        #return "A:" + prev_l0.stress + "~" + "x" + "_" + str(prev_l0.has_lps) + "/"
        # 不涉及重读，重音，置空
        return "A:" + 'xx' + "_" + 'xx' + "_" + str(prev_l0.has_lps) + "/"


def print_label_b(_i):
    cur_l0 = all_lps[_i].atl0
    result_str = "B:"
    result_str += 'xx' + "-"
    result_str += 'xx' + "-"
    result_str += str(cur_l0.has_lps) + "@"
    result_str += str(cur_l0.in_l1_f) + "-"
    result_str += str(cur_l0.in_l1_b) + "&"
    result_str += str(cur_l0.in_l3_f) + "-"
    result_str += str(cur_l0.in_l3_b) + "#"
    result_str += 'xx' + "-"
    result_str += 'xx' + "$"
    result_str += 'xx' + "-"
    result_str += 'xx' + "!"
    result_str += 'xx' + "-"
    result_str += 'xx' + ";"
    result_str += 'xx' + "-"
    result_str += 'xx' + "|"
    result_str += all_lps[_i].atl0.vowel + "/"
    return result_str

def print_label_c(_i):
    if all_lps[_i].atl0.in_l3_b == 1:
        return "C:xx+xx+xx/"
    else:
        cur_l0_index = all_lps[_i].atl0.in_sen
        next_l0_index = cur_l0_index + 1
        result_index = -1
        for j in range(len(all_l0s)):
            if all_l0s[j].in_sen == next_l0_index:
                result_index = j
                break
        next_l0 = all_l0s[result_index]
        # return "D:" + next_l0.stress + "~" + next_l0.accent + "_" + str(next_l0.has_lps) + "/"
        return "C:" + 'xx' + "+" + 'xx' + "+" + str(next_l0.has_lps) + "/"


def print_label_d(_i):
    if all_lps[_i].atl1.in_l3_f == 1:
        return "D:xx_xx/"
    else:
        prev_pos = all_lps[_i - 1].pos
        cur_l1_index = all_lps[_i].atl1.in_sen
        prev_l1_index = cur_l1_index - 1
        result_index = -1
        for j in range(len(all_l1s)):
            if all_l1s[j].in_sen == prev_l1_index:
                result_index = j
                break
        prev_l1 = all_l1s[result_index]
        return "D:" + 'xx' + "_" + str(prev_l1.has_l0s) + "/"


def print_label_e(_i):
    result_str = "E:"
    result_str += "xx" + "+"
    result_str += str(all_lps[_i].atl1.has_l0s) + "@"
    result_str += str(all_lps[_i].atl1.in_l3_f) + "+"
    result_str += str(all_lps[_i].atl1.in_l3_b) + "&"
    result_str += "xx" + "+"
    result_str += "xx" + "#"
    result_str += "xx" + "+"
    result_str += "xx" + "/"
    return result_str


def print_label_f(_i):
    if all_lps[_i].atl1.in_l3_b == 1:
        return "F:xx_xx/"
    else:
        next_pos = all_lps[_i + 1].pos
        cur_l1_index = all_lps[_i].atl1.in_sen
        next_l1_index = cur_l1_index + 1
        result_index = -1
        for j in range(len(all_l1s)):
            if all_l1s[j].in_sen == next_l1_index:
                result_index = j
                break
        next_l1 = all_l1s[result_index]
        return "F:" + "xx" + "_" + str(next_l1.has_l0s) + "/"


def print_label_g(_i):
    if all_lps[_i].atl3.in_l4_f == 1:
        return "G:xx_xx/"
    else:
        cur_l3_index = all_lps[_i].atl3.in_sen
        prev_l3_index = cur_l3_index - 1
        result_index = -1
        for j in range(len(all_l3s)):
            if all_l3s[j].in_sen == prev_l3_index:
                result_index = j
                break
        prev_l3 = all_l3s[result_index]
        return "G:" + str(prev_l3.has_l0s) + "_" + str(prev_l3.has_l1s) + "/"


def print_label_h(_i):
    result_str = "H:"
    result_str += str(all_lps[_i].atl3.has_l0s) + "="
    result_str += str(all_lps[_i].atl3.has_l1s) + "^"
    result_str += str(all_lps[_i].atl3.in_l4_f) + "="
    result_str += str(all_lps[_i].atl3.in_l4_b) + "|"
    result_str += str(all_lps[_i].atl3.tone) + "/"
    return result_str


def print_label_i(_i):
    if all_lps[_i].atl3.in_l4_b == 1:
        return "I:xx=xx/"
    else:
        cur_l3_index = all_lps[_i].atl3.in_sen
        next_l3_index = cur_l3_index + 1
        result_index = -1
        for j in range(len(all_l3s)):
            if all_l3s[j].in_sen == next_l3_index:
                result_index = j
                break
        next_l3 = all_l3s[result_index]
        return "I:" + str(next_l3.has_l0s) + "=" + str(next_l3.has_l1s) + "/"


def print_label_j(_i):
    result_str = "J:"
    result_str += str(all_lps[_i].atl4.has_l0s) + "+"
    result_str += str(all_lps[_i].atl4.has_l1s) + "-"
    result_str += str(all_lps[_i].atl4.has_l3s) + "/"
    return result_str


def print_label_k(_i):
    return ""


def print_label_l(_i):
    return "L:1/"


def print_label_t(_i):
    result_str = "T:"
    if all_lps[_i].atl0.in_l3_f == 1:
        result_str += "xx"
    else:
        cur_l0_index = all_lps[_i].atl0.in_sen
        prev_l0_index = cur_l0_index - 1
        result_index = -1
        for j in range(len(all_l0s)):
            if all_l0s[j].in_sen == prev_l0_index:
                result_index = j
                break
        prev_l0 = all_l0s[result_index]
        result_str += prev_l0.tone
    result_str += "_"
    result_str += all_lps[_i].atl0.tone + "@"
    if all_lps[_i].atl0.in_l3_b == 1:
        result_str += "xx" + "|"
    else:
        cur_l0_index = all_lps[_i].atl0.in_sen
        next_l0_index = cur_l0_index + 1
        result_index = -1
        for j in range(len(all_l0s)):
            if all_l0s[j].in_sen == next_l0_index:
                result_index = j
                break
        next_l0 = all_l0s[result_index]
        result_str += next_l0.tone + "|"
    return result_str


def get_stress_num_before(_i):
    stress_num = 0
    for i in range(len(all_l0s)):
        if all_l0s[i].atl3.in_sen == all_lps[_i].atl3.in_sen and all_l0s[i].in_l3_f <= all_lps[_i].atl0.in_l3_f and all_l0s[i].stress not in ["x", "0"]:
            stress_num += 1
    return str(stress_num)


def get_stress_num_after(_i):
    stress_num = 0
    for i in range(len(all_l0s)):
        if all_l0s[i].atl3.in_sen == all_lps[_i].atl3.in_sen and all_l0s[i].in_l3_f >= all_lps[_i].atl0.in_l3_f and all_l0s[i].stress not in ["x", "0"]:
            stress_num += 1
    return str(stress_num)


def get_accent_num_before(_i):
    accent_num = 0
    for i in range(len(all_l0s)):
        if all_l0s[i].atl3.in_sen == all_lps[_i].atl3.in_sen and all_l0s[i].in_l3_f <= all_lps[_i].atl0.in_l3_f and all_l0s[i].accent not in ["x", "0"]:
            accent_num += 1
    return str(accent_num)


def get_accent_num_after(_i):
    accent_num = 0
    for i in range(len(all_l0s)):
        if all_l0s[i].atl3.in_sen == all_lps[_i].atl3.in_sen and all_l0s[i].in_l3_f <= all_lps[_i].atl0.in_l3_f and all_l0s[i].accent not in ["x", "0"]:
            accent_num += 1
    return str(accent_num)


def get_stress_distance_before(_i):
    stress_position = 0
    begin_index = 0
    for i in range(len(all_l0s)):
        if all_l0s[i].in_sen == all_lps[_i].atl0.in_sen:
            begin_index = i
            break
    for i in range(begin_index - 1, -1, -1):
        if all_l0s[i].atl3.in_sen == all_lps[_i].atl3.in_sen and all_l0s[i].stress not in ["x", "0"]:
            stress_position = i
            break
    if get_stress_num_before(_i) == "0":
        return "0"
    else:
        return str(begin_index - stress_position)


def get_stress_distance_after(_i):
    stress_position = 0
    begin_index = 0
    for i in range(len(all_l0s)):
        if all_l0s[i].in_sen == all_lps[_i].atl0.in_sen:
            begin_index = i
            break
    for i in range(begin_index + 1, len(all_l0s)):
        if all_l0s[i].atl3.in_sen == all_lps[_i].atl3.in_sen and all_l0s[i].stress not in ["x", "0"]:
            stress_position = i
    if get_stress_num_after(_i) == "0":
        return "0"
    else:
        return str(stress_position - begin_index)


def get_accent_distance_before(_i):
    accent_position = 0
    begin_index = 0
    for i in range(len(all_l0s)):
        if all_l0s[i].in_sen == all_lps[_i].atl0.in_sen:
            begin_index = i
            break
    for i in range(begin_index - 1, -1, -1):
        if all_l0s[i].atl3.in_sen == all_lps[_i].atl3.in_sen and all_l0s[i].accent not in ["x", "0"]:
            accent_position = i
            break
    if get_accent_num_before(_i) == "0":
        return "0"
    else:
        return str(begin_index - accent_position)


def get_accent_distance_after(_i):
    accent_position = 0
    begin_index = 0
    for i in range(len(all_l0s)):
        if all_l0s[i].in_sen == all_lps[_i].atl0.in_sen:
            begin_index = i
            break
    for i in range(begin_index + 1, len(all_l0s)):
        if all_l0s[i].atl3.in_sen == all_lps[_i].atl3.in_sen and all_l0s[i].accent not in ["x", "0"]:
            accent_position = i
    if get_accent_num_after(_i) == "0":
        return "0"
    else:
        return str(accent_position - begin_index)
    

def main():
    open_path = r'/home/shaopf/study/BiaoBeiData/BZNSYP/Prosody.txt'
    save_path = r'/home/shaopf/study/BiaoBeiData/BZNSYP/fulllab2'
    mono_path = r'/home/shaopf/study/BiaoBeiData/BZNSYP/monolab'
    open_lines = IO.read(open_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    vowel_path = r'/home/shaopf/study/BiaoBeiData/BZNSYP/Vowel.lst'
    init_vowels(vowel_path)

    for line_index in range(0, len(open_lines), 3):
        print(open_lines[line_index].strip())
        save_fulllab_path = save_path + "/" + open_lines[line_index].strip() + ".lab"
        mono_lab_path = mono_path + "/" + open_lines[line_index].strip() + ".lab"
        phon_line = open_lines[line_index + 2].strip()
        read(phon_line)
        monos = IO.readList(mono_lab_path)
        write(save_fulllab_path, monos)


if __name__ == '__main__':
    main()
    print('Over!')
