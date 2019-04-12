#!usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@author:shaopf
@file: gen_hanlp_result.py 
@version:
@time: 2019/04/10 13:48:17
@email:feipengshao@163.com
@function： transfer Biaobei prosody result to HanLP result
########## Biaobei Data ##########
000001	卡尔普#2陪外孙#1玩滑梯#4。
	ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
000002	假语村言#2别再#1拥抱我#4。
	jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
########## HanLp Data ###########
hanzi_file:
000001|卡 尔 普* 陪 外 孙# 玩 滑 梯。%
000002|假 语 村 言* 别 再# 拥 抱 我。%
pinyin_file:
000001|ka2 er2 pu3* pei2 wai4 sun1# wan2 hua2 ti1%
000002|jia2 yu3 cun1 yan2* bie2 zai4# yong1 bao4 wo3%
"""

import IO


def getAllWords(word_str):
    """句子字符串转化为字的集合"""
    pre_func = ['“', '（']
    lat_func = ['”', '。', '，', '1', '2', '3', '4', '、', '？', '：', '！', '…', '—', '）', '；']
    en_letters = ['Ｐ', 'Ｂ']
    word_str = word_str.replace('#', '')
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
            if file_name in er_errors:
                if j < len(word_str) and word_str[j] == '儿':
                    tmp_word += word_str[j]
                    j = j + 1
                    while j < len(word_str) and word_str[j] in lat_func:
                        tmp_word += word_str[j]
                        j = j + 1
            words.append(tmp_word)
            tmp_word = ''
    return words


input_list = IO.readList(r'/home/shaopf/study/BiaoBeiData/BZNSYP/ProsodyLabeling/000001-010000_修复儿化字音一致.txt')
save_hanzi_file = r'/home/shaopf/study/BiaoBeiData/BZNSYP/ProsodyLabeling/biaobei_hanzi.txt'
save_pinyin_file = r'/home/shaopf/study/BiaoBeiData/BZNSYP/ProsodyLabeling/biaobei_pinyin.txt'

results_hanzi = []
results_pinyin = []
prosodys = {'1': '#', '2': '*', '3': '$', '4': '%'}

er_errors = IO.readList(r'/home/shaopf/study/BiaoBeiData/BZNSYP/errorfile.txt')

for i in range(0, len(input_list), 2):
    sarray = input_list[i].split()
    file_name = sarray[0]
    print(file_name)
    sentence_content = sarray[1]
    words_list = getAllWords(sentence_content)
    pinyins = input_list[i + 1].split()
    assert len(words_list) == len(pinyins)
    new_pinyins = []
    new_words = []
    for j in range(len(words_list)):
        prosody = ''
        tmp_word = words_list[j]
        if tmp_word.find('1') != -1:
            prosody = prosodys['1']
            tmp_word = tmp_word.replace('1', '')
        if tmp_word.find('2') != -1:
            prosody = prosodys['2']
            tmp_word = tmp_word.replace('2', '')
        if tmp_word.find('3') != -1:
            prosody = prosodys['3']
            tmp_word = tmp_word.replace('3', '')
        if tmp_word.find('4') != -1:
            prosody = prosodys['4']
            tmp_word = tmp_word.replace('4', '')
        new_pinyins.append(pinyins[j] + prosody)
        new_words.append(tmp_word + prosody)

    # output
    results_hanzi.append(file_name + '|' + ' '.join(new_words))
    results_pinyin.append(file_name + '|' + ' '.join(new_pinyins))
IO.write(save_hanzi_file, results_hanzi)
IO.write(save_pinyin_file, results_pinyin)
print('Done!')
