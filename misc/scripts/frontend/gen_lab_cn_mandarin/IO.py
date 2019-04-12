# coding=utf-8
#!/usr/bin/python
# Created on 2018-03-22 17:06
# @author: Administrator
# @name: ReadWrite.py
# @info: 

import os

def addLineBreak(_in_list):
    """每一个元素，增加一个换行符"""
    _print_result = []
    for ls in _in_list:
        if (ls != "" and ls[-1] != '\n') or ls == "":
            _print_result.append(ls + "\n")
        else:
            _print_result.append(ls)
    return _print_result

def read(_open_path, _code = None):
    """utf-8,直接读一个列表， 有换行符"""
    cur_code = 'utf-8'
    if _code:
        cur_code = _code
    input_file = open(_open_path, 'r', encoding = cur_code)
    input_lines = input_file.readlines()
    input_file.close()
    return input_lines

def readList(_open_path, _code = None):
    """utf-8， 无换行符"""
    cur_code = 'utf-8'
    if _code:
        cur_code = _code
    input_file = open(_open_path, 'r', encoding = cur_code)
    input_lines = input_file.readlines()
    result = []
    for line in input_lines:
        line = line.strip()
        result.append(line)
    input_file.close()
    return result

def write(_save_path, _save_list):
    output_file = open(_save_path, 'w', encoding = 'utf-8')
    output_file.writelines(addLineBreak(_save_list))
    output_file.close()
    
def addNewFileName(_old_path, _add_str):
    """在原始路径的文件名上增加一个缀，_add_str"""
    dir_name = os.path.dirname(_old_path)
    file_name = os.path.basename(_old_path)
    only_file_name = file_name[0:file_name.find('.')]
    file_ext_name = file_name[file_name.find('.'):len(file_name)]
    return os.path.join(dir_name, only_file_name + _add_str + file_ext_name)

def setNewFileName(_given_path, _new_name_str):
    """给定路径，获取当前目录，然后使用后面的文件名"""
    dir_name = os.path.dirname(_given_path)
    return os.path.join(dir_name, _new_name_str)

def createDir(new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        
def p():
    print("hello, shao")