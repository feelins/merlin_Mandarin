#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time         : 2018-02-28  09:40
# @Author       : pfshao
# @File         : TextGrid.py
# @Description  :

def process_line(cur_str):
    """Get value from each line of TextGridline, as {xmax = 0.29} will get {0.29}, {text = "kk"} will get {kk} """
    tmp_array = cur_str.strip().split('=')
    tmp_array = [item for item in filter(lambda x: x != '', tmp_array)]
    return tmp_array[-1].strip()
    
    
class Interval(object):
    """each interval of TextGrid"""
    def __init__(self, _index, _minTime, _maxTime, _name):
        """index from 1 to LEN, begin time, end Time and label of Interval"""
        self.index = _index
        self.begin = _minTime
        self.end = _maxTime
        self.name = _name.strip('"')


class Tier(object):
    """each Tier Level of TextGrid"""
    def __init__(self, _list = None, _newTier_name = None, _newTiers = None):
        """tier name as name, intervals_num as total intervals of each tier, intervals as the list"""
        if _list:
            self.list = _list
            self.name = process_line(self.list[2])
            self.intervals_num = int(process_line(self.list[5]))
            self.intervals = self.get_cur_intervals()
        else:
            self.name = _newTier_name
            self.intervals_num = len(_newTiers)
            self.intervals = _newTiers


    def get_cur_intervals(self):
        """get all intervals of current tier"""
        result = {}
        for i in range(1, self.intervals_num + 1):
            tmp_string = "        intervals [" + str(i) + "]:\n"
            tmp_index = self.list.index(tmp_string)
            if (tmp_index + 3) >= len(self.list):
                print("Error")
            tmp_interval = Interval((i),
                                    process_line(self.list[tmp_index + 1]),
                                    process_line(self.list[tmp_index + 2]),
                                    process_line(self.list[tmp_index + 3]))
            result[i] = tmp_interval
        return result
    
    def get_interval(self, refTime):
        """get interval from refTime, return the index of this interval"""
        _refTime = float(refTime)
        result = -1
        for i in range(1, self.intervals_num + 1):
            _curTime_begin = float(self.intervals[i].begin)
            _curTime_end = float(self.intervals[i].end)
            if _refTime >= _curTime_begin and _refTime < _curTime_end:
                result = i
                break
        if abs(_refTime - float(self.intervals[self.intervals_num].end)) < 0.0005 or (_refTime > float(self.intervals[self.intervals_num].begin) and _refTime < float(self.intervals[self.intervals_num].end)):
            result = self.intervals_num
        return result


class TextGrid(object):
    def __init__(self, _filePath):
        self.filePath = _filePath

    def read(self) -> None:
        """tier number, total duration, all tiers list, head means the first 8 lines of TextGrid File"""
        self.list = self.read_list()
        self.tier_number = int(self.get_tier_number())
        self.total_duration = float(self.get_total_duration())
        self.tiers = self.get_tiers()
        self.head = self.get_head()

    def read_list(self):
        infile = open(self.filePath, "r", encoding="utf-8")
        infile_lines = []
        try:
            infile_lines = infile.readlines()
        except UnicodeDecodeError as e:
            print(str(e) + ',' + self.filePath)
            exit(0)
        infile.close()
        return infile_lines

    def get_tier_number(self):
        return process_line(self.list[6])

    def get_total_duration(self):
        return process_line(self.list[4])
    
    def get_head(self, new_data = None):
        """TextGrid头部分，前7行，新TextGrid有参数生成，需要两个参数，一个是层数new_data[0]，一个是总时长new_data[1]"""
        output_list = []
        if new_data:
            output_list.append('File type = "ooTextFile"\n')
            output_list.append('Object class = "TextGrid"\n')
            output_list.append('\n')
            output_list.append('xmin = 0 \n')
            output_list.append('xmax = ' + new_data[1] + ' \n')
            output_list.append('tiers? <exists> \n')
            output_list.append('size = ' + new_data[0] + ' \n')
            output_list.append('item []: \n')
        else:
            output_list = self.list[0:8]
        return output_list

    def get_tiers(self):
        result = {}
        for i in range(1, int(self.tier_number) + 1):
            a = self.list.index("    item [" + str(i) + "]:\n")
            b = len(self.list)
            if i != int(self.tier_number):
                b = self.list.index("    item [" + str(i + 1) + "]:\n")
            cur_tier = Tier(self.list[a:b])
            result[i] = cur_tier
        return result

    def write(self, refTextGrid) -> None:
        """write to a new TextGrid from old TextGrid"""
        output_list = []
        output_list.extend(refTextGrid.head)
        for i in range(1, int(refTextGrid.tier_number) + 1):
            output_list.append("    item [" + str(i) + "]:\n")
            output_list.append('        class = "IntervalTier"\n')
            output_list.append('        name = ' + refTextGrid.tiers[i].name + '\n')
            output_list.append('        xmin = 0\n')
            output_list.append('        xmax = ' + str(refTextGrid.total_duration) + '\n')
            output_list.append('        intervals: size = ' + str(refTextGrid.tiers[i].intervals_num) + '\n')
            for j in range(1, refTextGrid.tiers[i].intervals_num + 1):
                output_list.append("        intervals [" + str(j) + "]:\n")
                output_list.append("            xmin = " + str(refTextGrid.tiers[i].intervals[j].begin) + "\n")
                output_list.append("            xmax = " + str(refTextGrid.tiers[i].intervals[j].end) + '\n')
                output_list.append('            text = "' + refTextGrid.tiers[i].intervals[j].name + '"\n')
        open(self.filePath, 'w', encoding='utf-8').writelines(output_list)
        
    def write_new(self, tier_number, total_duration, out_tiers):
        """write to a new TextGrid with ALL New list，全新生成一个TextGrid"""
        output_list = []
        output_list.extend(self.get_head([str(tier_number), str(total_duration)]))
        for i in range(1, tier_number + 1):
            tmp_tiers = {}
            try:
                tmp_tiers = out_tiers[i]
            except KeyError as e:
                print(e + "\t" + "没有" + i + "层")
                exit(0)
            else:
                output_list.append("    item [" + str(i) + "]:\n")
                output_list.append('        class = "IntervalTier"\n')
                output_list.append('        name = "' + tmp_tiers.name + '"\n')
                output_list.append('        xmin = 0\n')
                output_list.append('        xmax = ' + str(total_duration) + '\n')
                output_list.append('        intervals: size = ' + str(tmp_tiers.intervals_num) + '\n')
                for j in range(1, tmp_tiers.intervals_num + 1):
                    tmp_interval = []
                    try:
                        tmp_interval = tmp_tiers.intervals[j]
                    except IndexError as e:
                        print(e + "\t" + "数组索引超出，第" + str(j) + "个")
                        exit(0)
                    else:
                        output_list.append("        intervals [" + str(tmp_interval.index) + "]:\n")
                        output_list.append("            xmin = " + str(tmp_interval.begin) + "\n")
                        output_list.append("            xmax = " + str(tmp_interval.end) + '\n')
                        output_list.append('            text = "' + tmp_interval.name + '"\n')
        fw = open(self.filePath, 'w', encoding='utf-8')
        fw.writelines(output_list)
        fw.close()
