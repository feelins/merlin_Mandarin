# This Praat script will transfer TextGrid to valid one, because of the
# result of Montreal alignment is vary.
#
# This script is distributed under the GNU General Public License.
# Copyright 2019.04.10 feelins[feipengshao@163.com]

init_textgrid_path$ = ""
save_textgrid_path$ = ""

createDirectory: save_textgrid_path$

Create Strings as file list: "fileList", init_textgrid_path$ + "\*.TextGrid"
fileNum = Get number of strings
for file from 1 to fileNum
	selectObject: "Strings fileList"
	fileName$ = Get string: file
	textgrid_path$ = init_textgrid_path$ + "\" + fileName$
	Read from file: textgrid_path$
	simpleName$ = selected$("TextGrid", 1)
	selectObject: "TextGrid " + simpleName$
	
	# Process the first interval
	intervalNum2 = Get number of intervals: 2
	label01$ = Get label of interval: 1, 1
	label02$ = Get label of interval: 2, 1
	if label01$ = "" and label02$ = ""
		appendInfoLine: fileName$ + "the first interval of tier1 and tier2 are all null"
	elif label01$ <> "" and label02$ = ""
		appendInfoLine: fileName$ + "the first interval of tier1 is not null, while tier2 is"
	else
		if label02$ <> "sil"
			tmpStart = Get start time of interval: 2, 1
			tmpEnd = Get end time of interval: 2, 1
			tmpDuration = tmpEnd - tmpStart
			if tmpDuration < 0.01
				appendInfoLine: fileName$ + "duration of first interval is too short"
			else
				Insert boundary: 2, 0.01
				initLabel02$ = Get label of interval: 2, 1
				Set interval text: 2, 1, "sil"
				Set interval text: 2, 2, initLabel02$
			endif
		endif
	endif
	
	# Process the last interval
	intervalNum1 = Get number of intervals: 1
	intervalNum2 = Get number of intervals: 2
	label01$ = Get label of interval: 1, intervalNum1
	label02$ = Get label of interval: 2, intervalNum2
	if label01$ = "" and label02$ = ""
		before2 = intervalNum2 - 1
		labelBefore2$ = Get label of interval: 2, before2
		if labelBefore2$ = "sp"
			Remove left boundary: 2, intervalNum2
			intervalNum22 = Get number of intervals: 2
			Set interval text: 2, intervalNum22, "sil"
		else
			Set interval text: 2, intervalNum2, "sil"
		endif
	elif label01$ = "" and label02$ = "sp"
	    Set interval text: 2, intervalNum2, "sil"
	else
	    appendInfoLine: fileName$ + "invalid end of tier"
	endif
	selectObject: "TextGrid " + simpleName$
	Remove tier: 1
	Save as text file: save_textgrid_path$ + "\" + fileName$
	selectObject: "TextGrid " + simpleName$
	Remove
endfor
selectObject: "Strings fileList"
Remove
exit Done!