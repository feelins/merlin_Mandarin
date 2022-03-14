#!/bin/csh -f
# ----------------------------------------------------------------- #
#             The Speech Signal Processing Toolkit (SPTK)           #
#             developed by SPTK Working Group                       #
#             http://sp-tk.sourceforge.net/                         #
# ----------------------------------------------------------------- #
#                                                                   #
#  Copyright (c) 1984-2007  Tokyo Institute of Technology           #
#                           Interdisciplinary Graduate School of    #
#                           Science and Engineering                 #
#                                                                   #
#                1996-2017  Nagoya Institute of Technology          #
#                           Department of Computer Science          #
#                                                                   #
# All rights reserved.                                              #
#                                                                   #
# Redistribution and use in source and binary forms, with or        #
# without modification, are permitted provided that the following   #
# conditions are met:                                               #
#                                                                   #
# - Redistributions of source code must retain the above copyright  #
#   notice, this list of conditions and the following disclaimer.   #
# - Redistributions in binary form must reproduce the above         #
#   copyright notice, this list of conditions and the following     #
#   disclaimer in the documentation and/or other materials provided #
#   with the distribution.                                          #
# - Neither the name of the SPTK working group nor the names of its #
#   contributors may be used to endorse or promote products derived #
#   from this software without specific prior written permission.   #
#                                                                   #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND            #
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,       #
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF          #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS #
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,          #
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED   #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,     #
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON #
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY    #
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           #
# POSSIBILITY OF SUCH DAMAGE.                                       #
# ----------------------------------------------------------------- #
#                                                                   #
#                                      Jan. 2005  K. Tokuda         #
#                                      Aug. 2009  A. Saito          #

onintr clean

set path    = ( /usr/local/bin $path )
set sptkver = '3.11'
set cvsid   = '$Id$'


set cmnd	= $0
set cmnd	= $cmnd:t

set file
@ flagfile    = 0

set directory
@ flagdirectory = 0


@ swab		= 0
@ normalization	= 0
@ normalization_all = 0
@ frequency   = 16
set type1	= s
set type2	= $type1
set input = 0
set bit = 16

set _argv
set white
set whitespace = 0
set index

if ($?uid) then
	else
    		if ($?UID) then
        		set uid = $UID
    		else
        		set uid
    		endif
endif

@ i = 0
while ($i < $#argv)
    @ i++
    set whitespace = 0
    if ( "$argv[$i]" == '-n') then
        set _argv = '-n'
    else
        set _argv = `echo $argv[$i] | sed -e 's/ /_/g'`
        if ( "$_argv" != "$argv[$i]" ) then
            @ whitespace = 1
        endif
    endif
	switch ($_argv)
	case -swab:
		set swab = 1
		breaksw
	case -s:
		@ i++
		set frequency = $argv[$i]
        set cond = `echo "$frequency > 192" | bc`
        if ( $cond == 1 ) then
            echo2 "$cmnd : sampling frequency exceeds 192 kHz!"
        endif
        set cond = `echo "$frequency > 2147483647" | bc`
        if ( $cond == 1 ) then
            echo2 "$cmnd : sampling frequency is too large!"
            exit 1
        endif
		breaksw
    case +c:
        if ($input) then
            set type2 = c
            set bit = 8
        else
            set type1 = c
            set input = 1
        endif
        breaksw
    case +s:
        if ($input) then
            set type2 = s
            set bit = 16
        else
            set type1 = s
            set input = 1
        endif
        breaksw
    case +i3:
        if ($input) then
            set type2 = i3
            set bit = 24
        else
            set type1 = i3
            set input = 1
        endif
        breaksw
    case +i:
        if ($input) then
            set type2 = i
            set bit = 32
        else
            set type1 = i
            set input = 1
        endif
        breaksw
    case +l:
        if ($input) then
            set type2 = l
            set bit = 32
        else
            set type1 = l
            set input = 1
        endif
        breaksw
    case +le:
        if ($input) then
            set type2 = le
            set bit = 64
        else
            set type1 = le
            set input = 1
        endif
        breaksw
    case +C:
        if ($input) then
            set type2 = C
            set bit = 8
        else
            set type1 = C
            set input = 1
        endif
        breaksw
    case +S:
        if ($input) then
            set type2 = S
            set bit = 16
        else
            set type1 = S
            set input = 1
        endif
        breaksw
    case +I3
        if ($input) then
            set type2 = I3
            set bit = 24
        else
            set type1 = I3
            set input = 1
        endif
        breaksw
    case +I:
        if ($input) then
            set type2 = I
            set bit = 32
        else
            set type1 = I
            set input = 1
        endif
        breaksw
    case +L:
        if ($input) then
            set type2 = L
            set bit = 32
        else
            set type1 = L
            set input = 1
        endif
        breaksw
    case +LE:
        if ($input) then
            set type2 = LE
            set bit = 64
        else
            set type1 = LE
            set input = 1
        endif
        breaksw
    case +f:
        if ($input) then
            set type2 = f
            set bit = 32
        else
            set type1 = f
            set input = 1
        endif
        breaksw
    case +d:
        if ($input) then
            set type2 = d
            set bit = 64
        else
            set type1 = d
            set input = 1
        endif
        breaksw
    case +de:
        if ($input) then
            set type2 = de
            set bit = 96
        else
            set type1 = de
            set input = 1
        endif
        breaksw
	case -d:
		@ i++
		set directory = $argv[$i]
		set flagdirectory = 1
		if ( ! -d $argv[$i] ) then
			echo2 "${cmnd}: Can't find directory "'"'"$directory"'"'" \!"
			set exit_status = 1
			goto usage
		endif
		breaksw
	case -n:
		@ normalization = 1
		breaksw
	case -N:
		@ normalization = 1
		@ normalization_all = 1
		breaksw
	case -h:
		set exit_status = 0
		goto usage
		breaksw
	default:
		set file = ( $file $_argv )
		set white = ( $white $whitespace )
		set index = ( $index $i )
		set flagfile = 1
		if ( $whitespace == 1 ) then
			if ( ! -f "$argv[$i]" ) then
				echo2 "${cmnd}: Can't open file "'"'"$argv[$i]"'"'" \!"
				set exit_status = 1
			goto usage
			endif
		endif
		breaksw
	endsw
end

goto main

usage:
	echo2 ''
	echo2 " $cmnd - raw to wav (RIFF)"
	echo2 ''
	echo2 '  usage:'
	echo2 "       $cmnd [ options ] [ infile(s) ]"
	echo2 '  options:'
	echo2 '       -swab         : change endian                [FALSE]'
	echo2 '       -s s          : sampling frequency (kHz)     [16.0]'
	echo2 '       -d d          : destination directory        [N/A]'
	echo2 '       -n            : normalization                   '
        echo2 '                       with the maximum value'
        echo2 '                       if max >= 32767              [FALSE]'
	echo2 '       -N            : normalization                   '
        echo2 '                       with the maximum value       [FALSE]'
	echo2 '       +type1         : input data type               [s]'
	echo2 '       +type2         : output data type              [type1]'
        echo2 '                        c  (char)         C  (unsigned char) '
        echo2 '                        s  (short)        S  (unsigned short)'
        echo2 '                        i3 (int, 3byte)   I3 (unsigned int, 3byte)'
        echo2 '                        i  (int)          I  (unsigned int)'
        echo2 '                        l  (long)         L  (unsigned long)'
        echo2 '                        le (long long)    LE (unsigned long long)'
        echo2 '                        f  (float)        d  (double)'
        echo2 '                        de (long double)'
	echo2 '       -h            : print this message'
	echo2 '  infile(s):'
	echo2 '       waveform                                     [N/A]'
	echo2 '  output:'
	echo2 "       $cmnd attaches RIFF header(s) to input raw file(s)."
	echo2 '                                                         '
	echo2 '       The outfile has an extention ".wav", e.g.,'
	echo2 '          sample.m15 ---> sample.m15.wav'
	echo2 '                                                         '
	echo2 '       If the infile has an extention ".raw",'
	echo2 '       the extention is removed, e.g.,'
	echo2 '          sample.m15.raw ---> sample.m15.wav'
	echo2 '                                                         '
	echo2 '       The outfile is stored in the same directory'
	echo2 '       as the infile.'
	echo2 '       However, once a destination directory is specified,'
	echo2 '       all wav files are stored in that directory.'
	echo2 ''
        echo2 " SPTK: version $sptkver"
        echo2 " CVS Info: $cvsid"
        echo2 ''
exit $exit_status

main:

@ i = 1
while ( $i <= $#file )
   set f = $file[$i]
   if ( $white[$i] ) then
      /bin/ln -f -s "$argv[$index[$i]]" $f
   endif

   if ( $normalization ) then
      if ( $swab ) then
         swab +$type1 < $f |\
         x2x +{$type1}f > /tmp/sptk-raw2wav-$uid-$$.tmp
      else
         x2x +{$type1}f < $f > /tmp/sptk-raw2wav-$uid-$$.tmp
      endif
      set max = `minmax < /tmp/sptk-raw2wav-$uid-$$.tmp | sopr -ABS | minmax | bcut -s 1 | x2x +fa %.100f`
      if ( $normalization_all || `echo "$max < 0" | bc -l` ) then
         sopr -m 32767 -d $max < /tmp/sptk-raw2wav-$uid-$$.tmp |\
         x2x +f{$type2} > /tmp/sptk-raw2wav-$uid-$$.raw
      else
         x2x +f{$type2} < /tmp/sptk-raw2wav-$uid-$$.tmp > /tmp/sptk-raw2wav-$uid-$$.raw
      endif
   else
      if ( $swab ) then
         swab +$type1 < $f |\
         x2x +{$type1}{$type2} > /tmp/sptk-raw2wav-$uid-$$.raw
      else
         x2x +{$type1}{$type2} < $f > /tmp/sptk-raw2wav-$uid-$$.raw
      endif
   endif

   if ( $f:e == "raw" ) then
      if ( $flagdirectory ) then
         set outfile = $directory/"$argv[$index[$i]]:t:r".wav
      else
         set outfile = "$argv[$index[$i]]:r".wav
      endif
   else
      if ( $flagdirectory ) then
         set outfile = $directory/"$argv[$index[$i]]:t".wav
      else
         set outfile = "$argv[$index[$i]]".wav
      endif
   endif

   set frequency = `echo "$frequency * 1000" | bc`
   rawtowav $frequency $bit /tmp/sptk-raw2wav-$uid-$$.raw $outfile
   if ( $white[$i] ) then
      /bin/rm -f $f
   endif
   @ i++

   rm -fr /tmp/sptk-raw2wav-$uid-$$.raw
end

clean:
rm -fr /tmp/sptk-raw2wav-$uid-$$.raw /tmp/sptk-raw2wav-$uid-$$.tmp
