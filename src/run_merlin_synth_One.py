################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://github.com/CSTR-Edinburgh/merlin
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################

import pickle
import os, sys, errno
import re
import datetime
import textwrap

import subprocess
import multiprocessing
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing.dummy import Pool as ThrePool
import scipy.stats
import bandmat as bm
import bandmat.linalg as bla

#  numpy & theano imports need to be done in this order (only for some numpy installations, not sure why)
import numpy as np
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano
import theano

import logging # as logging
import logging.config

from shutil import copyfile



############################paths
merlin_dir = r'/home/shaopf/study/merlin_Mandarin'
project_dir = os.path.join(merlin_dir, 'egs/mandarin_voice/s1')
question_file_name = os.path.join(merlin_dir, 'misc/questions/questions_mandarin.dnn.hed')
duration_work_dir = os.path.join(project_dir, 'experiments/mandarin_voice/duration_model')
acoustic_work_dir = os.path.join(project_dir, 'experiments/mandarin_voice/acoustic_model')
duration_inter_dir = os.path.join(duration_work_dir, 'inter_module')
acoustic_inter_dir = os.path.join(acoustic_work_dir, 'inter_module')
duration_model_dir = os.path.join(duration_work_dir, 'nnets_model')
acoustic_model_dir = os.path.join(acoustic_work_dir, 'nnets_model')
test_synth_dir = os.path.join(project_dir, 'experiments/mandarin_voice/test_synthesis')
prompt_label_dir = os.path.join(test_synth_dir, 'prompt-lab')
if (not os.path.exists(test_synth_dir)) or (not os.path.exists(prompt_label_dir)):
    print('There is no test_synthesis dir or prompt-lab dir')
    exit(0)
file_name_scp = []
for line in os.listdir(prompt_label_dir):
  file_name_scp.append(line[:line.find('.')])
gen_dur_label_dir = os.path.join(test_synth_dir, 'gen-lab')
if (not os.path.exists(gen_dur_label_dir)):
    os.mkdir(gen_dur_label_dir)
gen_para_dir = os.path.join(test_synth_dir, 'gen-params')
if (not os.path.exists(gen_para_dir)):
    os.mkdir(gen_para_dir)
gen_wav_dir = os.path.join(test_synth_dir, 'gen-wav')
if (not os.path.exists(gen_wav_dir)):
    os.mkdir(gen_wav_dir)
########################### models ###########################
dur_model_file = os.path.join(duration_model_dir, 'feed_forward_6_tanh.model')
acou_model_file = os.path.join(acoustic_model_dir, 'feed_forward_6_tanh.model')
dur_info_norm_file = os.path.join(duration_inter_dir, 'norm_info__dur_1_MVN.dat')
acou_info_norm_file = os.path.join(acoustic_inter_dir, 'norm_info__mgc_lf0_vuv_bap_199_MVN.dat')
dur_label_norm_file = os.path.join(duration_inter_dir, 'label_norm_HTS_423.dat')
acou_label_norm_file = os.path.join(acoustic_inter_dir, 'label_norm_HTS_427.dat')
dur_var_file_dict = {}
dur_var_file_dict['dur'] = os.path.join(duration_inter_dir, 'var/dur_1')
acou_var_file_dict = {}
acou_var_file_dict['bap'] = os.path.join(acoustic_inter_dir, 'var/bap_15')
acou_var_file_dict['lf0'] = os.path.join(acoustic_inter_dir, 'var/lf0_3')
acou_var_file_dict['mgc'] = os.path.join(acoustic_inter_dir, 'var/mgc_180')
acou_var_file_dict['vuv'] = os.path.join(acoustic_inter_dir, 'var/vuv_1')
########################### paths ###########################
output_features_dur = ['dur']
output_features_acou = ['mgc', 'lf0', 'vuv', 'bap']
output_features_trans = {'mgc':'dmgc', 'lf0':'dlf0', 'vuv':'vuv', 'bap':'dbap'}
output_features_acou_dim = ['dmgc', 'dlf0', 'vuv', 'dbap']
gen_wav_features = ['mgc', 'lf0', 'bap']
output_feature_normalisation = 'MVN'
add_frame_features = False
subphone_feats_dur = 'none'
subphone_feats_acoustic = 'coarse_coding'
additional_features = {}
label_align_style = 'HTS'
label_align_type = 'phone_align'
silence_pattern = ['*-sil+*']
GenTestList = True
enforce_silence = False
FAST_MLPG = True
do_MLPG = True
model_file_name = 'feed_forward_6_tanh'
acoustic_dim = 199
dims = {'mgc':60,
            'dmgc':60 * 3,
            'vuv':1,
            'lf0':1,
            'dlf0':1 * 3,
            'bap':5,
            'dbap':5 * 3,
            'cmp':(60 * 3) + 1 + (1 * 3) + (25 * 3),
            'dur':1
            }
output_dims = {
            'mgc':60 * 3,
            'vuv':1,
            'lf0':1 * 3,
            'bap':5 * 3
            }
rnn_batch_training = False
file_extension_dict = {}
file_extension_dict['mgc'] = '.mgc'
file_extension_dict['lf0'] = '.lf0'
file_extension_dict['bap'] = '.bap'
file_extension_dict['dur'] = '.dur'
file_extension_dict['vuv'] = '.vuv'
file_extension_dict['sp'] = '.sp'


############################tools for generate wav
sptk_bindir = r'/usr/local/SPTK'
SPTK = {
            'X2X': os.path.join(sptk_bindir, 'x2x'),
            'MERGE': os.path.join(sptk_bindir, 'merge'),
            'BCP': os.path.join(sptk_bindir, 'bcp'),
            'MLPG': os.path.join(sptk_bindir, 'mlpg'),
            'MGC2SP': os.path.join(sptk_bindir, 'mgc2sp'),
            'VSUM': os.path.join(sptk_bindir, 'vsum'),
            'VSTAT': os.path.join(sptk_bindir, 'vstat'),
            'SOPR': os.path.join(sptk_bindir, 'sopr'),
            'VOPR': os.path.join(sptk_bindir, 'vopr'),
            'FREQT': os.path.join(sptk_bindir, 'freqt'),
            'C2ACR': os.path.join(sptk_bindir, 'c2acr'),
            'MC2B': os.path.join(sptk_bindir, 'mc2b'),
            'B2MC': os.path.join(sptk_bindir, 'b2mc')
        }
world_bindir = r'/home/shaopf/study/merlin_Mandarin/tools/bin/WORLD'
WORLD = {
            'SYNTHESIS'     : os.path.join(world_bindir, 'synth'),
            'ANALYSIS'      : os.path.join(world_bindir, 'analysis'),
            }

############################### gen wav parameters
pf_coef = 1.4
fw_alpha = 0.77
sr = 48000
co_coef = 1023
fl = 2048
apply_GV = False
shift = 5
do_post_filtering = True

###########################################


class   BinaryIOCollection(object):

    def load_binary_file(self, file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        features = features[:(dimension * (features.size // dimension))]
        features = features.reshape((-1, dimension))

        return features

    def array_to_binary_file(self, data, output_file_name):
        data = numpy.array(data, 'float32')

        fid = open(output_file_name, 'wb')
        data.tofile(fid)
        fid.close()

    def load_binary_file_frame(self, file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        frame_number = features.size // dimension
        features = features[:(dimension * frame_number)]
        features = features.reshape((-1, dimension))

        return features, frame_number


#######################################

class LinguisticBase(object):
    def __init__(self, dimension=0):
        self.dimension = dimension  ##the feature dimensionality of output (should that read 'input' ?)

        ## the number of utterances to be normalised
        self.utterance_num = 0

    ## the ori_file_list contains the file paths of the raw linguistic data
    ## the output_file_list contains the file paths of the normalised linguistic data
    ##
    def perform_normalisation(self, ori_file_list, output_file_list, label_type="state_align", dur_file_list=None):

        logger = logging.getLogger("perform_norm")
        logger.info('perform linguistic feature extraction')
        self.utterance_num = len(ori_file_list)
        if self.utterance_num != len(output_file_list):
            logger.error('the number of input and output linguistic files should be the same!\n')
            sys.exit(1)

        def _perform_normalisation(i):
            if not dur_file_list:
                self.extract_linguistic_features(ori_file_list[i], output_file_list[i], label_type)
            else:
                self.extract_linguistic_features(ori_file_list[i], output_file_list[i], label_type, dur_file_list[i])

        pool = Pool()
        pool.map(_perform_normalisation, range(self.utterance_num))
        pool.close()
        pool.join()

    ## the exact function to do the work
    ## need to be implemented in the specific class
    ## the function will write the linguistic features directly to the output file
    def extract_linguistic_features(self, in_file_name, out_file_name, label_type, dur_file_name=None):
        pass


class LabelNormalisation(LinguisticBase):

    # this class only knows how to deal with a single style of labels (XML or HTS)
    # (to deal with composite labels, use LabelComposer instead)

    def __init__(self, question_file_name=None, xpath_file_name=None):
        pass

    def extract_linguistic_features(self, in_file_name, out_file_name=None, label_type="state_align",
                                    dur_file_name=None):
        if label_type == "phone_align":
            A = self.load_labels_with_phone_alignment(in_file_name, dur_file_name)
        elif label_type == "state_align":
            A = self.load_labels_with_state_alignment(in_file_name)
        else:
            logger.critical("we don't support %s labels as of now!!" % (label_type))

        if out_file_name:
            io_funcs = BinaryIOCollection()
            io_funcs.array_to_binary_file(A, out_file_name)
        else:
            return A


#  -----------------------------


class HTSLabelNormalisation(LabelNormalisation):
    """This class is to convert HTS format labels into continous or binary values, and store as binary format with float32 precision.
    The class supports two kinds of questions: QS and CQS.
        **QS**: is the same as that used in HTS

        **CQS**: is the new defined question in the system.  Here is an example of the question: CQS C-Syl-Tone {_(\d+)+}. regular expression is used for continous values.
    Time alignments are expected in the HTS labels. Here is an example of the HTS labels:
    3050000 3100000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[2]
    3100000 3150000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[3]
    3150000 3250000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[4]
    3250000 3350000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[5]
    3350000 3900000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[6]
    305000 310000 are the starting and ending time.
    [2], [3], [4], [5], [6] mean the HMM state index.
    """

    # this subclass support HTS labels, which include time alignments

    def __init__(self, question_file_name=None, add_frame_features=True, subphone_feats='full', continuous_flag=True):

        logger = logging.getLogger("labels")

        self.question_dict = {}
        self.ori_question_dict = {}
        self.dict_size = 0
        self.continuous_flag = continuous_flag
        try:
            #            self.question_dict, self.ori_question_dict = self.load_question_set(question_file_name)
            self.discrete_dict, self.continuous_dict = self.load_question_set_continous(question_file_name)
        except:
            logger.critical('error whilst loading HTS question set')
            raise

        ###self.dict_size = len(self.question_dict)

        self.dict_size = len(self.discrete_dict) + len(self.continuous_dict)
        self.add_frame_features = add_frame_features
        self.subphone_feats = subphone_feats

        if self.subphone_feats == 'full':
            self.frame_feature_size = 9  ## zhizheng's original 5 state features + 4 phoneme features
        elif self.subphone_feats == 'minimal_frame':
            self.frame_feature_size = 2  ## the minimal features necessary to go from a state-level to frame-level model
        elif self.subphone_feats == 'state_only':
            self.frame_feature_size = 1  ## this is equivalent to a state-based system
        elif self.subphone_feats == 'none':
            self.frame_feature_size = 0  ## the phoneme level features only
        elif self.subphone_feats == 'frame_only':
            self.frame_feature_size = 1  ## this is equivalent to a frame-based system without relying on state-features
        elif self.subphone_feats == 'uniform_state':
            self.frame_feature_size = 2  ## this is equivalent to a frame-based system with uniform state-features
        elif self.subphone_feats == 'minimal_phoneme':
            self.frame_feature_size = 3  ## this is equivalent to a frame-based system with minimal features
        elif self.subphone_feats == 'coarse_coding':
            self.frame_feature_size = 4  ## this is equivalent to a frame-based positioning system reported in Heiga Zen's work
            self.cc_features = self.compute_coarse_coding_features(3)
        else:
            sys.exit('Unknown value for subphone_feats: %s' % (subphone_feats))

        self.dimension = self.dict_size + self.frame_feature_size

        ### if user wants to define their own input, simply set the question set to empty.
        if self.dict_size == 0:
            self.dimension = 0

        logger.debug('HTS-derived input feature dimension is %d + %d = %d' % (
        self.dict_size, self.frame_feature_size, self.dimension))

    def prepare_dur_data(self, ori_file_list, output_file_list, label_type="state_align", feature_type=None,
                         unit_size=None, feat_size=None):
        '''
        extracting duration binary features or numerical features.
        '''
        logger = logging.getLogger("dur")
        utt_number = len(ori_file_list)
        if utt_number != len(output_file_list):
            print("the number of input and output files should be the same!\n");
            sys.exit(1)

        ### set default feature type to numerical, if not assigned ###
        if not feature_type:
            feature_type = "numerical"

        ### set default unit size to state, if not assigned ###
        if not unit_size:
            unit_size = "state"
        if label_type == "phone_align":
            unit_size = "phoneme"

        ### set default feat size to frame or phoneme, if not assigned ###
        if feature_type == "binary":
            if not feat_size:
                feat_size = "frame"
        elif feature_type == "numerical":
            if not feat_size:
                feat_size = "phoneme"
        else:
            logger.critical(
                "Unknown feature type: %s \n Please use one of the following: binary, numerical\n" % (feature_type))
            sys.exit(1)

        for i in range(utt_number):
            self.extract_dur_features(ori_file_list[i], output_file_list[i], label_type, feature_type, unit_size,
                                      feat_size)

    def extract_dur_features(self, in_file_name, out_file_name=None, label_type="state_align", feature_type=None,
                             unit_size=None, feat_size=None):
        logger = logging.getLogger("dur")
        if label_type == "phone_align":
            A = self.extract_dur_from_phone_alignment_labels(in_file_name, feature_type, unit_size, feat_size)
        elif label_type == "state_align":
            A = self.extract_dur_from_state_alignment_labels(in_file_name, feature_type, unit_size, feat_size)
        else:
            logger.critical("we don't support %s labels as of now!!" % (label_type))
            sys.exit(1)

        if out_file_name:
            io_funcs = BinaryIOCollection()
            io_funcs.array_to_binary_file(A, out_file_name)
        else:
            return A

    def extract_dur_from_state_alignment_labels(self, file_name, feature_type, unit_size, feat_size):
        logger = logging.getLogger("dur")

        state_number = 5
        dur_dim = state_number

        if feature_type == "binary":
            dur_feature_matrix = numpy.empty((100000, 1))
        elif feature_type == "numerical":
            if unit_size == "state":
                dur_feature_matrix = numpy.empty((100000, dur_dim))
                current_dur_array = numpy.zeros((dur_dim, 1))
            else:  ## phoneme/syllable/word
                dur_feature_matrix = numpy.empty((100000, 1))

        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()

        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number))

        MLU_dur = [[], [], []]
        list_of_silences = ['#', 'sil', 'pau', 'SIL']
        current_index = 0
        dur_feature_index = 0
        syllable_duration = 0
        word_duration = 0
        for line in utt_labels:
            line = line.strip()

            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])

            full_label = temp_list[2]
            full_label_length = len(full_label) - 3  # remove state information [k]
            state_index = full_label[full_label_length + 1]
            state_index = int(state_index) - 1
            current_phone = full_label[full_label.index('-') + 1:full_label.index('+')]

            frame_number = int(end_time / 50000) - int(start_time / 50000)

            if state_index == 1:
                phone_duration = frame_number

                for i in range(state_number - 1):
                    line = utt_labels[current_index + i + 1].strip()
                    temp_list = re.split('\s+', line)
                    phone_duration += int((int(temp_list[1]) - int(temp_list[0])) / 50000)

                syllable_duration += phone_duration
                word_duration += phone_duration

                ### for syllable and word positional information ###
                label_binary_vector = self.pattern_matching_binary(full_label)
                label_continuous_vector = self.pattern_matching_continous_position(full_label)

                ### syllable ending information ###
                syl_end = 0
                if (label_continuous_vector[0, 1] == 1 or current_phone in list_of_silences):  ##pos-bw and c-silences
                    syl_end = 1

                ### word ending information ###
                word_end = 0
                if (syl_end and label_continuous_vector[0, 9] == 1 or current_phone in list_of_silences):
                    word_end = 1

            if feature_type == "binary":
                current_block_array = numpy.zeros((frame_number, 1))
                if unit_size == "state":
                    current_block_array[-1] = 1
                elif unit_size == "phoneme":
                    if state_index == state_number:
                        current_block_array[-1] = 1
                else:
                    logger.critical(
                        "Unknown unit size: %s \n Please use one of the following: state, phoneme\n" % (unit_size))
                    sys.exit(1)
            elif feature_type == "numerical":
                if unit_size == "state":
                    current_dur_array[current_index % 5] = frame_number
                    if feat_size == "phoneme" and state_index == state_number:
                        current_block_array = current_dur_array.transpose()
                    if feat_size == "frame":
                        current_block_array = numpy.tile(current_dur_array.transpose(), (frame_number, 1))
                elif state_index == state_number:
                    if unit_size == "phoneme":
                        current_block_array = numpy.array([phone_duration])
                    elif unit_size == "syllable":
                        current_block_array = numpy.array([syllable_duration])
                    elif unit_size == "word":
                        current_block_array = numpy.array([word_duration])
                    if syl_end:
                        syllable_duration = 0
                    if word_end:
                        word_duration = 0

            ### writing into dur_feature_matrix ###
            if feat_size == "frame":
                dur_feature_matrix[dur_feature_index:dur_feature_index + frame_number, ] = current_block_array
                dur_feature_index = dur_feature_index + frame_number
            elif state_index == state_number:
                if feat_size == "phoneme":
                    dur_feature_matrix[dur_feature_index:dur_feature_index + 1, ] = current_block_array
                    dur_feature_index = dur_feature_index + 1
                elif current_phone != '#':  ## removing silence here
                    if feat_size == "syllable" and syl_end:
                        dur_feature_matrix[dur_feature_index:dur_feature_index + 1, ] = current_block_array
                        dur_feature_index = dur_feature_index + 1
                    elif feat_size == "word" and word_end:
                        dur_feature_matrix[dur_feature_index:dur_feature_index + 1, ] = current_block_array
                        dur_feature_index = dur_feature_index + 1
                    elif feat_size == "MLU":
                        if word_end:
                            if current_phone == 'pau':
                                MLU_dur[0].append(1)
                            else:
                                MLU_dur[0].append(int(label_continuous_vector[0, 24]))
                        if syl_end:
                            if current_phone == 'pau':
                                MLU_dur[1].append(1)
                            else:
                                MLU_dur[1].append(int(label_continuous_vector[0, 7]))
                        MLU_dur[2].append(int(phone_duration))

            current_index += 1

        if feat_size == "MLU":
            for seg_indx in xrange(len(MLU_dur)):
                seg_len = len(MLU_dur[seg_indx])
                current_block_array = numpy.reshape(numpy.array(MLU_dur[seg_indx]), (-1, 1))
                dur_feature_matrix[dur_feature_index:dur_feature_index + seg_len, ] = current_block_array
                dur_feature_index = dur_feature_index + seg_len

        dur_feature_matrix = dur_feature_matrix[0:dur_feature_index, ]
        logger.debug('made duration matrix of %d frames x %d features' % dur_feature_matrix.shape)
        return dur_feature_matrix

    def extract_dur_from_phone_alignment_labels(self, file_name, feature_type, unit_size, feat_size):
        logger = logging.getLogger("dur")

        dur_dim = 1  # hard coded here

        if feature_type == "binary":
            dur_feature_matrix = numpy.empty((100000, dur_dim))
        elif feature_type == "numerical":
            if unit_size == "phoneme":
                dur_feature_matrix = numpy.empty((100000, dur_dim))

        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()

        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number))

        current_index = 0
        dur_feature_index = 0
        for line in utt_labels:
            line = line.strip()

            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])

            full_label = temp_list[2]

            frame_number = int(end_time / 50000) - int(start_time / 50000)

            phone_duration = frame_number

            if feature_type == "binary":
                current_block_array = numpy.zeros((frame_number, 1))
                if unit_size == "phoneme":
                    current_block_array[-1] = 1
                else:
                    logger.critical("Unknown unit size: %s \n Please use one of the following: phoneme\n" % (unit_size))
                    sys.exit(1)
            elif feature_type == "numerical":
                if unit_size == "phoneme":
                    current_block_array = numpy.array([phone_duration])

            ### writing into dur_feature_matrix ###
            if feat_size == "frame":
                dur_feature_matrix[dur_feature_index:dur_feature_index + frame_number, ] = current_block_array
                dur_feature_index = dur_feature_index + frame_number
            elif feat_size == "phoneme":
                dur_feature_matrix[dur_feature_index:dur_feature_index + 1, ] = current_block_array
                dur_feature_index = dur_feature_index + 1

            current_index += 1

        dur_feature_matrix = dur_feature_matrix[0:dur_feature_index, ]
        logger.debug('made duration matrix of %d frames x %d features' % dur_feature_matrix.shape)
        return dur_feature_matrix

    def load_labels_with_phone_alignment(self, file_name, dur_file_name):

        # this is not currently used ??? -- it works now :D
        logger = logging.getLogger("labels")
        # logger.critical('unused function ???')
        # raise Exception

        if dur_file_name:
            io_funcs = BinaryIOCollection()
            dur_dim = 1  ## hard coded for now
            manual_dur_data = io_funcs.load_binary_file(dur_file_name, dur_dim)

        if self.add_frame_features:
            assert self.dimension == self.dict_size + self.frame_feature_size
        elif self.subphone_feats != 'none':
            assert self.dimension == self.dict_size + self.frame_feature_size
        else:
            assert self.dimension == self.dict_size

        label_feature_matrix = numpy.empty((100000, self.dimension))

        ph_count = 0
        label_feature_index = 0
        with open(file_name) as fid:
            all_data = fid.readlines()
        for line in all_data:
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)

            if len(temp_list) == 1:
                frame_number = 0
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]

                # to do - support different frame shift - currently hardwired to 5msec
                # currently under beta testing: support different frame shift
                if dur_file_name:
                    frame_number = manual_dur_data[ph_count]
                else:
                    frame_number = int(end_time / 50000) - int(start_time / 50000)

                if self.subphone_feats == "coarse_coding":
                    cc_feat_matrix = self.extract_coarse_coding_features_relative(frame_number)

            ph_count = ph_count + 1
            # label_binary_vector = self.pattern_matching(full_label)
            label_binary_vector = self.pattern_matching_binary(full_label)

            # if there is no CQS question, the label_continuous_vector will become to empty
            label_continuous_vector = self.pattern_matching_continous_position(full_label)
            label_vector = numpy.concatenate([label_binary_vector, label_continuous_vector], axis=1)

            if self.add_frame_features:
                current_block_binary_array = numpy.zeros((frame_number, self.dict_size + self.frame_feature_size))
                for i in range(frame_number):
                    current_block_binary_array[i, 0:self.dict_size] = label_vector

                    if self.subphone_feats == 'minimal_phoneme':
                        ## features which distinguish frame position in phoneme
                        current_block_binary_array[i, self.dict_size] = float(i + 1) / float(
                            frame_number)  # fraction through phone forwards
                        current_block_binary_array[i, self.dict_size + 1] = float(frame_number - i) / float(
                            frame_number)  # fraction through phone backwards
                        current_block_binary_array[i, self.dict_size + 2] = float(frame_number)  # phone duration

                    elif self.subphone_feats == 'coarse_coding':
                        ## features which distinguish frame position in phoneme using three continous numerical features
                        current_block_binary_array[i, self.dict_size + 0] = cc_feat_matrix[i, 0]
                        current_block_binary_array[i, self.dict_size + 1] = cc_feat_matrix[i, 1]
                        current_block_binary_array[i, self.dict_size + 2] = cc_feat_matrix[i, 2]
                        current_block_binary_array[i, self.dict_size + 3] = float(frame_number)

                    elif self.subphone_feats == 'none':
                        pass

                    else:
                        sys.exit('unknown subphone_feats type')

                label_feature_matrix[
                label_feature_index:label_feature_index + frame_number, ] = current_block_binary_array
                label_feature_index = label_feature_index + frame_number

            elif self.subphone_feats == 'none':
                current_block_binary_array = label_vector
                label_feature_matrix[label_feature_index:label_feature_index + 1, ] = current_block_binary_array
                label_feature_index = label_feature_index + 1

        label_feature_matrix = label_feature_matrix[0:label_feature_index, ]

        logger.info('loaded %s, %3d labels' % (file_name, ph_count))
        logger.debug('made label matrix of %d frames x %d labels' % label_feature_matrix.shape)
        return label_feature_matrix

    def load_labels_with_state_alignment(self, file_name):
        ## setting add_frame_features to False performs either state/phoneme level normalisation

        logger = logging.getLogger("labels")

        if self.add_frame_features:
            assert self.dimension == self.dict_size + self.frame_feature_size
        elif self.subphone_feats != 'none':
            assert self.dimension == self.dict_size + self.frame_feature_size
        else:
            assert self.dimension == self.dict_size

        # label_feature_matrix = numpy.empty((100000, self.dict_size+self.frame_feature_size))
        label_feature_matrix = numpy.empty((100000, self.dimension))

        label_feature_index = 0

        state_number = 5

        lab_binary_vector = numpy.zeros((1, self.dict_size))
        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()
        current_index = 0
        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number))

        phone_duration = 0
        state_duration_base = 0
        for line in utt_labels:
            line = line.strip()

            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)

            if len(temp_list) == 1:
                frame_number = 0
                state_index = 1
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                frame_number = int(end_time / 50000) - int(start_time / 50000)
                full_label = temp_list[2]

                full_label_length = len(full_label) - 3  # remove state information [k]
                state_index = full_label[full_label_length + 1]

                state_index = int(state_index) - 1
                state_index_backward = 6 - state_index
                full_label = full_label[0:full_label_length]

            if state_index == 1:
                current_frame_number = 0
                phone_duration = frame_number
                state_duration_base = 0

                #                label_binary_vector = self.pattern_matching(full_label)
                label_binary_vector = self.pattern_matching_binary(full_label)

                # if there is no CQS question, the label_continuous_vector will become to empty
                label_continuous_vector = self.pattern_matching_continous_position(full_label)
                label_vector = numpy.concatenate([label_binary_vector, label_continuous_vector], axis=1)

                if len(temp_list) == 1:
                    state_index = state_number
                else:
                    for i in range(state_number - 1):
                        line = utt_labels[current_index + i + 1].strip()
                        temp_list = re.split('\s+', line)
                        phone_duration += int((int(temp_list[1]) - int(temp_list[0])) / 50000)

                    if self.subphone_feats == "coarse_coding":
                        cc_feat_matrix = self.extract_coarse_coding_features_relative(phone_duration)

            if self.add_frame_features:
                current_block_binary_array = numpy.zeros((frame_number, self.dict_size + self.frame_feature_size))
                for i in range(frame_number):
                    current_block_binary_array[i, 0:self.dict_size] = label_vector

                    if self.subphone_feats == 'full':
                        ## Zhizheng's original 9 subphone features:
                        current_block_binary_array[i, self.dict_size] = float(i + 1) / float(
                            frame_number)  ## fraction through state (forwards)
                        current_block_binary_array[i, self.dict_size + 1] = float(frame_number - i) / float(
                            frame_number)  ## fraction through state (backwards)
                        current_block_binary_array[i, self.dict_size + 2] = float(
                            frame_number)  ## length of state in frames
                        current_block_binary_array[i, self.dict_size + 3] = float(
                            state_index)  ## state index (counting forwards)
                        current_block_binary_array[i, self.dict_size + 4] = float(
                            state_index_backward)  ## state index (counting backwards)

                        current_block_binary_array[i, self.dict_size + 5] = float(
                            phone_duration)  ## length of phone in frames
                        current_block_binary_array[i, self.dict_size + 6] = float(frame_number) / float(
                            phone_duration)  ## fraction of the phone made up by current state
                        current_block_binary_array[i, self.dict_size + 7] = float(
                            phone_duration - i - state_duration_base) / float(
                            phone_duration)  ## fraction through phone (backwards)
                        current_block_binary_array[i, self.dict_size + 8] = float(state_duration_base + i + 1) / float(
                            phone_duration)  ## fraction through phone (forwards)

                    elif self.subphone_feats == 'state_only':
                        ## features which only distinguish state:
                        current_block_binary_array[i, self.dict_size] = float(
                            state_index)  ## state index (counting forwards)

                    elif self.subphone_feats == 'frame_only':
                        ## features which distinguish frame position in phoneme:
                        current_frame_number += 1
                        current_block_binary_array[i, self.dict_size] = float(current_frame_number) / float(
                            phone_duration)  ## fraction through phone (counting forwards)

                    elif self.subphone_feats == 'uniform_state':
                        ## features which distinguish frame position in phoneme:
                        current_frame_number += 1
                        current_block_binary_array[i, self.dict_size] = float(current_frame_number) / float(
                            phone_duration)  ## fraction through phone (counting forwards)
                        new_state_index = max(1, round(float(current_frame_number) / float(phone_duration) * 5))
                        current_block_binary_array[i, self.dict_size + 1] = float(
                            new_state_index)  ## state index (counting forwards)

                    elif self.subphone_feats == "coarse_coding":
                        ## features which distinguish frame position in phoneme using three continous numerical features
                        current_block_binary_array[i, self.dict_size + 0] = cc_feat_matrix[current_frame_number, 0]
                        current_block_binary_array[i, self.dict_size + 1] = cc_feat_matrix[current_frame_number, 1]
                        current_block_binary_array[i, self.dict_size + 2] = cc_feat_matrix[current_frame_number, 2]
                        current_block_binary_array[i, self.dict_size + 3] = float(phone_duration)
                        current_frame_number += 1

                    elif self.subphone_feats == 'minimal_frame':
                        ## features which distinguish state and minimally frame position in state:
                        current_block_binary_array[i, self.dict_size] = float(i + 1) / float(
                            frame_number)  ## fraction through state (forwards)
                        current_block_binary_array[i, self.dict_size + 1] = float(
                            state_index)  ## state index (counting forwards)
                    elif self.subphone_feats == 'none':
                        pass
                    else:
                        sys.exit('unknown subphone_feats type')

                label_feature_matrix[
                label_feature_index:label_feature_index + frame_number, ] = current_block_binary_array
                label_feature_index = label_feature_index + frame_number
            elif self.subphone_feats == 'state_only' and state_index == state_number:
                current_block_binary_array = numpy.zeros((state_number, self.dict_size + self.frame_feature_size))
                for i in range(state_number):
                    current_block_binary_array[i, 0:self.dict_size] = label_vector
                    current_block_binary_array[i, self.dict_size] = float(i + 1)  ## state index (counting forwards)
                label_feature_matrix[
                label_feature_index:label_feature_index + state_number, ] = current_block_binary_array
                label_feature_index = label_feature_index + state_number
            elif self.subphone_feats == 'none' and state_index == state_number:
                current_block_binary_array = label_vector
                label_feature_matrix[label_feature_index:label_feature_index + 1, ] = current_block_binary_array
                label_feature_index = label_feature_index + 1

            state_duration_base += frame_number

            current_index += 1

        label_feature_matrix = label_feature_matrix[0:label_feature_index, ]
        logger.debug('made label matrix of %d frames x %d labels' % label_feature_matrix.shape)
        return label_feature_matrix

    def extract_durational_features(self, dur_file_name=None, dur_data=None):

        if dur_file_name:
            io_funcs = BinaryIOCollection()
            dur_dim = 1  ## hard coded for now
            dur_data = io_funcs.load_binary_file(dur_file_name, dur_dim)

        ph_count = len(dur_data)
        total_num_of_frames = int(sum(dur_data))

        duration_feature_array = numpy.zeros((total_num_of_frames, self.frame_feature_size))

        frame_index = 0
        for i in range(ph_count):
            frame_number = int(dur_data[i])
            if self.subphone_feats == "coarse_coding":
                cc_feat_matrix = self.extract_coarse_coding_features_relative(frame_number)

                for j in range(frame_number):
                    duration_feature_array[frame_index, 0] = cc_feat_matrix[j, 0]
                    duration_feature_array[frame_index, 1] = cc_feat_matrix[j, 1]
                    duration_feature_array[frame_index, 2] = cc_feat_matrix[j, 2]
                    duration_feature_array[frame_index, 3] = float(frame_number)
                    frame_index += 1

            elif self.subphone_feats == 'full':
                state_number = 5  # hard coded here
                phone_duration = sum(dur_data[i, :])
                state_duration_base = 0
                for state_index in xrange(1, state_number + 1):
                    state_index_backward = (state_number - state_index) + 1
                    frame_number = int(dur_data[i][state_index - 1])
                    for j in xrange(frame_number):
                        duration_feature_array[frame_index, 0] = float(j + 1) / float(
                            frame_number)  ## fraction through state (forwards)
                        duration_feature_array[frame_index, 1] = float(frame_number - j) / float(
                            frame_number)  ## fraction through state (backwards)
                        duration_feature_array[frame_index, 2] = float(frame_number)  ## length of state in frames
                        duration_feature_array[frame_index, 3] = float(state_index)  ## state index (counting forwards)
                        duration_feature_array[frame_index, 4] = float(
                            state_index_backward)  ## state index (counting backwards)

                        duration_feature_array[frame_index, 5] = float(phone_duration)  ## length of phone in frames
                        duration_feature_array[frame_index, 6] = float(frame_number) / float(
                            phone_duration)  ## fraction of the phone made up by current state
                        duration_feature_array[frame_index, 7] = float(
                            phone_duration - j - state_duration_base) / float(
                            phone_duration)  ## fraction through phone (forwards)
                        duration_feature_array[frame_index, 8] = float(state_duration_base + j + 1) / float(
                            phone_duration)  ## fraction through phone (backwards)
                        frame_index += 1

                    state_duration_base += frame_number

        return duration_feature_array

    def compute_coarse_coding_features(self, num_states):
        assert num_states == 3

        npoints = 600
        cc_features = numpy.zeros((num_states, npoints))

        x1 = numpy.linspace(-1.5, 1.5, npoints)
        x2 = numpy.linspace(-1.0, 2.0, npoints)
        x3 = numpy.linspace(-0.5, 2.5, npoints)

        mu1 = 0.0
        mu2 = 0.5
        mu3 = 1.0

        sigma = 0.4

        # cc_features[0, :] = mlab.normpdf(x1, mu1, sigma)
        # cc_features[1, :] = mlab.normpdf(x2, mu2, sigma)
        # cc_features[2, :] = mlab.normpdf(x3, mu3, sigma)
        cc_features[0, :] = scipy.stats.norm.pdf(x1, mu1, sigma)
        cc_features[1, :] = scipy.stats.norm.pdf(x2, mu2, sigma)
        cc_features[2, :] = scipy.stats.norm.pdf(x3, mu3, sigma)

        return cc_features

    def extract_coarse_coding_features_relative(self, phone_duration):
        dur = int(phone_duration)

        cc_feat_matrix = numpy.zeros((dur, 3))

        for i in range(dur):
            rel_indx = int((200 / float(dur)) * i)
            cc_feat_matrix[i, 0] = self.cc_features[0, 300 + rel_indx]
            cc_feat_matrix[i, 1] = self.cc_features[1, 200 + rel_indx]
            cc_feat_matrix[i, 2] = self.cc_features[2, 100 + rel_indx]

        return cc_feat_matrix

    ### this function is not used now
    def extract_coarse_coding_features_absolute(self, phone_duration):
        dur = int(phone_duration)

        cc_feat_matrix = numpy.zeros((dur, 3))

        npoints1 = (dur * 2) * 10 + 1
        npoints2 = (dur - 1) * 10 + 1
        npoints3 = (2 * dur - 1) * 10 + 1

        x1 = numpy.linspace(-dur, dur, npoints1)
        x2 = numpy.linspace(1, dur, npoints2)
        x3 = numpy.linspace(1, 2 * dur - 1, npoints3)

        mu1 = 0
        mu2 = (1 + dur) / 2
        mu3 = dur
        variance = 1
        sigma = variance * ((dur / 10) + 2)
        sigma1 = sigma
        sigma2 = sigma - 1
        sigma3 = sigma

        # y1 = mlab.normpdf(x1, mu1, sigma1)
        # y2 = mlab.normpdf(x2, mu2, sigma2)
        # y3 = mlab.normpdf(x3, mu3, sigma3)
        y1 = scipy.stats.norm.pdf(x1, mu1, sigma1)
        y2 = scipy.stats.norm.pdf(x2, mu2, sigma2)
        y3 = scipy.stats.norm.pdf(x3, mu3, sigma3)

        for i in range(dur):
            cc_feat_matrix[i, 0] = y1[(dur + 1 + i) * 10]
            cc_feat_matrix[i, 1] = y2[i * 10]
            cc_feat_matrix[i, 2] = y3[i * 10]

        for i in range(3):
            cc_feat_matrix[:, i] = cc_feat_matrix[:, i] / max(cc_feat_matrix[:, i])

        return cc_feat_matrix

    ### this function is not used now
    def pattern_matching(self, label):
        # this function is where most time is spent during label preparation
        #
        # it might be possible to speed it up by using pre-compiled regular expressions?
        # (not trying this now, since we may change to to XML tree format for input instead of HTS labels)
        #
        label_size = len(label)

        lab_binary_vector = numpy.zeros((1, self.dict_size))

        for i in range(self.dict_size):
            current_question_list = self.question_dict[str(i)]
            binary_flag = 0
            for iq in range(len(current_question_list)):
                current_question = current_question_list[iq]
                current_size = len(current_question)
                if current_question[0] == '*' and current_question[current_size - 1] == '*':
                    temp_question = current_question[1:current_size - 1]
                    for il in range(1, label_size - current_size + 2):
                        if temp_question == label[il:il + current_size - 2]:
                            binary_flag = 1
                elif current_question[current_size - 1] != '*':
                    temp_question = current_question[1:current_size]
                    if temp_question == label[label_size - current_size + 1:label_size]:
                        binary_flag = 1
                elif current_question[0] != '*':
                    temp_question = current_question[0:current_size - 1]
                    if temp_question == label[0:current_size - 1]:
                        binary_flag = 1
                if binary_flag == 1:
                    break
            lab_binary_vector[0, i] = binary_flag

        return lab_binary_vector

    def pattern_matching_binary(self, label):

        dict_size = len(self.discrete_dict)
        lab_binary_vector = numpy.zeros((1, dict_size))

        for i in range(dict_size):
            current_question_list = self.discrete_dict[str(i)]
            binary_flag = 0
            for iq in range(len(current_question_list)):
                current_compiled = current_question_list[iq]

                ms = current_compiled.search(label)
                if ms is not None:
                    binary_flag = 1
                    break
            lab_binary_vector[0, i] = binary_flag

        return lab_binary_vector

    def pattern_matching_continous_position(self, label):

        dict_size = len(self.continuous_dict)

        lab_continuous_vector = numpy.zeros((1, dict_size))

        for i in range(dict_size):
            continuous_value = -1.0

            current_compiled = self.continuous_dict[str(i)]

            ms = current_compiled.search(label)
            if ms is not None:
                #                assert len(ms.group()) == 1
                continuous_value = ms.group(1)

            lab_continuous_vector[0, i] = continuous_value

        return lab_continuous_vector

    def load_question_set(self, qs_file_name):
        fid = open(qs_file_name)
        question_index = 0
        question_dict = {}
        ori_question_dict = {}
        for line in fid.readlines():
            line = line.replace('\n', '')
            if len(line) > 5:
                temp_list = line.split('{')
                temp_line = temp_list[1]
                temp_list = temp_line.split('}')
                temp_line = temp_list[0]
                question_list = temp_line.split(',')
                question_dict[str(question_index)] = question_list
                ori_question_dict[str(question_index)] = line
                question_index += 1
        fid.close()

        logger = logging.getLogger("labels")
        logger.debug('loaded question set with %d questions' % len(question_dict))

        return question_dict, ori_question_dict

    def load_question_set_continous(self, qs_file_name):

        logger = logging.getLogger("labels")

        fid = open(qs_file_name)
        binary_qs_index = 0
        continuous_qs_index = 0
        binary_dict = {}
        continuous_dict = {}
        LL = re.compile(re.escape('LL-'))
        LAST_QUESTION = re.compile(re.escape('(\d+)') + '$')  # regex for last question

        for line in fid.readlines():
            line = line.replace('\n', '').replace('\t', ' ')

            if len(line) > 5:
                temp_list = line.split('{')
                temp_line = temp_list[1]
                temp_list = temp_line.split('}')
                temp_line = temp_list[0]
                temp_line = temp_line.strip()
                question_list = temp_line.split(',')

                temp_list = line.split(' ')
                question_key = temp_list[1]
                #                print   line
                if temp_list[0] == 'CQS':
                    assert len(question_list) == 1
                    processed_question = self.wildcards2regex(question_list[0], convert_number_pattern=True)
                    if LAST_QUESTION.search(question_list[0]):
                        processed_question = processed_question + '$'  # last question must only match at end of HTS label string
                    continuous_dict[str(continuous_qs_index)] = re.compile(
                        processed_question)  # save pre-compiled regular expression
                    continuous_qs_index = continuous_qs_index + 1
                elif temp_list[0] == 'QS':
                    re_list = []
                    for temp_question in question_list:
                        processed_question = self.wildcards2regex(temp_question)
                        if LL.search(question_key):
                            processed_question = '^' + processed_question
                        re_list.append(re.compile(processed_question))

                    binary_dict[str(binary_qs_index)] = re_list
                    binary_qs_index = binary_qs_index + 1
                else:
                    logger.critical('The question set is not defined correctly: %s' % (line))
                    raise Exception

        #                question_index = question_index + 1
        return binary_dict, continuous_dict

    def wildcards2regex(self, question, convert_number_pattern=False):
        """
        Convert HTK-style question into regular expression for searching labels.
        If convert_number_pattern, keep the following sequences unescaped for
        extracting continuous values):
            (\d+)       -- handles digit without decimal point
            ([\d\.]+)   -- handles digits with and without decimal point
        """

        ## handle HTK wildcards (and lack of them) at ends of label:
        prefix = ""
        postfix = ""
        if '*' in question:
            if not question.startswith('*'):
                prefix = "\A"
            if not question.endswith('*'):
                postfix = "\Z"
        question = question.strip('*')
        question = re.escape(question)
        ## convert remaining HTK wildcards * and ? to equivalent regex:
        question = question.replace('\\*', '.*')
        question = question.replace('\\?', '.')
        question = prefix + question + postfix

        if convert_number_pattern:
            question = question.replace('\\(\\\\d\\+\\)', '(\d+)')
            question = question.replace('\\(\\[\\\\d\\\\\\.\\]\\+\\)', '([\d\.]+)')
        return question


class HTSDurationLabelNormalisation(HTSLabelNormalisation):
    """
    Unlike HTSLabelNormalisation, HTSDurationLabelNormalisation does not accept timings.
    One line of labels is converted into 1 datapoint, that is, the label is not 'unpacked'
    into frames. HTK state index [\d] is not handled in any special way.
    """

    def __init__(self, question_file_name=None, subphone_feats='full', continuous_flag=True):
        super(HTSDurationLabelNormalisation, self).__init__(question_file_name=question_file_name, \
                                                            subphone_feats=subphone_feats,
                                                            continuous_flag=continuous_flag)
        ## don't use extra features beyond those in questions for duration labels:
        self.dimension = self.dict_size

    def load_labels_with_state_alignment(self, file_name, add_frame_features=False):
        ## add_frame_features not used in HTSLabelNormalisation -- only in XML version

        logger = logging.getLogger("labels")

        assert self.dimension == self.dict_size

        label_feature_matrix = numpy.empty((100000, self.dimension))

        label_feature_index = 0

        lab_binary_vector = numpy.zeros((1, self.dict_size))
        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()
        current_index = 0
        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number))

        ## remove empty lines
        utt_labels = [line for line in utt_labels if line != '']

        for (line_number, line) in enumerate(utt_labels):
            temp_list = re.split('\s+', line.strip())
            full_label = temp_list[-1]  ## take last entry -- ignore timings if present

            label_binary_vector = self.pattern_matching_binary(full_label)

            # if there is no CQS question, the label_continuous_vector will become to empty
            label_continuous_vector = self.pattern_matching_continous_position(full_label)
            label_vector = numpy.concatenate([label_binary_vector, label_continuous_vector], axis=1)

            label_feature_matrix[line_number, :] = label_vector[:]

        label_feature_matrix = label_feature_matrix[:line_number + 1, :]
        logger.debug('made label matrix of %d frames x %d labels' % label_feature_matrix.shape)
        return label_feature_matrix


######################################################################


class SilenceRemover(object):
    def __init__(self, n_cmp, silence_pattern=['*-#+*'], label_type="state_align", remove_frame_features=True,
                 subphone_feats="none"):
        self.silence_pattern = silence_pattern
        self.silence_pattern_size = len(silence_pattern)
        self.label_type = label_type
        self.remove_frame_features = remove_frame_features
        self.subphone_feats = subphone_feats
        self.n_cmp = n_cmp

    def remove_silence(self, in_data_list, in_align_list, out_data_list, dur_file_list=None):
        file_number = len(in_data_list)
        align_file_number = len(in_align_list)

        if file_number != align_file_number:
            print("The number of input and output files does not equal!\n")
            sys.exit(1)
        if file_number != len(out_data_list):
            print("The number of input and output files does not equal!\n")
            sys.exit(1)

        io_funcs = BinaryIOCollection()

        def _remove_silence(i):
            if self.label_type == "phone_align":
                if dur_file_list:
                    dur_file_name = dur_file_list[i]
                else:
                    dur_file_name = None
                nonsilence_indices = self.load_phone_alignment(in_align_list[i], dur_file_name)
            else:
                nonsilence_indices = self.load_alignment(in_align_list[i])

            ori_cmp_data = io_funcs.load_binary_file(in_data_list[i], self.n_cmp)

            frame_number = ori_cmp_data.size / self.n_cmp

            if len(nonsilence_indices) == frame_number:
                print('WARNING: no silence found!')
                # previsouly: continue -- in fact we should keep non-silent data!

            ## if labels have a few extra frames than audio, this can break the indexing, remove them:
            nonsilence_indices = [ix for ix in nonsilence_indices if ix < frame_number]

            new_cmp_data = ori_cmp_data[nonsilence_indices,]

            io_funcs.array_to_binary_file(new_cmp_data, out_data_list[i])

        pool = ThrePool()
        pool.map(_remove_silence, range(file_number))
        pool.close()
        pool.join()

    ## OSW: rewrote above more succintly
    def check_silence_pattern(self, label):
        for current_pattern in self.silence_pattern:
            current_pattern = current_pattern.strip('*')
            if current_pattern in label:
                return 1
        return 0

    def load_phone_alignment(self, alignment_file_name, dur_file_name=None):

        if dur_file_name:
            io_funcs = BinaryIOCollection()
            dur_dim = 1  ## hard coded for now
            manual_dur_data = io_funcs.load_binary_file(dur_file_name, dur_dim)

        ph_count = 0
        base_frame_index = 0
        nonsilence_frame_index_list = []
        fid = open(alignment_file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)

            if len(temp_list) == 1:
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]

                # to do - support different frame shift - currently hardwired to 5msec
                # currently under beta testing: supports different frame shift
                if dur_file_name:
                    frame_number = manual_dur_data[ph_count]
                    ph_count = ph_count + 1
                else:
                    frame_number = int((end_time - start_time) / 50000)

            label_binary_flag = self.check_silence_pattern(full_label)

            if self.remove_frame_features:
                if label_binary_flag == 0:
                    for frame_index in range(frame_number):
                        nonsilence_frame_index_list.append(base_frame_index + frame_index)
                base_frame_index = base_frame_index + frame_number
            elif self.subphone_feats == 'none':
                if label_binary_flag == 0:
                    nonsilence_frame_index_list.append(base_frame_index)
                base_frame_index = base_frame_index + 1

        fid.close()

        return nonsilence_frame_index_list

    def load_alignment(self, alignment_file_name, dur_file_name=None):

        state_number = 5
        base_frame_index = 0
        nonsilence_frame_index_list = []
        fid = open(alignment_file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            if len(temp_list) == 1:
                state_index = state_number
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]
                full_label_length = len(full_label) - 3  # remove state information [k]
                state_index = full_label[full_label_length + 1]
                state_index = int(state_index) - 1
                frame_number = int((end_time - start_time) / 50000)

            label_binary_flag = self.check_silence_pattern(full_label)

            if self.remove_frame_features:
                if label_binary_flag == 0:
                    for frame_index in range(frame_number):
                        nonsilence_frame_index_list.append(base_frame_index + frame_index)
                base_frame_index = base_frame_index + frame_number
            elif self.subphone_feats == 'state_only':
                if label_binary_flag == 0:
                    nonsilence_frame_index_list.append(base_frame_index)
                base_frame_index = base_frame_index + 1
            elif self.subphone_feats == 'none' and state_index == state_number:
                if label_binary_flag == 0:
                    nonsilence_frame_index_list.append(base_frame_index)
                base_frame_index = base_frame_index + 1

        fid.close()

        return nonsilence_frame_index_list


####################################################

class MinMaxNormalisation(object):
    def __init__(self, feature_dimension, min_value = 0.01, max_value = 0.99, min_vector = 0.0, max_vector = 0.0, exclude_columns=[]):

        # this is the wrong name for this logger because we can also normalise labels here too
        logger = logging.getLogger("acoustic_norm")

        self.target_min_value = min_value
        self.target_max_value = max_value

        self.feature_dimension = feature_dimension

        self.min_vector = min_vector
        self.max_vector = max_vector

        self.exclude_columns = exclude_columns

        if type(min_vector) != float:
            try:
                assert( len(self.min_vector) == self.feature_dimension)
            except AssertionError:
                logger.critical('inconsistent feature_dimension (%d) and length of min_vector (%d)' % (self.feature_dimension,len(self.min_vector)))
                raise

        if type(max_vector) != float:
            try:
                assert( len(self.max_vector) == self.feature_dimension)
            except AssertionError:
                logger.critical('inconsistent feature_dimension (%d) and length of max_vector (%d)' % (self.feature_dimension,len(self.max_vector)))
                raise

        logger.debug('MinMaxNormalisation created for feature dimension of %d' % self.feature_dimension)

    def load_min_max_values(self, label_norm_file):

        logger = logging.getLogger("acoustic_norm")

        io_funcs = BinaryIOCollection()
        min_max_vector, frame_number = io_funcs.load_binary_file_frame(label_norm_file, 1)
        min_max_vector = numpy.reshape(min_max_vector, (-1, ))
        self.min_vector = min_max_vector[0:frame_number//2]
        self.max_vector = min_max_vector[frame_number//2:]

        logger.info('Loaded min max values from the trained data for feature dimension of %d' % self.feature_dimension)

    def find_min_max_values(self, in_file_list):

        logger = logging.getLogger("acoustic_norm")

        file_number = len(in_file_list)
        min_value_matrix = numpy.zeros((file_number, self.feature_dimension))
        max_value_matrix = numpy.zeros((file_number, self.feature_dimension))
        io_funcs = BinaryIOCollection()
        for i in range(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)

            temp_min = numpy.amin(features, axis = 0)
            temp_max = numpy.amax(features, axis = 0)

            min_value_matrix[i, ] = temp_min;
            max_value_matrix[i, ] = temp_max;

        self.min_vector = numpy.amin(min_value_matrix, axis = 0)
        self.max_vector = numpy.amax(max_value_matrix, axis = 0)
        self.min_vector = numpy.reshape(self.min_vector, (1, self.feature_dimension))
        self.max_vector = numpy.reshape(self.max_vector, (1, self.feature_dimension))

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('across %d files found min/max values of length %d:' % (file_number,self.feature_dimension) )
        logger.info('  min: %s' % self.min_vector)
        logger.info('  max: %s' % self.max_vector)
        # restore the print options
        # numpy.set_printoptions(po)

    def normalise_data(self, in_file_list, out_file_list):
        file_number = len(in_file_list)

        fea_max_min_diff = self.max_vector - self.min_vector
        diff_value = self.target_max_value - self.target_min_value
        fea_max_min_diff = numpy.reshape(fea_max_min_diff, (1, self.feature_dimension))

        target_max_min_diff = numpy.zeros((1, self.feature_dimension))
        target_max_min_diff.fill(diff_value)

        target_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
        fea_max_min_diff[fea_max_min_diff <= 0.0] = 1.0

        io_funcs = BinaryIOCollection()
        for i in range(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)

            frame_number = features.size // self.feature_dimension
            fea_min_matrix = numpy.tile(self.min_vector, (frame_number, 1))
            target_min_matrix = numpy.tile(self.target_min_value, (frame_number, self.feature_dimension))

            fea_diff_matrix = numpy.tile(fea_max_min_diff, (frame_number, 1))
            diff_norm_matrix = numpy.tile(target_max_min_diff, (frame_number, 1)) / fea_diff_matrix

            norm_features = diff_norm_matrix * (features - fea_min_matrix) + target_min_matrix

            ## If we are to keep some columns unnormalised, use advanced indexing to
            ## reinstate original values:
            m,n = numpy.shape(features)
            for col in self.exclude_columns:
                norm_features[list(range(m)),[col]*m] = features[list(range(m)),[col]*m]

            io_funcs.array_to_binary_file(norm_features, out_file_list[i])

#            norm_features = numpy.array(norm_features, 'float32')
#            fid = open(out_file_list[i], 'wb')
#            norm_features.tofile(fid)
#            fid.close()

    def denormalise_data(self, in_file_list, out_file_list):

        logger = logging.getLogger("acoustic_norm")

        file_number = len(in_file_list)
        logger.info('MinMaxNormalisation.denormalise_data for %d files' % file_number)

        # print   self.max_vector, self.min_vector
        fea_max_min_diff = self.max_vector - self.min_vector
        diff_value = self.target_max_value - self.target_min_value
        # logger.debug('reshaping fea_max_min_diff from shape %s to (1,%d)' % (fea_max_min_diff.shape, self.feature_dimension) )

        fea_max_min_diff = numpy.reshape(fea_max_min_diff, (1, self.feature_dimension))

        target_max_min_diff = numpy.zeros((1, self.feature_dimension))
        target_max_min_diff.fill(diff_value)

        target_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
        fea_max_min_diff[fea_max_min_diff <= 0.0] = 1.0

        io_funcs = BinaryIOCollection()
        for i in range(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)

            frame_number = features.size // self.feature_dimension
            fea_min_matrix = numpy.tile(self.min_vector, (frame_number, 1))
            target_min_matrix = numpy.tile(self.target_min_value, (frame_number, self.feature_dimension))

            fea_diff_matrix = numpy.tile(fea_max_min_diff, (frame_number, 1))
            diff_norm_matrix = fea_diff_matrix / numpy.tile(target_max_min_diff, (frame_number, 1))
            norm_features = diff_norm_matrix * (features - target_min_matrix) + fea_min_matrix
            io_funcs.array_to_binary_file(norm_features, out_file_list[i])

    def normal_standardization(self, in_file_list, out_file_list):
        mean_vector = self.compute_mean(in_file_list)
        std_vector = self.compute_std(in_file_list, mean_vector)

        io_funcs = BinaryIOCollection()
        file_number = len(in_file_list)
        for i in range(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)
            current_frame_number = features.size // self.feature_dimension

            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))
            std_matrix = numpy.tile(std_vector, (current_frame_number, 1))

            norm_features = (features - mean_matrix) / std_matrix

            io_funcs.array_to_binary_file(norm_features, out_file_list[i])

    def compute_mean(self, file_list):

        logger = logging.getLogger("acoustic_norm")

        mean_vector = numpy.zeros((1, self.feature_dimension))
        all_frame_number = 0

        io_funcs = BinaryIOCollection()
        for file_name in file_list:
            features = io_funcs.load_binary_file(file_name, self.feature_dimension)
            current_frame_number = features.size // self.feature_dimension
            mean_vector += numpy.reshape(numpy.sum(features, axis=0), (1, self.feature_dimension))
            all_frame_number += current_frame_number

        mean_vector /= float(all_frame_number)

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('computed mean vector of length %d :' % mean_vector.shape[1] )
        logger.info(' mean: %s' % mean_vector)
        # restore the print options
        # numpy.set_printoptions(po)

        return  mean_vector

    def compute_std(self, file_list, mean_vector):

        logger = logging.getLogger("acoustic_norm")

        std_vector = numpy.zeros((1, self.feature_dimension))
        all_frame_number = 0

        io_funcs = BinaryIOCollection()
        for file_name in file_list:
            features = io_funcs.load_binary_file(file_name, self.feature_dimension)
            current_frame_number = features.size // self.feature_dimension
            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))

            std_vector += numpy.reshape(numpy.sum((features - mean_matrix) ** 2, axis=0), (1, self.feature_dimension))
            all_frame_number += current_frame_number

        std_vector /= float(all_frame_number)

        std_vector = std_vector ** 0.5

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('computed  std vector of length %d' % std_vector.shape[1] )
        logger.info('  std: %s' % std_vector)
        # restore the print options
        # numpy.set_printoptions(po)

        return  std_vector


#####################################################


class DeepRecurrentNetwork(object):
    """
    This class is to assemble various neural network architectures. From basic feedforward neural network to bidirectional gated recurrent neural networks and hybrid architecture. **Hybrid** means a combination of feedforward and recurrent architecture.

    """

    def __init__(self, n_in, hidden_layer_size, n_out, L1_reg, L2_reg, hidden_layer_type, output_type='LINEAR',
                 dropout_rate=0.0, optimizer='sgd', loss_function='MMSE', rnn_batch_training=False):
        """ This function initialises a neural network

        :param n_in: Dimensionality of input features
        :type in: Integer
        :param hidden_layer_size: The layer size for each hidden layer
        :type hidden_layer_size: A list of integers
        :param n_out: Dimensionality of output features
        :type n_out: Integrer
        :param hidden_layer_type: the activation types of each hidden layers, e.g., TANH, LSTM, GRU, BLSTM
        :param L1_reg: the L1 regulasation weight
        :param L2_reg: the L2 regulasation weight
        :param output_type: the activation type of the output layer, by default is 'LINEAR', linear regression.
        :param dropout_rate: probability of dropout, a float number between 0 and 1.
        """

        logger = logging.getLogger("DNN initialization")

        self.n_in = int(n_in)
        self.n_out = int(n_out)

        self.n_layers = len(hidden_layer_size)

        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.is_train = T.iscalar('is_train')
        self.rnn_batch_training = rnn_batch_training

        assert len(hidden_layer_size) == len(hidden_layer_type)

        self.list_of_activations = ['TANH', 'SIGMOID', 'SOFTMAX', 'RELU', 'RESU']

        if self.rnn_batch_training:
            self.x = T.tensor3('x')
            self.y = T.tensor3('y')
        else:
            self.x = T.matrix('x')
            self.y = T.matrix('y')

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.rnn_layers = []
        self.params = []
        self.delta_params = []

        rng = np.random.RandomState(123)

        for i in range(self.n_layers):
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_size[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.rnn_layers[i - 1].output
                if hidden_layer_type[i - 1] == 'BSLSTM' or hidden_layer_type[i - 1] == 'BLSTM':
                    input_size = hidden_layer_size[i - 1] * 2

            if hidden_layer_type[i] in self.list_of_activations:
                hidden_activation = hidden_layer_type[i].lower()
                hidden_layer = GeneralLayer(rng, layer_input, input_size, hidden_layer_size[i],
                                            activation=hidden_activation, p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'TANH_LHUC':
                hidden_layer = SigmoidLayer_LHUC(rng, layer_input, input_size, hidden_layer_size[i], activation=T.tanh,
                                                 p=self.dropout_rate, training=self.is_train)
            elif hidden_layer_type[i] == 'SLSTM':
                hidden_layer = SimplifiedLstm(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate,
                                              training=self.is_train, rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'SGRU':
                hidden_layer = SimplifiedGRU(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate,
                                             training=self.is_train, rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'GRU':
                hidden_layer = GatedRecurrentUnit(rng, layer_input, input_size, hidden_layer_size[i],
                                                  p=self.dropout_rate, training=self.is_train,
                                                  rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'LSTM_NFG':
                hidden_layer = LstmNFG(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate,
                                       training=self.is_train, rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'LSTM_NOG':
                hidden_layer = LstmNOG(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate,
                                       training=self.is_train, rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'LSTM_NIG':
                hidden_layer = LstmNIG(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate,
                                       training=self.is_train, rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'LSTM_NPH':
                hidden_layer = LstmNoPeepholes(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate,
                                               training=self.is_train, rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'LSTM':
                hidden_layer = VanillaLstm(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate,
                                           training=self.is_train, rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'BSLSTM':
                hidden_layer = BidirectionSLstm(rng, layer_input, input_size, hidden_layer_size[i],
                                                hidden_layer_size[i], p=self.dropout_rate, training=self.is_train,
                                                rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'BLSTM':
                hidden_layer = BidirectionLstm(rng, layer_input, input_size, hidden_layer_size[i], hidden_layer_size[i],
                                               p=self.dropout_rate, training=self.is_train,
                                               rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'RNN':
                hidden_layer = VanillaRNN(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate,
                                          training=self.is_train, rnn_batch_training=self.rnn_batch_training)
            elif hidden_layer_type[i] == 'LSTM_LHUC':
                hidden_layer = VanillaLstm_LHUC(rng, layer_input, input_size, hidden_layer_size[i], p=self.dropout_rate,
                                                training=self.is_train, rnn_batch_training=self.rnn_batch_training)
            else:
                logger.critical(
                    "This hidden layer type: %s is not supported right now! \n Please use one of the following: SLSTM, BSLSTM, TANH, SIGMOID\n" % (
                    hidden_layer_type[i]))
                sys.exit(1)

            self.rnn_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

        input_size = hidden_layer_size[-1]
        if hidden_layer_type[-1] == 'BSLSTM' or hidden_layer_type[-1] == 'BLSTM':
            input_size = hidden_layer_size[-1] * 2

        output_activation = output_type.lower()
        if output_activation == 'linear':
            self.final_layer = LinearLayer(rng, self.rnn_layers[-1].output, input_size, self.n_out)
        elif output_activation == 'recurrent':
            self.final_layer = RecurrentOutputLayer(rng, self.rnn_layers[-1].output, input_size, self.n_out,
                                                    rnn_batch_training=self.rnn_batch_training)
        elif output_type.upper() in self.list_of_activations:
            self.final_layer = GeneralLayer(rng, self.rnn_layers[-1].output, input_size, self.n_out,
                                            activation=output_activation)
        else:
            logger.critical(
                "This output layer type: %s is not supported right now! \n Please use one of the following: LINEAR, BSLSTM\n" % (
                    output_type))
            sys.exit(1)

        self.params.extend(self.final_layer.params)

        self.updates = {}
        for param in self.params:
            self.updates[param] = theano.shared(value=np.zeros(param.get_value(borrow=True).shape,
                                                               dtype=theano.config.floatX), name='updates')

        if self.loss_function == 'CCE':
            self.finetune_cost = self.categorical_crossentropy_loss(self.final_layer.output, self.y)
            self.errors = self.categorical_crossentropy_loss(self.final_layer.output, self.y)
        elif self.loss_function == 'Hinge':
            self.finetune_cost = self.multiclass_hinge_loss(self.final_layer.output, self.y)
            self.errors = self.multiclass_hinge_loss(self.final_layer.output, self.y)
        elif self.loss_function == 'MMSE':
            if self.rnn_batch_training:
                self.y_mod = T.reshape(self.y, (-1, n_out))
                self.final_layer_output = T.reshape(self.final_layer.output, (-1, n_out))

                nonzero_rows = T.any(self.y_mod, 1).nonzero()

                self.y_mod = self.y_mod[nonzero_rows]
                self.final_layer_output = self.final_layer_output[nonzero_rows]

                self.finetune_cost = T.mean(T.sum((self.final_layer_output - self.y_mod) ** 2, axis=1))
                self.errors = T.mean(T.sum((self.final_layer_output - self.y_mod) ** 2, axis=1))
            else:
                self.finetune_cost = T.mean(T.sum((self.final_layer.output - self.y) ** 2, axis=1))
                self.errors = T.mean(T.sum((self.final_layer.output - self.y) ** 2, axis=1))

    def categorical_crossentropy_loss(self, predictions, targets):
        return T.nnet.categorical_crossentropy(predictions, targets).mean()

    def multiclass_hinge_loss(self, predictions, targets, delta=1):
        num_cls = predictions.shape[1]
        if targets.ndim == predictions.ndim - 1:
            targets = T.extra_ops.to_one_hot(targets, num_cls)
        elif targets.ndim != predictions.ndim:
            raise TypeError('rank mismatch between targets and predictions')
        corrects = predictions[targets.nonzero()]
        rest = T.reshape(predictions[(1 - targets).nonzero()],
                         (-1, num_cls - 1))
        rest = T.max(rest, axis=1)
        return T.nnet.relu(rest - corrects + delta).mean()

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, use_lhuc=False, layer_index=0):
        """ This function is to build finetune functions and to update gradients

        :param train_shared_xy: theano shared variable for input and output training data
        :type train_shared_xy: tuple of shared variable
        :param valid_shared_xy: theano shared variable for input and output development data
        :type valid_shared_xy: tuple of shared variable
        :returns: finetune functions for training and development

        """

        logger = logging.getLogger("DNN initialization")

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        lr = T.scalar('lr', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)  # momentum

        cost = self.finetune_cost  # + self.L2_reg * self.L2_sqr

        ## added for LHUC
        if use_lhuc:
            # In lhuc the parameters are only scaling parameters which have the name 'c'
            self.lhuc_params = []
            for p in self.params:
                if p.name == 'c':
                    self.lhuc_params.append(p)
            params = self.lhuc_params
            gparams = T.grad(cost, params)
        else:
            params = self.params
            gparams = T.grad(cost, params)

        freeze_params = 0
        for layer in range(layer_index):
            freeze_params += len(self.rnn_layers[layer].params)

        # use optimizer
        if self.optimizer == 'sgd':
            # zip just concatenate two lists
            updates = OrderedDict()

            for i, (param, gparam) in enumerate(zip(params, gparams)):
                weight_update = self.updates[param]
                upd = mom * weight_update - lr * gparam
                updates[weight_update] = upd

                # freeze layers and update weights
                if i >= freeze_params:
                    updates[param] = param + upd

        elif self.optimizer == 'adam':
            updates = compile_ADAM_train_function(self, gparams, learning_rate=lr)
        elif self.optimizer == 'rprop':
            updates = compile_RPROP_train_function(self, gparams)
        else:
            logger.critical(
                "This optimizer: %s is not supported right now! \n Please use one of the following: sgd, adam, rprop\n" % (
                    self.optimizer))
            sys.exit(1)

        train_model = theano.function(inputs=[lr, mom],  # index, batch_size
                                      outputs=self.errors,
                                      updates=updates,
                                      givens={self.x: train_set_x,  # [index*batch_size:(index + 1)*batch_size]
                                              self.y: train_set_y,
                                              self.is_train: np.cast['int32'](1)}, on_unused_input='ignore')

        valid_model = theano.function(inputs=[],
                                      outputs=self.errors,
                                      givens={self.x: valid_set_x,
                                              self.y: valid_set_y,
                                              self.is_train: np.cast['int32'](0)}, on_unused_input='ignore')

        return train_model, valid_model

    def parameter_prediction(self, test_set_x):  # , batch_size
        """ This function is to predict the output of NN

        :param test_set_x: input features for a testing sentence
        :type test_set_x: python array variable
        :returns: predicted features

        """

        n_test_set_x = test_set_x.shape[0]

        test_out = theano.function([], self.final_layer.output,
                                   givens={self.x: test_set_x, self.is_train: np.cast['int32'](0)},
                                   on_unused_input='ignore')

        predict_parameter = test_out()

        return predict_parameter

    ## the function to output activations at a hidden layer
    def generate_hidden_layer(self, test_set_x, bn_layer_index):
        """ This function is to predict the bottleneck features of NN

        :param test_set_x: input features for a testing sentence
        :type test_set_x: python array variable
        :returns: predicted bottleneck features

        """

        n_test_set_x = test_set_x.shape[0]

        test_out = theano.function([], self.rnn_layers[bn_layer_index].output,
                                   givens={self.x: test_set_x, self.is_train: np.cast['int32'](0)},
                                   on_unused_input='ignore')

        predict_parameter = test_out()

        return predict_parameter


#######################################


class HTSLabelModification(object):
    """This class is to modify HTS format labels with predicted duration.

    Time alignments are expected in the HTS labels. Here is an example of the HTS labels:

    3050000 3100000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[2]

    3100000 3150000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[3]

    3150000 3250000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[4]

    3250000 3350000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[5]

    3350000 3900000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[6]

    305000 310000 are the starting and ending time.
    [2], [3], [4], [5], [6] mean the HMM state index.

    """

    def __init__(self, silence_pattern=['*-#+*'], label_type="state_align"):

        logger = logging.getLogger("labels")

        self.silence_pattern = silence_pattern
        self.silence_pattern_size = len(silence_pattern)
        self.label_type = label_type
        self.state_number = 5

    def check_silence_pattern(self, label):
        for current_pattern in self.silence_pattern:
            current_pattern = current_pattern.strip('*')
            if current_pattern in label:
                return 1
        return 0

    def modify_duration_labels(self, in_gen_label_align_file_list, gen_dur_list, gen_label_list):
        '''
        modifying duration from label alignments with predicted duration.
        '''
        utt_number = len(gen_dur_list)
        if utt_number != len(in_gen_label_align_file_list):
            print("the number of input and output files should be the same!\n");
            sys.exit(1)

        for i in range(utt_number):
            if (self.label_type == "state_align"):
                self.modify_dur_from_state_alignment_labels(in_gen_label_align_file_list[i], gen_dur_list[i],
                                                            gen_label_list[i])
            elif (self.label_type == "phone_align"):
                self.modify_dur_from_phone_alignment_labels(in_gen_label_align_file_list[i], gen_dur_list[i],
                                                            gen_label_list[i])
            else:
                logger.critical("we don't support %s labels as of now!!" % (self.label_type))
                sys.exit(1)

    def modify_dur_from_state_alignment_labels(self, label_file_name, gen_dur_file_name, gen_lab_file_name):
        logger = logging.getLogger("dur")

        state_number = self.state_number
        dur_dim = state_number

        io_funcs = BinaryIOCollection()
        dur_features, frame_number = io_funcs.load_binary_file_frame(gen_dur_file_name, dur_dim)

        fid = open(label_file_name)
        utt_labels = fid.readlines()
        fid.close()

        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (label_file_name, label_number))

        out_fid = open(gen_lab_file_name, 'w')

        current_index = 0
        prev_end_time = 0
        for line in utt_labels:
            line = line.strip()

            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)

            if len(temp_list) == 1:
                start_time = 0
                end_time = 600000  ## hard-coded silence duration
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]

                full_label_length = len(full_label) - 3  # remove state information [k]
                state_index = full_label[full_label_length + 1]
                state_index = int(state_index) - 1

            label_binary_flag = self.check_silence_pattern(full_label)

            if len(temp_list) == 1:
                for state_index in range(1, state_number + 1):
                    if label_binary_flag == 1:
                        current_state_dur = end_time - start_time
                    else:
                        pred_state_dur = dur_features[current_index, state_index - 1]
                        current_state_dur = int(pred_state_dur) * 5 * 10000
                    out_fid.write(str(prev_end_time) + ' ' + str(
                        prev_end_time + current_state_dur) + ' ' + full_label + '[' + str(state_index + 1) + ']\n')
                    prev_end_time = prev_end_time + current_state_dur
            else:
                if label_binary_flag == 1:
                    current_state_dur = end_time - start_time
                else:
                    pred_state_dur = dur_features[current_index, state_index - 1]
                    current_state_dur = int(pred_state_dur) * 5 * 10000
                out_fid.write(
                    str(prev_end_time) + ' ' + str(prev_end_time + current_state_dur) + ' ' + full_label + '\n')
                prev_end_time = prev_end_time + current_state_dur

            if state_index == state_number and label_binary_flag != 1:
                current_index += 1

        logger.debug('modifed label with predicted duration of %d frames x %d features' % dur_features.shape)

    def modify_dur_from_phone_alignment_labels(self, label_file_name, gen_dur_file_name, gen_lab_file_name):
        logger = logging.getLogger("dur")

        dur_dim = 1

        io_funcs = BinaryIOCollection()
        dur_features, frame_number = io_funcs.load_binary_file_frame(gen_dur_file_name, dur_dim)

        fid = open(label_file_name)
        utt_labels = fid.readlines()
        fid.close()

        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (label_file_name, label_number))

        out_fid = open(gen_lab_file_name, 'w')

        current_index = 0
        prev_end_time = 0
        for line in utt_labels:
            line = line.strip()

            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)

            if len(temp_list) == 1:
                start_time = 0
                end_time = 3000000  ## hard-coded silence duration
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]

            label_binary_flag = self.check_silence_pattern(full_label)

            if label_binary_flag == 1:
                current_phone_dur = end_time - start_time
                out_fid.write(
                    str(prev_end_time) + ' ' + str(prev_end_time + current_phone_dur) + ' ' + full_label + '\n')
                prev_end_time = prev_end_time + current_phone_dur
                continue;
            else:
                phone_dur = dur_features[current_index]
                phone_dur = int(phone_dur) * 5 * 10000
                out_fid.write(str(prev_end_time) + ' ' + str(prev_end_time + phone_dur) + ' ' + full_label + '\n')
                prev_end_time = prev_end_time + phone_dur

            current_index += 1

        logger.debug('modifed label with predicted duration of %d frames x %d features' % dur_features.shape)


##############################################


class FeatureNormBase(object):
    '''
    to normalise feature into specific range
    to de-normalise feature back
    support min-max norm, MVN,
    this is a genetic class
    '''
    def __init__(self):
        self.logger = logging.getLogger('feature_normalisation')

        self.dimension_dict = {}
        self.start_index_dict = {}
        self.feature_dimension = 0

    def feature_normalisation(self):
        pass

    def feature_denormalisation(self):
        pass


    def normal_standardization(self, in_file_list, out_file_list, feature_dimension):

#        self.dimension_dict = dimension_dict
        self.feature_dimension = feature_dimension

        mean_vector = self.compute_mean(in_file_list, 0, feature_dimension)
        std_vector = self.compute_std(in_file_list, mean_vector, 0, feature_dimension)

        io_funcs = BinaryIOCollection()
        file_number = len(in_file_list)

        for i in range(file_number):

            features, current_frame_number = io_funcs.load_binary_file_frame(in_file_list[i], self.feature_dimension)

            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))
            std_matrix = numpy.tile(std_vector, (current_frame_number, 1))

            norm_features = (features - mean_matrix) / std_matrix

            io_funcs.array_to_binary_file(norm_features, out_file_list[i])

        return  mean_vector, std_vector

    def find_min_max_values(self, in_file_list, start_index, end_index):

        local_feature_dimension = end_index - start_index

        file_number = len(in_file_list)
        min_value_matrix = numpy.zeros((file_number, local_feature_dimension))
        max_value_matrix = numpy.zeros((file_number, local_feature_dimension))
        io_funcs = BinaryIOCollection()
        for i in range(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)

            temp_min = numpy.amin(features[:, start_index:end_index], axis = 0)
            temp_max = numpy.amax(features[:, start_index:end_index], axis = 0)

            min_value_matrix[i, ] = temp_min;
            max_value_matrix[i, ] = temp_max;

        self.min_vector = numpy.amin(min_value_matrix, axis = 0)
        self.max_vector = numpy.amax(max_value_matrix, axis = 0)
        self.min_vector = numpy.reshape(self.min_vector, (1, local_feature_dimension))
        self.max_vector = numpy.reshape(self.max_vector, (1, local_feature_dimension))

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        self.logger.info('found min/max values of length %d:' % local_feature_dimension)
        self.logger.info('  min: %s' % self.min_vector)
        self.logger.info('  max: %s' % self.max_vector)
        # restore the print options
        # numpy.set_printoptions(po)

    def compute_mean(self, file_list, start_index, end_index):

        local_feature_dimension = end_index - start_index

        mean_vector = numpy.zeros((1, local_feature_dimension))
        all_frame_number = 0

        io_funcs = BinaryIOCollection()
        for file_name in file_list:
            features, current_frame_number = io_funcs.load_binary_file_frame(file_name, self.feature_dimension)

            mean_vector += numpy.reshape(numpy.sum(features[:, start_index:end_index], axis=0), (1, local_feature_dimension))
            all_frame_number += current_frame_number

        mean_vector /= float(all_frame_number)

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        self.logger.info('computed mean vector of length %d :' % mean_vector.shape[1] )
        self.logger.info(' mean: %s' % mean_vector)
        # restore the print options
        # numpy.set_printoptions(po)

        return  mean_vector

    def compute_std(self, file_list, mean_vector, start_index, end_index):
        local_feature_dimension = end_index - start_index

        std_vector = numpy.zeros((1, self.feature_dimension))
        all_frame_number = 0

        io_funcs = BinaryIOCollection()
        for file_name in file_list:
            features, current_frame_number = io_funcs.load_binary_file_frame(file_name, self.feature_dimension)

            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))

            std_vector += numpy.reshape(numpy.sum((features[:, start_index:end_index] - mean_matrix) ** 2, axis=0), (1, local_feature_dimension))
            all_frame_number += current_frame_number

        std_vector /= float(all_frame_number)

        std_vector = std_vector ** 0.5

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        self.logger.info('computed  std vector of length %d' % std_vector.shape[1] )
        self.logger.info('  std: %s' % std_vector)
        # restore the print options
        # numpy.set_printoptions(po)

        return  std_vector


##########################


class   MeanVarianceNorm(FeatureNormBase):
    '''
    plan: 1: support normal MVN and denormalisation for both input and output
          2: support stream-based operation: for example, some streams can use min-max, other streams use MVN, may need one more class
    '''
#    def __init__(self, feature_dimension):
    def __init__(self, feature_dimension):

        self.mean_vector = None
        self.std_vector  = None
        self.feature_dimension = feature_dimension

    def feature_normalisation(self, in_file_list, out_file_list):
        logger = logging.getLogger('feature_normalisation')

#        self.feature_dimension = feature_dimension
        try:
            assert len(in_file_list) == len(out_file_list)
        except  AssertionError:
            logger.critical('The input and output file numbers are not the same! %d vs %d' %(len(in_file_list), len(out_file_list)))
            raise

        if self.mean_vector is None:
            self.mean_vector = self.compute_mean(in_file_list, 0, self.feature_dimension)
        if self.std_vector  is None:
            self.std_vector = self.compute_std(in_file_list, self.mean_vector, 0, self.feature_dimension)

        io_funcs = BinaryIOCollection()
        file_number = len(in_file_list)
        for i in range(file_number):
            features, current_frame_number = io_funcs.load_binary_file_frame(in_file_list[i], self.feature_dimension)

            mean_matrix = numpy.tile(self.mean_vector, (current_frame_number, 1))
            std_matrix = numpy.tile(self.std_vector, (current_frame_number, 1))

            norm_features = (features - mean_matrix) / std_matrix

            io_funcs.array_to_binary_file(norm_features, out_file_list[i])

        return  self.mean_vector, self.std_vector

    def feature_denormalisation(self, in_file_list, out_file_list, mean_vector, std_vector):
        io_funcs = BinaryIOCollection()
        file_number = len(in_file_list)
        try:
            assert len(in_file_list) == len(out_file_list)
        except  AssertionError:
            logger.critical('The input and output file numbers are not the same! %d vs %d' %(len(in_file_list), len(out_file_list)))
            raise

        try:
            assert  mean_vector.size == self.feature_dimension and std_vector.size == self.feature_dimension
        except AssertionError:
            logger.critical('the dimensionalities of the mean and standard derivation vectors are not the same as the dimensionality of the feature')
            raise

        for i in range(file_number):
            features, current_frame_number = io_funcs.load_binary_file_frame(in_file_list[i], self.feature_dimension)

            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))
            std_matrix = numpy.tile(std_vector, (current_frame_number, 1))

            norm_features = features * std_matrix + mean_matrix

            io_funcs.array_to_binary_file(norm_features, out_file_list[i])

    def load_mean_std_values(self, acoustic_norm_file):

        logger = logging.getLogger('feature_normalisation')

        io_funcs = BinaryIOCollection()
        mean_std_vector, frame_number = io_funcs.load_binary_file_frame(acoustic_norm_file, 1)
        mean_std_vector = numpy.reshape(mean_std_vector, (-1, ))
        self.mean_vector = mean_std_vector[0:frame_number//2]
        self.std_vector = mean_std_vector[frame_number//2:]

        logger.info('Loaded mean std values from the trained data for feature dimension of %d' % self.feature_dimension)
        return self.mean_vector, self.std_vector

    def compute_mean(self, file_list, start_index, end_index):

        logger = logging.getLogger('feature_normalisation')

        local_feature_dimension = end_index - start_index

        mean_vector = numpy.zeros((1, local_feature_dimension))
        all_frame_number = 0

        io_funcs = BinaryIOCollection()
        for file_name in file_list:
            features, current_frame_number = io_funcs.load_binary_file_frame(file_name, self.feature_dimension)

            mean_vector += numpy.reshape(numpy.sum(features[:, start_index:end_index], axis=0), (1, local_feature_dimension))
            all_frame_number += current_frame_number

        mean_vector /= float(all_frame_number)

        # setting the print options in this way seems to break subsequent printing of numpy float32 types
        # no idea what is going on - removed until this can be solved
        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('computed mean vector of length %d :' % mean_vector.shape[1] )
        logger.info(' mean: %s' % mean_vector)
        # restore the print options
        # numpy.set_printoptions(po)

        self.mean_vector = mean_vector

        return  mean_vector

    def compute_std(self, file_list, mean_vector, start_index, end_index):

        logger = logging.getLogger('feature_normalisation')

        local_feature_dimension = end_index - start_index

        std_vector = numpy.zeros((1, self.feature_dimension))
        all_frame_number = 0

        io_funcs = BinaryIOCollection()
        for file_name in file_list:
            features, current_frame_number = io_funcs.load_binary_file_frame(file_name, self.feature_dimension)

            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))

            std_vector += numpy.reshape(numpy.sum((features[:, start_index:end_index] - mean_matrix) ** 2, axis=0), (1, local_feature_dimension))
            all_frame_number += current_frame_number

        std_vector /= float(all_frame_number)

        std_vector = std_vector ** 0.5

        # setting the print options in this way seems to break subsequent printing of numpy float32 types
        # no idea what is going on - removed until this can be solved
        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('computed  std vector of length %d' % std_vector.shape[1] )
        logger.info('  std: %s' % std_vector)
        # restore the print options
        # numpy.set_printoptions(po)

        self.std_vector = std_vector

        return  std_vector


##############################################

from theano import tensor as T


################################################

from numpy import float64
class MLParameterGeneration(object):
    def __init__(self, delta_win = [-0.5, 0.0, 0.5], acc_win = [1.0, -2.0, 1.0]):
        self.delta_win = delta_win
        self.acc_win   = acc_win
        ###assume the delta and acc windows have the same length
        self.win_length = int(len(delta_win)/2)

    def build_win_mats(self, windows, frames):
        win_mats = []
        for l, u, win_coeff in windows:
            assert l >= 0 and u >= 0
            assert len(win_coeff) == l + u + 1
            win_coeffs = np.tile(np.reshape(win_coeff, (l + u + 1, 1)), frames)
            win_mat = bm.band_c_bm(u, l, win_coeffs).T
            win_mats.append(win_mat)

        return win_mats

    def build_poe(self, b_frames, tau_frames, win_mats, sdw=None):
#        tau_frames.astype('float64')

        if sdw is None:
            sdw = max([ win_mat.l + win_mat.u for win_mat in win_mats ])
        num_windows = len(win_mats)
        frames = len(b_frames)
        assert np.shape(b_frames) == (frames, num_windows)
        assert np.shape(tau_frames) == (frames, num_windows)
        assert all([ win_mat.l + win_mat.u <= sdw for win_mat in win_mats ])

        b = np.zeros((frames,))
        prec = bm.zeros(sdw, sdw, frames)

        for win_index, win_mat in enumerate(win_mats):
            bm.dot_mv_plus_equals(win_mat.T, b_frames[:, win_index], target=b)
            bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=prec,
                                  diag=float64(tau_frames[:, win_index]))

        return b, prec

    def generation(self, features, covariance, static_dimension):

        windows = [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ]
        num_windows = len(windows)

        frame_number = features.shape[0]
        #frame_number = frame_number / 2
        #covariance = covariance[244:]
        #features = features[0:244]

        logger = logging.getLogger('param_generation')
        logger.debug('starting MLParameterGeneration.generation')
        #frame_number = frame_number / 2

        gen_parameter = np.zeros((frame_number, static_dimension))

        win_mats = self.build_win_mats(windows, frame_number)
        mu_frames = np.zeros((frame_number, 3))
        var_frames = np.zeros((frame_number, 3))

        for d in range(static_dimension):
            var_frames[:, 0] = covariance[:, d]
            var_frames[:, 1] = covariance[:, static_dimension+d]
            var_frames[:, 2] = covariance[:, static_dimension*2+d]
            mu_frames[:, 0] = features[:, d]
            mu_frames[:, 1] = features[:, static_dimension+d]
            mu_frames[:, 2] = features[:, static_dimension*2+d]
            var_frames[0, 1] = 100000000000;
            var_frames[0, 2] = 100000000000;
            var_frames[frame_number-1, 1] = 100000000000;
            var_frames[frame_number-1, 2] = 100000000000;

            b_frames = mu_frames / var_frames
            tau_frames = 1.0 / var_frames

            b, prec = self.build_poe(b_frames, tau_frames, win_mats)
            mean_traj = bla.solveh(prec, b)

            gen_parameter[0:frame_number, d] = mean_traj

        return  gen_parameter


################################################


class   ParameterGeneration(object):

    def __init__(self, gen_wav_features = ['mgc', 'lf0', 'bap'], enforce_silence=False):
        self.gen_wav_features = gen_wav_features
        self.enforce_silence  = enforce_silence

        # Debug:
        self.inf_float = -1.0e+10
        #self.inf_float = -50000

        # not really necessary to have the logger rembered in the class - can easily obtain it by name instead
        # self.logger = logging.getLogger('param_generation')

        self.var = {}

    def duration_decomposition(self, in_file_list, dimension, out_dimension_dict, file_extension_dict):

        logger = logging.getLogger('param_generation')

        logger.debug('duration_decomposition for %d files' % len(in_file_list) )

        state_number = 5  ## hard coding, try removing in future?

        if len(list(out_dimension_dict.keys()))>1:
            logger.critical("we don't support any additional features along with duration as of now.")
            sys.exit(1)
        else:
            feature_name = list(out_dimension_dict.keys())[0]

        io_funcs = BinaryIOCollection()

        findex=0
        flen=len(in_file_list)
        for file_name in in_file_list:

            findex=findex+1

            dir_name = os.path.dirname(file_name)
            file_id = os.path.splitext(os.path.basename(file_name))[0]

            features, frame_number = io_funcs.load_binary_file_frame(file_name, dimension)
            gen_features = numpy.int32(numpy.round(features))
            gen_features[gen_features<1]=1

            if dimension > state_number:
                gen_features = gen_features[:, state_number]

            logger.info('processing %4d of %4d: %s' % (findex,flen,file_name) )

            new_file_name = os.path.join(dir_name, file_id + file_extension_dict[feature_name])
            io_funcs.array_to_binary_file(gen_features, new_file_name)

            logger.debug('wrote to file %s' % new_file_name)

    def acoustic_decomposition(self, in_file_list, dimension, out_dimension_dict, file_extension_dict, var_file_dict, do_MLPG=True, cfg=None):

        logger = logging.getLogger('param_generation')

        logger.debug('acoustic_decomposition for %d files' % len(in_file_list) )

        self.load_covariance(var_file_dict, out_dimension_dict)

        stream_start_index = {}
        dimension_index = 0
        recorded_vuv = False
        vuv_dimension = None

        for feature_name in list(out_dimension_dict.keys()):
#            if feature_name != 'vuv':
            stream_start_index[feature_name] = dimension_index
#            else:
#                vuv_dimension = dimension_index
#                recorded_vuv = True

            dimension_index += out_dimension_dict[feature_name]

        io_funcs = BinaryIOCollection()

        mlpg_algo = MLParameterGeneration()

        findex=0
        flen=len(in_file_list)
        for file_name in in_file_list:

            findex=findex+1

            dir_name = os.path.dirname(file_name)
            file_id = os.path.splitext(os.path.basename(file_name))[0]

            features, frame_number = io_funcs.load_binary_file_frame(file_name, dimension)
            #features = features[:40]
            #frame_number = 40
            #frame_number = frame_number - 244

            logger.info('processing %4d of %4d: %s' % (findex,flen,file_name) )

            for feature_name in self.gen_wav_features:

                logger.debug(' feature: %s' % feature_name)

                current_features = features[:, stream_start_index[feature_name]:stream_start_index[feature_name]+out_dimension_dict[feature_name]]
                if FAST_MLPG:
                    ### fast version wants variance per frame, not single global one:
                    var = self.var[feature_name]
                    var = numpy.transpose(numpy.tile(var,frame_number))
                else:
                    var = self.var[feature_name]

#                print  var.shape[1]
                if do_MLPG == False:
                    gen_features = current_features
                else:
                    gen_features = mlpg_algo.generation(current_features, var, out_dimension_dict[feature_name]//3)
#                else:
#                    self.logger.critical("the dimensions do not match for MLPG: %d vs %d" %(var.shape[1], out_dimension_dict[feature_name]))
#                    raise

                logger.debug(' feature dimensions: %d by %d' %(gen_features.shape[0], gen_features.shape[1]))

                if feature_name in ['lf0', 'F0']:
                    if 'vuv' in stream_start_index:
                        vuv_feature = features[:, stream_start_index['vuv']:stream_start_index['vuv']+1]

                        for i in range(frame_number):
                            if vuv_feature[i, 0] < 0.5 or gen_features[i, 0] < numpy.log(20):
                                gen_features[i, 0] = self.inf_float

                new_file_name = os.path.join(dir_name, file_id + file_extension_dict[feature_name])

                if self.enforce_silence:
                    #_silence_pattern = silence_pattern
                    label_align_dir = gen_dur_label_dir
                    in_f = open(label_align_dir+'/'+file_id+'.lab','r')
                    for line in in_f.readlines():
                        line = line.strip()

                        if len(line) < 1:
                            continue
                        temp_list  = re.split('\s+', line)
                        start_time = int(int(temp_list[0])*(10**-4)/5)
                        end_time   = int(int(temp_list[1])*(10**-4)/5)

                        full_label = temp_list[2]

                        label_binary_flag = self.check_silence_pattern(full_label, silence_pattern)

                        if label_binary_flag:
                            if feature_name in ['lf0', 'F0', 'mag']:
                                gen_features[start_time:end_time, :] = self.inf_float
                            else:
                                gen_features[start_time:end_time, :] = 0.0

                io_funcs.array_to_binary_file(gen_features, new_file_name)
                logger.debug(' wrote to file %s' % new_file_name)


    def load_covariance(self, var_file_dict, out_dimension_dict):

        io_funcs = BinaryIOCollection()
        for feature_name in list(var_file_dict.keys()):
            var_values, dimension = io_funcs.load_binary_file_frame(var_file_dict[feature_name], 1)

            var_values = numpy.reshape(var_values, (out_dimension_dict[feature_name], 1))

            self.var[feature_name] = var_values


    def check_silence_pattern(self, label, silence_pattern):
        for current_pattern in silence_pattern:
            current_pattern = current_pattern.strip('*')
            if current_pattern in label:
                return 1
        return 0


#######################################################

def combin_features(feature_names):
    combined_feature_name = ''
    for feature_name in feature_names:
        combined_feature_name += '_'
        combined_feature_name += feature_name
    return combined_feature_name

def cal_dim(feature_names):
    cmp_dim = 0
    for feature_name in feature_names:
        cmp_dim += dims[feature_name]
    return cmp_dim

def out_dim(feature_names):
    output_dim = {}
    for feature_name in feature_names:
        output_dim[feature_name] = dims[feature_name]
    return output_dim

def out_ext(feature_names):
    output_ext = {}
    for feature_name in feature_names:
        output_ext[feature_name] = file_extension_dict[feature_name]
    return output_ext

def read_file_list(file_path):
  file_lists = []
  for line in os.listdir(file_path):
      file_lists.append(line[:line.find('.')])
  return file_lists


def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
  logger = logging.getLogger('prepare path')
  if not os.path.exists(file_dir) and new_dir_switch:
    os.makedirs(file_dir)
  logger.info('Preparing file_list for %s in dir %s', file_extension, file_dir)
  return [os.path.join(file_dir, file_id + file_extension) for file_id in file_id_list]


def extract_file_id_list(file_list):
    file_id_list = []
    for file_name in file_list:
        file_id = os.path.basename(os.path.splitext(file_name)[0])
        file_id_list.append(file_id)

    return  file_id_list

def make_output_file_list(out_dir, in_file_lists):
    out_file_lists = []

    for in_file_name in in_file_lists:
        file_id = os.path.basename(in_file_name)
        out_file_name = out_dir + '/' + file_id
        out_file_lists.append(out_file_name)

    return  out_file_lists


def load_covariance(var_file_dict, out_dimension_dict):
    var = {}
    io_funcs = BinaryIOCollection()
    for feature_name in list(var_file_dict.keys()):
        var_values, dimension = io_funcs.load_binary_file_frame(var_file_dict[feature_name], 1)
        var_values = numpy.reshape(var_values, (out_dimension_dict[feature_name], 1))
        var[feature_name] = var_values

    return  var


def dnn_generation(valid_file_list, nnets_file_name, n_ins, n_outs, out_file_list, reshape_io=False):
    logger = logging.getLogger("dnn_generation")
    #logger.debug('Starting dnn_generation')
    #plotlogger = logging.getLogger("plotting")

    dnn_model = pickle.load(open(nnets_file_name, 'rb'))
    file_number = len(valid_file_list)

    for i in range(file_number):  #file_number
        logger.info('generating %4d of %4d: %s' % (i+1,file_number,valid_file_list[i]) )
        fid_lab = open(valid_file_list[i], 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        features = features[:(n_ins * (features.size // n_ins))]
        test_set_x = features.reshape((-1, n_ins))
        n_rows = test_set_x.shape[0]
        
        if reshape_io:
            test_set_x = numpy.reshape(test_set_x, (1, test_set_x.shape[0], n_ins))
            test_set_x = numpy.array(test_set_x, 'float32')

        predicted_parameter = dnn_model.parameter_prediction(test_set_x)
        predicted_parameter = predicted_parameter.reshape(-1, n_outs)
        predicted_parameter = predicted_parameter[0:n_rows]
        
        ### write to cmp file
        predicted_parameter = numpy.array(predicted_parameter, 'float32')
        temp_parameter = predicted_parameter
        fid = open(out_file_list[i], 'wb')
        predicted_parameter.tofile(fid)
        logger.debug('saved to %s' % out_file_list[i])
        fid.close()

def run_process(args,log=True):
    logger = logging.getLogger("subprocess")
    # a convenience function instead of calling subprocess directly
    # this is so that we can do some logging and catch exceptions

    # we don't always want debug logging, even when logging level is DEBUG
    # especially if calling a lot of external functions
    # so we can disable it by force, where necessary
    if log:
        logger.debug('%s' % args)
    try:
        # the following is only available in later versions of Python
        # rval = subprocess.check_output(args)

        # bufsize=-1 enables buffering and may improve performance compared to the unbuffered case
        p = subprocess.Popen(args, bufsize=-1, shell=True,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        close_fds=True, env=os.environ)
        # better to use communicate() than read() and write() - this avoids deadlocks
        (stdoutdata, stderrdata) = p.communicate()
        if p.returncode != 0:
            # for critical things, we always log, even if log==False
            logger.critical('exit status %d' % p.returncode )
            logger.critical(' for command: %s' % args )
            logger.critical('      stderr: %s' % stderrdata )
            logger.critical('      stdout: %s' % stdoutdata )
            raise OSError
        return (stdoutdata, stderrdata)
    except subprocess.CalledProcessError as e:
        # not sure under what circumstances this exception would be raised in Python 2.6
        logger.critical('exit status %d' % e.returncode )
        logger.critical(' for command: %s' % args )
        # not sure if there is an 'output' attribute under 2.6 ? still need to test this...
        logger.critical('  output: %s' % e.output )
        raise
    except ValueError:
        logger.critical('ValueError for %s' % args )
        raise
    except OSError:
        logger.critical('OSError for %s' % args )
        raise
    except KeyboardInterrupt:
        logger.critical('KeyboardInterrupt during %s' % args )
        try:
            # try to kill the subprocess, if it exists
            p.kill()
        except UnboundLocalError:
            # this means that p was undefined at the moment of the keyboard interrupt
            # (and we do nothing)
            pass
        raise KeyboardInterrupt


def bark_alpha(sr):
    return 0.8517*np.sqrt(np.arctan(0.06583*sr/1000.0))-0.1916

def erb_alpha(sr):
    return 0.5941*np.sqrt(np.arctan(0.1418*sr/1000.0))+0.03237

def post_filter(mgc_file_in, mgc_file_out, mgc_dim, pf_coef, fw_coef, co_coef, fl_coef, gen_dir):

    line = "echo 1 1 "
    for i in range(2, mgc_dim):
        line = line + str(pf_coef) + " "

    run_process('{line} | {x2x} +af > {weight}'
                .format(line=line, x2x=SPTK['X2X'], weight=os.path.join(gen_dir, 'weight')))

    run_process('{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | {c2acr} -m {co} -M 0 -l {fl} > {base_r0}'
                .format(freqt=SPTK['FREQT'], order=mgc_dim-1, fw=fw_coef, co=co_coef, mgc=mgc_file_in, c2acr=SPTK['C2ACR'], fl=fl_coef, base_r0=mgc_file_in+'_r0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {freqt} -m {order} -a {fw} -M {co} -A 0 | {c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        freqt=SPTK['FREQT'], fw=fw_coef, co=co_coef,
                        c2acr=SPTK['C2ACR'], fl=fl_coef, base_p_r0=mgc_file_in+'_p_r0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 0 -e 0 > {base_b0}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        mc2b=SPTK['MC2B'], fw=fw_coef,
                        bcp=SPTK['BCP'], base_b0=mgc_file_in+'_b0'))

    run_process('{vopr} -d < {base_r0} {base_p_r0} | {sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
                .format(vopr=SPTK['VOPR'], base_r0=mgc_file_in+'_r0', base_p_r0=mgc_file_in+'_p_r0',
                        sopr=SPTK['SOPR'],
                        base_b0=mgc_file_in+'_b0', base_p_b0=mgc_file_in+'_p_b0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 1 -e {order} | {merge} -n {order2} -s 0 -N 0 {base_p_b0} | {b2mc} -m {order} -a {fw} > {base_p_mgc}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        mc2b=SPTK['MC2B'],  fw=fw_coef,
                        bcp=SPTK['BCP'],
                        merge=SPTK['MERGE'], order2=mgc_dim-2, base_p_b0=mgc_file_in+'_p_b0',
                        b2mc=SPTK['B2MC'], base_p_mgc=mgc_file_out))
    return


def generate_wav(gen_dir, file_id_list):
    """
    Waveform generation with WORLD vocoders.
    (whose acoustic parameters are: mgc, bap, and lf0)
    """
    logger = logging.getLogger("wav_generation")

    ## to be moved
    if isinstance(fw_alpha, str):
        if fw_alpha=='Bark':
            fw_coef = bark_alpha(sr)
        elif fw_alpha=='ERB':
            fw_coef = bark_alpha(sr)
        else:
            raise ValueError('cfg.fw_alpha='+fw_alpha+' not implemented, the frequency warping coefficient "fw_coef" cannot be deduced.')
    else:
        fw_coef = fw_alpha
    fl_coef = fl

    counter=1
    max_counter = len(file_id_list)

    for filename in file_id_list:

        logger.info('creating waveform for %4d of %4d: %s' % (counter,max_counter,filename) )

        base = filename
        files = {'sp': base + file_extension_dict['sp'],
                 'mgc': base + file_extension_dict['mgc'],
                 'f0': base + '.f0',
                 'lf0': base + file_extension_dict['lf0'],
                 'ap': base + '.ap',
                 'bap': base + file_extension_dict['bap'],
                 'wav': base + '.wav'}

        mgc_file_name = files['mgc']
        bap_file_name = files['bap']

        cur_dir = os.getcwd()
        os.chdir(gen_dir)

        ### post-filtering
        if do_post_filtering:
            logger.info('creating waveform for %4d of %4d: %s do_post_filtering' % (counter, max_counter, filename))
            mgc_file_name = files['mgc']+'_p_mgc'
            post_filter(files['mgc'], mgc_file_name, dims['mgc'], pf_coef, fw_coef, co_coef, fl_coef, gen_dir)
            logger.info('creating waveform for %4d of %4d: %s end-do_post_filtering' % (counter, max_counter, filename))

        ###mgc to sp to wav using WORLD
        logger.info('creating waveform for %4d of %4d: %s start-sopr' % (counter, max_counter, filename))
        run_process('{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | {x2x} +fd > {f0}'.format(sopr=SPTK['SOPR'],
                                                                                             lf0=files['lf0'],
                                                                                             x2x=SPTK['X2X'],
                                                                                             f0=files['f0']))

        run_process('{sopr} -c 0 {bap} | {x2x} +fd > {ap}'.format(sopr=SPTK['SOPR'], bap=files['bap'], x2x=SPTK['X2X'],
                                                                  ap=files['ap']))

        logger.info('creating waveform for %4d of %4d: %s end-sopr' % (counter, max_counter, filename))
        ### If using world v2, please comment above line and uncomment this
        # run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 0 {bap} | {sopr} -d 32768.0 -P | {x2x} +fd > {ap}'
        #            .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.bap_dim, fl=cfg.fl, bap=bap_file_name, sopr=SPTK['SOPR'], x2x=SPTK['X2X'], ap=files['ap']))

        logger.info('creating waveform for %4d of %4d: %s start-mgc2sp' % (counter, max_counter, filename))
        run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} | {sopr} -d 32768.0 -P | {x2x} +fd > {sp}'
                    .format(mgc2sp=SPTK['MGC2SP'], alpha=fw_alpha, order=dims['mgc'] - 1, fl=fl,
                            mgc=mgc_file_name, sopr=SPTK['SOPR'], x2x=SPTK['X2X'], sp=files['sp']))
        logger.info('creating waveform for %4d of %4d: %s end-mgc2sp' % (counter, max_counter, filename))
        logger.info('creating waveform for %4d of %4d: %s start-world' % (counter, max_counter, filename))
        run_process('{synworld} {fl} {sr} {f0} {sp} {ap} {wav}'
                    .format(synworld=WORLD['SYNTHESIS'], fl=fl, sr=sr, f0=files['f0'], sp=files['sp'],
                            ap=files['ap'], wav=files['wav']))
        logger.info('creating waveform for %4d of %4d: %s end-world' % (counter, max_counter, filename))
        # run_process('rm -f {ap} {sp} {f0}'.format(ap=files['ap'],sp=files['sp'],f0=files['f0']))
        counter = counter + 1

        os.chdir(cur_dir)


def perform_acoustic_composition_on_split(args):
    """ Performs acoustic composition on one chunk of data.
        This is used as input for Pool.map to allow parallel acoustic composition.
    """
    (delta_win, acc_win, in_file_list_dict, nn_cmp_file_list, in_dimension_dict, out_dimension_dict) = args
    acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)
    acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, in_dimension_dict, out_dimension_dict)


def perform_acoustic_composition(delta_win, acc_win, in_file_list_dict, nn_cmp_file_list, cfg, parallel=True):
    """ Runs acoustic composition from in_file_list_dict to nn_cmp_file_list.
        If parallel is true, splits the data into multiple chunks and calls
        perform_acoustic_composition_on_split for each chunk.
    """
    if parallel:
        num_splits = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_splits)

        # split data into a list of num_splits tuples with each tuple representing
        # the parameters for perform_acoustic_compositon_on_split
        splits_full = [
             (delta_win,
              acc_win,
              {stream: in_file_list_dict[stream][i::num_splits] for stream in in_file_list_dict},
              nn_cmp_file_list[i::num_splits],
              cfg.in_dimension_dict,
              cfg.out_dimension_dict
             ) for i in range(num_splits) ]

        pool.map(perform_acoustic_composition_on_split, splits_full)
        pool.close()
        pool.join()
    else:
        acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)
        acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, cfg.in_dimension_dict, cfg.out_dimension_dict)


def logging_configuration():
    # get a logger
    logger = logging.getLogger()
    log_file_name = '%s_%s.log' % (model_file_name, datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)6s%(name)20s: %(message)s'))
    console_handler.setLevel(logging.INFO)

    log_path = os.path.join(test_synth_dir, log_file_name)
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)6s%(name)20s: %(message)s'))
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)


def main_function_dur():
    # get a logger for this main function
    #logging_configuration()
    logger = logging.getLogger()

    ##### Label Normalisation #####
    label_normaliser = HTSLabelNormalisation(question_file_name, add_frame_features, subphone_feats_dur)
    lab_dim = label_normaliser.dimension
    logger.info('Input label dimension is %d' % lab_dim)

    in_label_align_file_list = prepare_file_path_list(file_name_scp, prompt_label_dir, '.lab')
    binary_label_file_list = prepare_file_path_list(file_name_scp, gen_dur_label_dir, '.labbin')
    nn_label_file_list = prepare_file_path_list(file_name_scp, gen_dur_label_dir, '.lab')
    nn_label_norm_file_list = prepare_file_path_list(file_name_scp, gen_dur_label_dir, '.lab')

    logger.info('preparing label data (input) using standard HTS style labels')
    label_normaliser.perform_normalisation(in_label_align_file_list, binary_label_file_list, label_align_type)

    ##### Label Remove Silence #####
    remover = SilenceRemover(n_cmp=lab_dim, silence_pattern=silence_pattern, label_type=label_align_type,
                             remove_frame_features=add_frame_features, subphone_feats=subphone_feats_dur)
    remover.remove_silence(binary_label_file_list, in_label_align_file_list, nn_label_file_list)

    ##### Label MinMax Normalisation #####
    min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)
    min_max_normaliser.load_min_max_values(dur_label_norm_file)
    ### enforce silence such that the normalization runs without removing silence: only for final synthesis
    if GenTestList and enforce_silence:
        min_max_normaliser.normalise_data(binary_label_file_list, nn_label_norm_file_list)
    else:
        min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)

    ##### DNN Generation #####
    logger.info('generating from DNN')
    test_x_file_list = nn_label_norm_file_list
    gen_file_list = prepare_file_path_list(file_name_scp, gen_dur_label_dir, '.cmp')
    reshape_io = True if rnn_batch_training else False
    dnn_generation(test_x_file_list, dur_model_file, lab_dim, cal_dim(output_features_dur), gen_file_list, reshape_io)
    fid = open(dur_info_norm_file, 'rb')
    cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
    fid.close()
    cmp_min_max = cmp_min_max.reshape((2, -1))
    cmp_min_vector = cmp_min_max[0, ]
    cmp_max_vector = cmp_min_max[1, ]
    denormaliser = MeanVarianceNorm(feature_dimension = cal_dim(output_features_dur))
    denormaliser.feature_denormalisation(gen_file_list, gen_file_list, cmp_min_vector, cmp_max_vector)

    ##### Duration Labels Generation #####
    gen_dur_list   = prepare_file_path_list(file_name_scp, gen_dur_label_dir, '.dur')
    gen_label_list = prepare_file_path_list(file_name_scp, gen_dur_label_dir, '.lab')
    in_gen_label_align_file_list = prepare_file_path_list(file_name_scp, prompt_label_dir, '.lab', False)

    generator = ParameterGeneration(gen_wav_features = gen_wav_features)
    generator.duration_decomposition(gen_file_list, cal_dim(output_features_dur), out_dim(output_features_dur), out_ext(output_features_dur))

    label_modifier = HTSLabelModification(silence_pattern = silence_pattern, label_type = label_align_type)
    label_modifier.modify_duration_labels(in_gen_label_align_file_list, gen_dur_list, gen_label_list)
    logger.info('duration model generate finish')


def main_function_acoustic():
    logger = logging.getLogger()

    ##### Label Normalisation #####
    add_frame_features = True
    label_normaliser = HTSLabelNormalisation(question_file_name, add_frame_features, subphone_feats_acoustic)
    lab_dim = label_normaliser.dimension
    logger.info('Input label dimension is %d' % lab_dim)

    in_label_align_file_list = prepare_file_path_list(file_name_scp, gen_dur_label_dir, '.lab')
    binary_label_file_list = prepare_file_path_list(file_name_scp, gen_para_dir, '.labbin')
    nn_label_file_list = prepare_file_path_list(file_name_scp, gen_para_dir, '.lab')
    nn_label_norm_file_list = prepare_file_path_list(file_name_scp, gen_para_dir, '.lab')
    logger.info('preparing label data (input) using standard HTS style labels')
    label_normaliser.perform_normalisation(in_label_align_file_list, binary_label_file_list, label_align_type)

    remover = SilenceRemover(n_cmp=lab_dim, silence_pattern=silence_pattern, label_type=label_align_type,
                             remove_frame_features=add_frame_features, subphone_feats=subphone_feats_acoustic)
    remover.remove_silence(binary_label_file_list, in_label_align_file_list, nn_label_file_list)

    min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)
    min_max_normaliser.load_min_max_values(acou_label_norm_file)
    enforce_silence = True
    if GenTestList and enforce_silence:
        min_max_normaliser.normalise_data(binary_label_file_list, nn_label_norm_file_list)
    else:
        min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)

    ##### DNN Generation #####
    test_x_file_list = nn_label_norm_file_list
    logger.info('generating from DNN')
    gen_file_list = prepare_file_path_list(file_name_scp, gen_para_dir, '.cmp')
    reshape_io = True if rnn_batch_training else False
    dnn_generation(test_x_file_list, acou_model_file, lab_dim, acoustic_dim, gen_file_list, reshape_io)


    fid = open(acou_info_norm_file, 'rb')
    cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
    fid.close()
    cmp_min_max = cmp_min_max.reshape((2, -1))
    cmp_min_vector = cmp_min_max[0, ]
    cmp_max_vector = cmp_min_max[1, ]

    denormaliser = MeanVarianceNorm(feature_dimension = acoustic_dim)
    denormaliser.feature_denormalisation(gen_file_list, gen_file_list, cmp_min_vector, cmp_max_vector)

    ##### Generate Speech #####
    generator = ParameterGeneration(gen_wav_features = gen_wav_features, enforce_silence = enforce_silence)
    generator.acoustic_decomposition(gen_file_list, acoustic_dim, output_dims, out_ext(output_features_acou), acou_var_file_dict, do_MLPG=do_MLPG)
    generate_wav(gen_para_dir, file_name_scp)  # generated speech
    for file in os.listdir(gen_para_dir):
        if file[file.find('.'):] == '.wav':
            copyfile(os.path.join(gen_para_dir, file), os.path.join(gen_wav_dir, file))
    logger.info('Generate Speech finish')


if __name__ == '__main__':
    logging_configuration()

    main_function_dur()
    main_function_acoustic()

    sys.exit(0)
