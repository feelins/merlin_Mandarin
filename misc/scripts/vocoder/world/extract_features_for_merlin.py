import os
import sys
import shutil
import glob
import time
import multiprocessing as mp
import numpy as np

if len(sys.argv)!=5:
    print("Usage: ")
    print("python extract_features_for_merlin.py <path_to_merlin_dir> <path_to_wav_dir> <path_to_feat_dir> <sampling rate>")
    sys.exit(1)

# top merlin directory
merlin_dir = sys.argv[1]

# input audio directory
wav_dir = sys.argv[2]

# Output features directory
out_dir = sys.argv[3]

# initializations
fs = int(sys.argv[4])

# tools directory
world  = os.path.join(merlin_dir, "tools/bin/WORLD")
sptk   = os.path.join(merlin_dir, "tools/bin/SPTK-3.9")
reaper = os.path.join(merlin_dir, "tools/bin/REAPER")

sp_dir  = os.path.join(out_dir, 'sp' )
mgc_dir = os.path.join(out_dir, 'mgc')
ap_dir  = os.path.join(out_dir, 'ap' )
bap_dir = os.path.join(out_dir, 'bap')
f0_dir  = os.path.join(out_dir, 'f0' )
lf0_dir = os.path.join(out_dir, 'lf0')

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if not os.path.exists(sp_dir):
    os.mkdir(sp_dir)

if not os.path.exists(mgc_dir):
    os.mkdir(mgc_dir)

if not os.path.exists(bap_dir):
    os.mkdir(bap_dir)

if not os.path.exists(f0_dir):
    os.mkdir(f0_dir)

if not os.path.exists(lf0_dir):
    os.mkdir(lf0_dir)

if fs == 16000:
    nFFTHalf = 1024
    alpha = 0.58

elif fs == 22050:
    nFFTHalf = 1024
    alpha = 0.65

elif fs == 44100:
    nFFTHalf = 2048
    alpha = 0.76

elif fs == 48000:
    nFFTHalf = 2048
    alpha = 0.77

else:
    print("As of now, we don't support %d Hz sampling rate." %(fs))
    print("Please consider either downsampling to 16000 Hz or upsampling to 48000 Hz")
    sys.exit(1)

#bap order depends on sampling rate.
mcsize=59
b_use_reaper=True # If True: Reaper is used for f0 extraction. If False: The vocoder is used for f0 extraction.

def get_wav_filelist(wav_dir):
    wav_files = []
    for file in os.listdir(wav_dir):
        whole_filepath = os.path.join(wav_dir,file)
        if os.path.isfile(whole_filepath) and str(whole_filepath).endswith(".wav"):
            wav_files.append(whole_filepath)
        elif os.path.isdir(whole_filepath):
            wav_files += get_wav_filelist(whole_filepath)

    wav_files.sort()

    return wav_files


def read_binfile(filename, dim=60, dtype=np.float64):
    '''
    Reads binary file into numpy array.
    '''
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=dtype)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)).astype('float64') # This is to keep compatibility with numpy default dtype.
    m_data = np.squeeze(m_data)
    return  m_data

def write_binfile(m_data, filename, dtype=np.float64):
    '''
    Writes numpy array into binary file.
    '''
    m_data = np.array(m_data, dtype)
    fid = open(filename, 'wb')
    m_data.tofile(fid)
    fid.close()
    return

def read_reaper_f0_file(est_file, skiprows=7):
    '''
    Reads f0 track into numpy array from EST file generated by REAPER.
    '''
    v_f0 = np.loadtxt(est_file, skiprows=skiprows, usecols=[2])
    v_f0[v_f0<0] = 0
    return v_f0

def reaper_f0_extract(in_wavfile, f0_file_ref, f0_file_out, frame_shift_ms=5.0):
    '''
    Extracts f0 track using REAPER.
    To keep consistency with the vocoder, it also fixes for the difference in number
    of frames between the REAPER f0 track and the acoustic parameters extracted by the vocoder.
    f0_file_ref: f0 extracted by the vocoder. It is used as a reference to fix the number of frames, as explained.
    '''

    # Run REAPER:
    print("Running REAPER f0 extraction...")
    cmd = "%s -a -s -x 400 -m 50 -u %1.4f -i %s -f %s" % (os.path.join(reaper, 'reaper'), frame_shift_ms / 1000.0, in_wavfile, f0_file_out + "_reaper")
    os.system(cmd)

    # Protection - number of frames:
    v_f0_ref = read_binfile(f0_file_ref, dim=1)
    v_f0     = read_reaper_f0_file(f0_file_out + "_reaper")
    frm_diff = v_f0.size - v_f0_ref.size
    if frm_diff<0:
        v_f0 = np.r_[ v_f0, np.zeros(-frm_diff) + v_f0[-1]]
    if frm_diff>0:
        v_f0 = v_f0[:-frm_diff]

    # Save f0 file:
    write_binfile(v_f0, f0_file_out)
    return


def process(filename):
    '''
    The function decomposes a wav file into F0, mel-cepstral coefficients, and aperiodicity
    :param filename: path to wav file
    :return: .lf0, .mgc and .bap files
    '''

    file_id = os.path.basename(filename).split(".")[0]
    print('\n' + file_id)

    ### WORLD ANALYSIS -- extract vocoder parameters ###
    ### extract sp, ap ###
    f0_file = os.path.join(f0_dir, file_id + '.f0')
    f0_world_file = f0_file
    if b_use_reaper:
        f0_world_file = f0_file + "_world"

    world_analysis_cmd = "%s %s %s %s %s" % (os.path.join(world, 'analysis'), \
                                             filename,
                                             f0_world_file, \
                                             os.path.join(sp_dir, file_id + '.sp'), \
                                             os.path.join(bap_dir, file_id + '.bapd'))
    os.system(world_analysis_cmd)

    ### Extract f0 using reaper ###
    if b_use_reaper:
        reaper_f0_extract(filename, f0_world_file, f0_file)

    ### convert f0 to lf0 ###
    sptk_x2x_da_cmd = "%s +da %s > %s" % (os.path.join(sptk, 'x2x'), f0_file, \
                                          os.path.join(f0_dir, file_id + '.f0a'))
    os.system(sptk_x2x_da_cmd)

    sptk_x2x_af_cmd = "%s +af %s | %s > %s " % (os.path.join(sptk, 'x2x'), \
                                                os.path.join(f0_dir, file_id + '.f0a'), \
                                                os.path.join(sptk, 'sopr') + ' -magic 0.0 -LN -MAGIC -1.0E+10', \
                                                os.path.join(lf0_dir, file_id + '.lf0'))
    os.system(sptk_x2x_af_cmd)

    ### convert sp to mgc ###
    sptk_x2x_df_cmd1 = "%s +df %s | %s | %s >%s" % (os.path.join(sptk, 'x2x'), \
                                                    os.path.join(sp_dir, file_id + '.sp'), \
                                                    os.path.join(sptk, 'sopr') + ' -R -m 32768.0', \
                                                    os.path.join(sptk, 'mcep') + ' -a ' + str(alpha) + ' -m ' + str(
                                                        mcsize) + ' -l ' + str(
                                                        nFFTHalf) + ' -e 1.0E-8 -j 0 -f 0.0 -q 3 ', \
                                                    os.path.join(mgc_dir, file_id + '.mgc'))
    os.system(sptk_x2x_df_cmd1)

    ### convert bapd to bap ###
    sptk_x2x_df_cmd2 = "%s +df %s > %s " % (os.path.join(sptk, "x2x"), \
                                            os.path.join(bap_dir, file_id + ".bapd"), \
                                            os.path.join(bap_dir, file_id + '.bap'))
    os.system(sptk_x2x_df_cmd2)

print("--- Feature extraction started ---")
start_time = time.time()

# get wav files list
wav_files = get_wav_filelist(wav_dir)

# do multi-processing
pool = mp.Pool(mp.cpu_count())
pool.map(process, wav_files)

# DEBUG:
#for nxf in xrange(len(wav_files)):
#    process(wav_files[nxf])

# clean temporal files
shutil.rmtree(sp_dir, ignore_errors=True)
shutil.rmtree(f0_dir, ignore_errors=True)


for zippath in glob.iglob(os.path.join(bap_dir, '*.bapd')):
    os.remove(zippath)

print("You should have your features ready in: "+out_dir)    

(m, s) = divmod(int(time.time() - start_time), 60)
print(("--- Feature extraction completion time: %d min. %d sec ---" % (m, s)))

