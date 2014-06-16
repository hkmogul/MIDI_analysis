import os
import glob
import sys
sys.path.append('../')
import key_identify

MIDI_PATH="../../midi-dataset/data/sanity"

midi_glob = sorted(glob.glob(os.path.join(MIDI_PATH, 'midi', '*.mid')))

for midi_filename in midi_glob:
  print "Identifying key of {}".format(os.path.basename(midi_filename))
  key_identify.identify_key(midi_filename, command_line_print = False, measure_value = 4)
