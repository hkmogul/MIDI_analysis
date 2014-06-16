import os
import glob
import sys
sys.path.append('../')
import key_identify

if len(sys.argv) < 2:
  print "WARNING: Unspecified measure value from command line. Will use default of 0."
  measure_value = 0
else:
  try:
    measure_value = int(sys.argv[1])
  except ValueError:
    print "Invalid command line argument.  Will use default of 0"
    measure_value = 0

MIDI_PATH="../../midi-dataset/data/sanity"

midi_glob = sorted(glob.glob(os.path.join(MIDI_PATH, 'midi', '*.mid')))

for midi_filename in midi_glob:
  print "Identifying key of {}".format(os.path.basename(midi_filename))
  key_identify.identify_key(midi_filename, command_line_print = False, measure_value= measure_value)
