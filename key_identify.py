# Identifies the time varying key of a MIDI file
# Hilary Mogul
# hilary.mogul@gmail.com
# May 2014
# Version 0.0.1
from pretty_midi import PrettyMIDI
import numpy as np
import midi
import os
import sys

def make_chroma_vector(chroma_slice):
  """Returns chroma vector of sums from starting time to ending time"""
  chroma_vector = np.zeros((12,1))
  chroma_vector[0] = np.sum(chroma_slice[11,])
  for i in range(1,12):
    chroma_vector[i] = np.sum(chroma_slice[11-i])
  return chroma_vector

def get_approx_key(chroma_vector, keys):
  """Returns index of approximated key, and the score that this approximated key got"""
  chroma_vector = chroma_vector[:]
  scores = []
  end = keys.shape[0] -1
  for i in range(0,end):
    key_vector = keys[i,:]
    score = np.dot(key_vector,chroma_vector)
    scores.append(score)
  key_index = scores.index(max(scores))
  arr = np.array([[key_index, max(scores)]])
  return arr

def get_key_score(chroma_vector, keys, key_index):
  """Returns the score of an approximated key, given the index of the key weights to try out"""
  chroma_vector = np.rot90(chroma_vector,3)
  chroma_vector = chroma_vector[0,:]
  key_vector = keys[key_index,:]
  score = np.dot(key_vector,chroma_vector)
  return score


def load_keys():
  """ Returns arrays of the weighted keys, and the corresponding names of them """
  #Building of weighted vectors
  if not os.path.exists("key_base_info.npz"):
    print "Building default array- major keys only"
    key_weight = np.array([[ 3, -1,  1, -1,  2,  1, -1,  2, -1,  1, -1,  2]])
    key_weight = np.vstack((key_weight,np.array([[2,  3, -1,  1, -1,  2,  1, -1,  2, -1,  1, -1]])))
    key_weight = np.vstack((key_weight,np.array([[-1,  2,  3, -1,  1, -1,  2,  1, -1,  2, -1,  1]])))
    key_weight = np.vstack((key_weight,np.array([[1, -1,  2,  3, -1,  1, -1,  2,  1, -1,  2, -1]])))
    key_weight = np.vstack((key_weight,np.array([[-1,  1, -1,  2,  3, -1,  1, -1,  2,  1, -1,  2]])))
    key_weight = np.vstack((key_weight,np.array([[2, -1,  1, -1,  2,  3, -1,  1, -1,  2,  1, -1]])))
    key_weight = np.vstack((key_weight,np.array([[-1,  2, -1,  1, -1,  2,  3, -1,  1, -1,  2,  1]])))
    key_weight = np.vstack((key_weight,np.array([[1, -1,  2, -1,  1, -1,  2,  3, -1,  1, -1,  2]])))
    key_weight = np.vstack((key_weight,np.array([[2,  1, -1,  2, -1,  1, -1,  2,  3, -1,  1, -1]])))
    key_weight = np.vstack((key_weight,np.array([[-1,  2,  1, -1,  2, -1,  1, -1,  2,  3, -1,  1]])))
    key_weight = np.vstack((key_weight,np.array([[1, -1,  2,  1, -1,  2, -1,  1, -1,  2,  3, -1]])))
    key_weight = np.vstack((key_weight,np.array([[-1,  1, -1,  2,  1, -1,  2, -1,  1, -1,  2,  3]])))
    names = np.array([['C-Maj', 'C#-Maj', 'D-Maj', 'D#-Maj', 'E-Maj', 'F-Maj','F#-Maj','G-Maj','G#-Maj', 'A-Maj', 'A#-Maj','B-Maj']])
    np.savez("key_base_info", key_weight, names)
  npzfile = np.load("key_base_info.npz")
  key_weight = npzfile['arr_0']
  names = npzfile['arr_1']
  #vector of key names (This version will use only major keys)

  return key_weight, names


#Main function of program:
def identify_key(midi_filename, command_line_print = True, save_results = True, measure_value = 1):
  """ Runs key identification algorithm

  :parameters:
    - midi_filename : String of name of existing midi file to process.
    - command_line_print : Boolean to allow for printing results to command line.
    - save_results : Boolean to allow saving the approximated key and the names of those keys to a .npz file matching the midi files name
    - measure_value : int > 0 that sets how many beats the program will process per approximation.
    """

  song = PrettyMIDI(midi.read_midifile(midi_filename))

  #get beat locations for slices
  beats = song.get_beats() #output is an np.ndarray
  times = beats.flatten()
  sectionBeats = True
  #create slices of chroma data to process including summing them up
  #output: every "measure" of beats

  if measure_value< 1 or measure_value >times.size or measure_value == None:
    sectionBeats = False
    print "WARNING: measure_value selected less than 1 or greater than beat size.  Will do one approximation for the whole song."



  key_weight, names = load_keys()
  #get Chroma features
  chroma = song.get_chroma()
  #normalize chroma features
  chroma /= (chroma.max( axis = 0 ) + (chroma.max( axis = 0 ) == 0))

  # first case
  if sectionBeats:
    chroma_slice = chroma[:,0:round(times[0])*100]
  else:
    chroma_slice = chroma
  # Sum rows to find intensity of each note.
  vec = np.sum(chroma_slice, axis=1)
  keys_approx = get_approx_key(vec, key_weight)

  #for each slice, get approximated key and score into 2 column array (key, score)
  #possiblymay need to use indices of key names instead of actual keys
  #chroma time indices have a resolution of 10 ms

  times_used = np.array([[times[0]]])

  if sectionBeats:
    for t in range(1, times.size-1,measure_value):
    #make chroma slice based on time
      if times.size -t < measure_value:
        chroma_slice = chroma[:,round(times[t]*100):round(times[t+1]*100)]
      # print chroma_slice
      # vec = make_chroma_vector(chroma_slice)
        vec = np.sum(chroma_slice, axis=1)
      # vec = vec[::-1]
      else:
        chroma_slice = chroma[:,round(times[t]*100):round(times[t+1]*100)]
      # print chroma_slice
      # vec = make_chroma_vector(chroma_slice)
        vec = np.sum(chroma_slice, axis=1)
      # vec = vec[::-1]
      apr = get_approx_key(vec, key_weight)
      #if the score isn't 0 (which happens in silence), add the approximated key to the list
      if not apr[0,1] == 0:
        keys_approx = np.vstack((keys_approx, apr))
        times_used = np.vstack((times_used, np.array([[times[t]]])))


  # DUMMIED OUT CODE FOR FUTURE IMPLEMENTATION


  #iterate through rows of array- if there is a change, get % difference in scores for each key and use threshold to figure
  #if it is a key change.  mark key change in final 2 column array of (key, time start)
  threshold = .15 #experimental value
  keys_final = np.array([[keys_approx[0,0]]])

  times_final = np.array([[times_used[0,0]]])


  # if times.size > 1:
  #   print "going thru removal loop"
  #   for t in range (1, keys_approx.shape[0]):
  #     current = keys_approx[t,0]
  #     prev = keys_approx[t-1,0]
  #     if not current == prev: #if not equal to previous, check % difference of scores
  #       print "In key change check"
  #       score1 = keys_approx[t,1] #score of key of this time slice
  #       print score1
  #       vec = make_chroma_vector(chroma, round(times[t,0]*100,round(times[t+1,0])*100 ))
  #       print vec
  #       score2 = get_key_score(vec, key_weight, prev) #score it would have gotten with last input key
  #       print score2
  #       diff = abs(score1-score2)/(score1+score2)
  #       print diff
  #       if diff > threshold:
  #         arr = np.array([[keys_approx[t,0], times[t,0] ]])
  #         # print arr
  #         print "Key change at index: ", times[t,0]
  #         final_keys = np.vstack((final_keys, arr))
  #       else:
  #         print "difference not large enough to constitute key change "
  if times.size > 1:
    for t in range(1, keys_approx.shape[0]):
      current = keys_approx[t,0]
      prev = keys_approx[t-1,0]
      # in the meantime, just put any that are different up
      # TODO: set up threshold experiment to check for % difference in key change
      if current != prev:
        keys_final = np.vstack((keys_final, np.array([[current]])))
        times_final = np.vstack((times_final, np.array([[times_used[t,0]]])))


  e = keys_final.shape[0]
  if command_line_print:
    for i in range(0, e):
      key_int = int(keys_final[i,0])
      print "In key: ",names[0,key_int],"at time: ", times_final[i,0]
  if save_results:
    filename = os.path.basename(midi_filename)
    filename_raw = os.path.splitext(filename)[0]
    # if not os.path.exists(directory):
    dirname = "Results_of_key_approximation"
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    np.savez(dirname+"/"+filename_raw+"_key_approx-vars", keys_approx = keys_approx, names = names)
    file_results = open(dirname+"/"+filename_raw+"_key_approx-text_form.txt",'w')
    file_results.write(filename_raw+"\n")
    for i in range(0, e):
      key_int = int(keys_final[i,0])
      file_results.write("In key: "+names[0,key_int]+" at time: "+ str(times_final[i,0])+"\n")
    file_results.close()
  return keys_approx, names
