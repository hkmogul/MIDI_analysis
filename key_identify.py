# Identifies the time varying key of a MIDI file
# Hilary Mogul
# hilary.mogul@gmail.com
# May 2014
# Version 0.0.1
from pretty_midi import PrettyMIDI
import numpy as np
import midi


def make_chroma_vector(chroma_slice):
  """Returns chroma vector of sums from starting time to ending time"""
  # print time_start, " to ", time_end
  chroma_vector = np.zeros((12,1))
  chroma_vector[0] = np.sum(chroma_slice[11,])
  for i in range(1,12):
    chroma_vector[i] = np.sum(chroma_slice[11-i])
  return chroma_vector

def get_approx_key(chroma_vector, keys):
  """Returns index of approximated key, and the score that this approximated key got"""
  # chroma_vector = np.rot90(chroma_vector,3) #rotate vector from chroma to match key matrix
  chroma_vector = chroma_vector[:]
  # print chroma_vector
  scores = []
  end = keys.shape[0] -1
  for i in range(0,end):
    key_vector = keys[i,:]
    score = np.dot(key_vector,chroma_vector)
    scores.append(score)
  key_index = scores.index(max(scores))
  # print key_index
  # print max(scores)
  arr = np.array([[key_index, max(scores)]])
  return arr

def get_key_score(chroma_vector, keys, key_index):
  """Returns the score of an approximated key, given the index of the key weights to try out"""
  chroma_vector = np.rot90(chroma_vector,3)
  chroma_vector = chroma_vector[0,:]
  # print chroma_vector
  key_vector = keys[key_index,:]
  score = np.dot(key_vector,chroma_vector)
  return score


#Main scripting of program:

#Building of weighted vectors
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


#vector of key names (This version will use only major keys)
names = ['C-Maj', 'C#-Maj', 'D-Maj', 'D#-Maj', 'E-Maj', 'F-Maj','F#-Maj','G-Maj','G#-Maj', 'A-Maj', 'A#-Maj','B-Maj']

filename = raw_input('Enter MIDI filename: ')
song = PrettyMIDI(midi.read_midifile(filename))

#get beat locations for slices
beats = song.get_beats() #output is an np.ndarray
times = beats.flatten()
sectionBeats = True
#create slices of chroma data to process including summing them up
#output: every "measure" of beats

m= int(raw_input('Choose integer value for resolution (beats per key approximation): '))
if m< 1 or m >times.size:
  sectionBeats = False
  print "Will do one approximation for whole song."




#get Chroma features
chroma = song.get_chroma()
#normalize chroma features
chroma /= (chroma.max( axis = 0 ) + (chroma.max( axis = 0 ) == 0))

# first case
if sectionBeats:
  chroma_slice = chroma[:,0:round(times[0])*100]
else:
  chroma_slice = chroma
# print chroma_slice
# Sum rows to find intensity of each note.
vec = np.sum(chroma_slice, axis=1)
# print vec
# reverse vector so index corresponding to C is at 11 instead of 0
# vec = vec[::-1]
# print vec
keys_approx = get_approx_key(vec, key_weight)

#for each slice, get approximated key and score into 2 column array (key, score)
#possiblymay need to use indices of key names instead of actual keys
#chroma time indices have a resolution of 10 ms

times_used = np.array([[times[0]]])

# print "length of iterable", len(range(1,times.size -1,m))
if sectionBeats:
  for t in range(1, times.size-1,m):
  #make chroma slice based on time
    if times.size -t <m:
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
    if not apr[0,1] == 0: #if the score isn't 0 (which happens in silence)
      keys_approx = np.vstack((keys_approx, apr))
      times_used = np.vstack((times_used, np.array([[times[t]]])))


# DUMMIED OUT CODE FOR FUTURE IMPLEMENTATION
# final_keys = np.array([[keys_approx[0,0],times[0,0]]]) #mark first
# print final_keys
#
# #iterate through rows of array- if there is a change, get % difference in scores for each key and use threshold to figure
# #if it is a key change.  mark key change in final 2 column array of (key, time start)
# threshold = .15 #experimental value
#
# if times.size > 1:
#   # print "going thru removal loop"
#   for t in range (1, keys_approx.shape[0]):
#     current = keys_approx[t,0]
#     prev = keys_approx[t-1,0]
#     if not current == prev: #if not equal to previous, check % difference of scores
#       print "In key change check"
#       score1 = keys_approx[t,1] #score of key of this time slice
#       # print score1
#       vec = make_chroma_vector(chroma, round(times[0,t])*100,round(times[0,t+1])*100 )
#       # print vec
#       score2 = get_key_score(vec, key_weight, prev) #score it would have gotten with last input key
#       # print score2
#       diff = abs(score1-score2)/(score1+score2)
#       print diff
#       if diff > threshold:
#         arr = np.array([[keys_approx[t,0], times[t,0] ]])
#         # print arr
#         print "Key change at index: ", times[t,0]
#         final_keys = np.vstack((final_keys, arr))
#       else:
#         print "difference not large enough to constitute key change "
#

e = keys_approx.shape[0]

for i in range(0, e):
  # print i
  print "In key:",names[int(keys_approx[i,0])],"at time:", times_used[i,0]
