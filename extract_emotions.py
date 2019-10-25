import numpy
import subprocess
import os

#Runs the classifier, see arguments.py for more information. The arguments here load the LSTM classifier (link in README.md), uses the corpus: newsentences.csv, runs preprocessing, and runs on the cpu. If you have a gpu in your machine remove the --cpu flag.
subprocess.call("python3.7 run_classifier.py --load mlstm_semeval.clf --data newsentences.csv --text-key sentence --preprocess --cpu", shell=True)
sentiment_prob = numpy.load("clf_results.npy.prob.npy")
corpus = open("newsentences.csv", 'r').readlines()
output_emotions = open("concat.txt", "w")
plutchik = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

for i in range(len(sentiment_prob)):
  emos_scores = filter(lambda x: x > 0.5, sentiment_prob[i])
  if len(emos_scores) > 0:
    emos_scores.sort(reverse = True)
    emos_idxs = map(lambda x: numpy.where(sentiment_prob[i] == x)[0][0], emos_scores[:2])
    emos = map(lambda x: plutchik[x], emos_idxs)
  else:
    emos = ["neutral"]
  output_emotions.write('"'+str(corpus[i+1])[:-1]+'"'+': '+str(emos)+',\n')
