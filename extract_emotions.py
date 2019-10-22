import numpy
import subprocess
import os

#subprocess.call("python3.7 run_classifier.py --load mlstm_semeval.clf --data newsentences.csv --text-key sentence --preprocess --cpu", shell=True)
sentiment_prob = numpy.load("clf_results.npy.prob.npy")
corpus = open("newsentences.csv", 'r').readlines()
output_emotions = open("concat.txt", "w")
plutchik = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

EMOTIONS = {
  "Mad": ["Someone cut in front of you in line", 
        "You just stubbed your toe", 
        "Your rival got a job you applied for"],
  "Scared": ["You're alone and you hear a concerning noise", 
          "You're lost in an unfamiliar area", 
          "You're speaking to an audience and experieincing stagefright."],
  "Joyful": ["You're having fun at a comedy show'", 
          "Your crush smiles at you", 
          "You just received an offer for your dream job"],
  "Powerful": ["You just aced an exam", 
             "You just defeated your rival in a game", 
             "You're wearing an outfit that makes you feel professional"],
  "Peaceful": ["You just had a relaxing massage", 
             "Your friends gave you a warm hug", 
             "You are enjoying a nice meal with your mother"],
  "Sad": ["You just failed an exam", 
        "Your friends are having fun at a concert you weren't invited to", 
        "You're up late at night doing homework"],
  "Neutral": ["Use a neutral tone"]
}

emotions_idx = {"Mad":0, "Scared":0, "Joyful":0, "Powerful":0, "Peaceful":0, "Sad":0, "Neutral":0}
for i in range(len(sentiment_prob)):
  emos_scores = filter(lambda x: x > 0.5, sentiment_prob[i])
  if len(emos_scores) > 0:
    emos_scores.sort(reverse = True)
    emos_idxs = map(lambda x: numpy.where(sentiment_prob[i] == x)[0][0], emos_scores[:2])
    emos = map(lambda x: plutchik[x], emos_idxs)
  else:
    emos = []
  print(emos)
  output_emotions.write('"'+str(corpus[i+1])[:-1]+'"'+': '+str(emos)+',\n')
