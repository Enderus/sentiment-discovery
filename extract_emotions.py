import numpy
import subprocess
import os

subprocess.call("python3.7 run_classifier.py --load mlstm_semeval.clf --data newsentences.csv --text-key sentence --preprocess --cpu", shell=True)
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
    if (max(sentiment_prob[i])>0.4): 
        e = numpy.argmax(sentiment_prob[i])
        if (plutchik[e] == plutchik[0]):
            emotions_idx["Mad"]+=1
            output_emotions.write('"'+EMOTIONS['Mad'][emotions_idx["Mad"]%len(EMOTIONS['Mad'])]+'",\n')
        if (plutchik[e] == plutchik[1]) or (plutchik[e] == plutchik[3]):
            emotions_idx["Scared"]+=1
            output_emotions.write('"'+EMOTIONS['Scared'][emotions_idx["Scared"]%len(EMOTIONS['Scared'])]+'",\n')
        if (plutchik[e] == plutchik[2]) or (plutchik[e] == plutchik[5]):
            emotions_idx["Sad"]+=1
            output_emotions.write('"'+EMOTIONS['Sad'][emotions_idx["Sad"]%len(EMOTIONS['Sad'])]+'",\n')
        if (plutchik[e] == plutchik[4]):
            emotions_idx["Joyful"]+=1
            output_emotions.write('"'+EMOTIONS['Joyful'][emotions_idx["Joyful"]%len(EMOTIONS['Joyful'])]+'",\n')
    else:
        emotions_idx["Neutral"]+=1
        output_emotions.write('"'+EMOTIONS['Neutral'][emotions_idx["Neutral"]%len(EMOTIONS['Neutral'])]+'",\n')

print(str(emotions_idx["Mad"]) + " Mad")
print(str(emotions_idx["Scared"]) + " Scared")
print(str(emotions_idx["Joyful"]) + " Joyful")
print(str(emotions_idx["Powerful"]) + " Powerful")
print(str(emotions_idx["Peaceful"]) + " Peaceful")
print(str(emotions_idx["Sad"]) + " Sad")
print(str(emotions_idx["Neutral"]) + " Neutral")




