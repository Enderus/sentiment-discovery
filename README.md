NVIDIA's REPO: https://github.com/NVIDIA/sentiment-discovery/blob/master/

Finetuned Plutchik mLSTM model: https://drive.google.com/file/d/1ieiWFrYBqzBgGPc3R36x9oL7vlj3lt2F/view

Included in this repo is a script: extract_emotions.py, that runs the pretrained model on a configurable corpus. And returns the list of sentences labled with the detected emotion(s) for each given sentence. If the model does not classify any of the Plutchik emotions above a defined threshold of likelihood it will be labled neutral. The output is configured such that each sentence can have a maximum of two of the classified emotions.