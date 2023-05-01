from app import app
from app.inception_encoder import InceptionEncoder
import numpy as np
import os
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

tokenizer = Tokenizer()

tokenizer_name = "tokenizer.pkl"
tokenizer_path = os.path.join(app.config['BASEDIR'], 'tf_files', tokenizer_name)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 34

def predict_beam_search(image_path, beam_width, model):
    indeption = InceptionEncoder()
    # encode the image
    image_vec = indeption.encode(image_path)
    image_vec = image_vec.reshape(1, -1)
    # initialize the caption with the start token
    caption = tokenizer.texts_to_sequences(["startseq"])[0]
    # initialize beam search
    beam = [(caption, 0)]
    
    # loop until the end token or the maximum length is reached
    for i in range(max_length):
        # generate new candidates
        candidates = []
        for j in range(len(beam)):
            seq, score = beam[j]
            # check if the sequence ends with endseq
            if seq[-1] == tokenizer.word_index["endseq"]:
                candidates.append((seq, score))
                continue
            # predict the next word using the model
            padded_caption = pad_sequences([seq], maxlen=max_length, padding='post')
            prediction = model.predict([image_vec, padded_caption], verbose=0)[0]
            # get the top k words with the highest probability
            top_k = prediction.argsort()[-beam_width:][::-1]
            # add new candidates to the list
            for w in top_k:
                new_seq = seq + [w]
                new_score = score + np.log(prediction[w])
                candidates.append((new_seq, new_score))
        # select top k candidates
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
    # select the best candidate
    seq, score = beam[0]
    # convert the caption indices to words
    caption_words = tokenizer.sequences_to_texts([seq])[0].split()
    # join the words to form a sentence
    caption_sentence = ' '.join(caption_words[1:-1])
    return caption_sentence