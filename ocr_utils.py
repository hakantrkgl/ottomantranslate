# ocr_utils.py

import os
import cv2 as cv
from character_segmentation import segment
from train import prepare_char, featurizer
import pickle

model_name = '2L_NN.sav'

def load_model():
    location = 'models'
    if os.path.exists(location):
        model = pickle.load(open(f'models/{model_name}', 'rb'))
        return model


def run2(obj):
    word, line = obj
    model = load_model()
    char_imgs = segment(line, word)
    txt_word = ''
    for char_img in char_imgs:
        try:
            ready_char = prepare_char(char_img)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            continue
        feature_vector = featurizer(ready_char)
        predicted_char = model.predict([feature_vector])[0]
        if len(predicted_char) == 1 and predicted_char.isalpha():  # Sadece bir karakter ve harf ise ekle
            txt_word += predicted_char
        else:
            print(f"Geçersiz karakter tahmini: {predicted_char}")
    return txt_word
