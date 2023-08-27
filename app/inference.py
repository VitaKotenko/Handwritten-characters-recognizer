#!/usr/bin/env python
# coding: utf-8


import argparse
import cv2
import os
import tensorflow as tf
import numpy as np

tf.keras.utils.disable_interactive_logging()


def predict(directory):
    
    model = tf.keras.models.load_model("model.h5")
    characters_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',9:'J',10:'K',11:'L',12:'M',13:'N',15:'P',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z', 26: 0, 27: 1, 28: 2, 29: 3, 30: 4, 31: 5, 32: 6, 33: 7, 34: 8, 35: 9}

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        
        # checking if it is a file
        if os.path.isfile(f): 
            try:
                img = cv2.imread(f"{directory}/{filename}")[:,:,0]
                # resize image by specifying custom width and height
                img = cv2.resize(img, (28, 28))
                img = np.invert(np.array([img])) 
                prediction = model.predict(img)
                print (f"0{ord(str(characters_dict[np.argmax(prediction)]))}, /{directory}/{filename}")
                #print (f"{word_dict[np.argmax(prediction)]}, /{directory}/{filename}")
            except:
                print (f"Incorect format: {directory}/{filename}")
            


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

result = predict(args.input)


