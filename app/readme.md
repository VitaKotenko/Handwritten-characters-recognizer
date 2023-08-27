HANDWRITTEN NUMBER & LETTERS RECOGNIZER

This script can recognize small squared black&white image with single
handwritten character on it.

METHODS
The CNN model that was used we designed for training the model over the training dataset. 
For character classifying was used the CNN model which was designed for classifying handwritten letters. 
The model was trained over the training dataset and evaluated accuracy.
The dataset consist of two datasets: MNIST and A-Z Handwritten Alphabets in .csv format.
The dataset for this project contains 377694 images of characters of 28×2, all present in the form of a CSV file.


ACCURACY of model:
traih set:
    9443/9443 [==============================] - 185s 20ms/step - loss: 0.2130 - accuracy: 0.9410
test set:
    2361/2361 [==============================] - 12s 5ms/step - loss: 0.1239 - accuracy: 0.9660

INSTALL

python3 -m venv /app/myenv
source /app/myenv/bin/activate
pip3 install -r requirements.txt

USAGE
python3 /app/inference.py --input <path to the test data directory >


AUTHOR
Viktoriia Kotenko