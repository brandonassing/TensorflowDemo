import tensorflow as tf
from tensorflow import keras

# helper funcs to decode/encode reviews
# A dictionary mapping words to an integer index
imdb = keras.datasets.imdb
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
# print(word_index)
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def encode_review(text):
    text = text.split(' ')
    encoded_text = []
    for i in text:
        encoded_text.append(word_index.get(i, 2))
    return encoded_text

# padding
def pad(data):
	return keras.preprocessing.sequence.pad_sequences(data, value=word_index["<PAD>"], padding='post', maxlen=256)

