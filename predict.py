import tensorflow as tf
from tensorflow import keras
from helpers import decode_review, encode_review, pad

# predictions
new_model = tf.keras.models.load_model('movie_review.model')
new_reviews = [encode_review("this was the worst movie I've ever seen. I could not believe how horrible it was. 0/10 terrible experience"), encode_review("I enjoyed this movie so much, it was so great!!! The absolute best"), encode_review("it was ok")]
new_reviews = pad(new_reviews)
new_pred = new_model.predict([new_reviews])
print("custom:", new_pred)


