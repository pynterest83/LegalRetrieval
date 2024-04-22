import nltk
import string
import re

# Tokenization
def lowercase_text(text):
    return text.lower()

input_str = "Weather is too Cloudy.Possiblity of Rain is High,Today!!"
print(lowercase_text(input_str))

# Remove Numbers
def remove_num(text):
    result = re.sub(r'\d+', '', text)
    return result

input_s = "You bought 6 candies from shop, and 4 candies are in home."
print(remove_num(input_s))

# Remove Punctuation
def rem_punct(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

input_str = "Hey, Are you excited??, After a week, we will be in Shimla!!!"
print(rem_punct(input_str))

# importing nltk library
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet='true')
nltk.download('punkt', quiet='true')

# remove stopwords function
def rem_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text

ex_text = "Data is the new oil. A.I is the last invention"
print(rem_stopwords(ex_text))

# Vietnamese Tokenization
from pyvi import ViTokenizer

line = "Phân loại văn bản tự động bằng Machine Learning"
lines = ViTokenizer.tokenize(line)
print(lines)