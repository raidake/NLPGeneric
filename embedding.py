from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from collections import Counter
import gensim.downloader as api
import numpy as np
import fasttext
import fasttext.util
import json


dataset = load_dataset("rotten_tomatoes")
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

# Download required tokenizer
nltk.download('punkt')

# Tokenize and build the vocabulary from the training dataset
vocab = Counter({"<UNK>": 0, "<pad>": 1})

# Tokenize and update vocabulary
for item in train_dataset:
    tokens = word_tokenize(item['text'].lower())  # Tokenizing and converting to lowercase
    vocab.update(tokens)

# Get the size of the vocabulary
vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")

# Display the top 10 most common words in the dataset (can delete this, for fun :)
print("Most common words in the dataset:", vocab.most_common(10))

# Load pre-trained Word2Vec model
word2vec_model = api.load('glove-wiki-gigaword-100')

# Check OOV words (words in training data but not in the pre-trained embeddings)
oov_words_word2vec = [word for word in vocab if word not in word2vec_model]
oov_count_word2vec = len(oov_words_word2vec)

print(f"Number of OOV words: {oov_count_word2vec}")
print("Some OOV words:", oov_words_word2vec[:10])

# Replace OOV

fasttext.util.download_model('en', if_exists='ignore')  # Downloads 'cc.en.300.bin'
fasttext_model = fasttext.load_model('cc.en.300.bin')
print(fasttext_model.dimension)
fasttext.util.reduce_model(fasttext_model, 100)
print(fasttext_model.dimension)


embedding_dim = 100 # match input dimension of layer, dimension 300 because fasttext model default is 300 dimensions
embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))

word_index = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}

with open('vocab.json', 'w') as f:
    json.dump(word_index, f)
print(word_index)
for word, i in word_index.items():
    if word in word2vec_model:
        # If word is in Word2Vec, use its vector
        embedding_matrix[i] = word2vec_model[word]
    else:
        #embedding_matrix[i] = np.random.normal(size=(embedding_dim,))  # Random for OOV words
        # If OOV, use FastText to generate embedding
        embedding_matrix[i] = fasttext_model.get_word_vector(word)
embedding_matrix = np.array(embedding_matrix, dtype=np.float32)
print(embedding_matrix)
np.save('embedded_matrix.npy', embedding_matrix)

