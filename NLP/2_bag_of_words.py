import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv('data.csv')
df['text'] = df['Make'] + ' ' + df['Model']

# Clean text
df['cleaned_text'] = df['text'].apply(lambda text: re.sub(r'[^\w\s]', '', text.lower()))

# Bag-of-Words: Count Occurrence
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(df['cleaned_text'])

print(f"Vocabulary size: {len(count_vectorizer.vocabulary_)}")
print(f"Feature matrix shape: {X_count.shape}")
print("First 5 feature vectors:")
print(X_count[:5].toarray())
print("\n" + "="*50 + "\n")

# Bag-of-Words: Normalized Count Occurrence
X_normalized = X_count.toarray() / np.sum(X_count.toarray(), axis=1, keepdims=True)
print("First 5 normalized feature vectors:")
print(X_normalized[:5])

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# Word2Vec Embeddings
tokenized_texts = [text.split() for text in df['cleaned_text']]
w2v_model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

print(f"TF-IDF Feature matrix shape: {X_tfidf.shape}")
print("First 5 TF-IDF feature vectors:")
print(X_tfidf[:5].toarray())

# Visualize Word2Vec embeddings
def plot_word_embeddings(model, words):
    word_vectors = np.array([model.wv[word] for word in words])
    result = PCA(n_components=2).fit_transform(word_vectors)
    plt.figure(figsize=(10, 8))
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.title('Word Embeddings Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()

# Uncomment to visualize top 20 words
plot_word_embeddings(w2v_model, list(w2v_model.wv.key_to_index.keys())[:20])
