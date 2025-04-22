# Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, WhitespaceTokenizer, RegexpTokenizer, TreebankWordTokenizer, TweetTokenizer, MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Sample sentence for demonstration
sample_sentence = "Hello world! This is an example sentence for NLP tokenization, stemming, and lemmatization. I'm using NLTK for this purpose. Let's see how it works with words like 'running', 'ran', and 'runs'."

print("Original Sample Sentence:")
print(sample_sentence)

# STEP 1: Tokenization

# Different Tokenization techniques 
# tokenizer = WhitespaceTokenizer()
# tokenizer = RegexpTokenizer(r'\w+')  # Removes punctuation and gets only words
# tokenizer = TreebankWordTokenizer()
# tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
# tokenizer = MWETokenizer([('NLTK', 'for'), ('how', 'it')])
# tokens = tokenizer.tokenize(sample_sentence)

#Default NLTK Word Tokenization
print("NLTK Word Tokenization:")
default_word_tokens = word_tokenize(sample_sentence)
print(default_word_tokens)
print(f"Number of tokens: {len(default_word_tokens)}")
print("\n")

# NLTK Sentence Tokenization
print("NLTK Sentence Tokenization:")
sentence_tokens = sent_tokenize(sample_sentence)
print(sentence_tokens)
print(f"Number of sentences: {len(sentence_tokens)}")
print("\n")

# STEP 2: Stemming
print("STEMMING TECHNIQUES")

# Porter Stemmer
print("Porter Stemmer:")
porter_stemmer = PorterStemmer()
porter_stemmed_words = [porter_stemmer.stem(word) for word in default_word_tokens]
print(porter_stemmed_words)
print("\n")

# Snowball Stemmer (Porter2)
print("Snowball Stemmer (Porter2):")
snowball_stemmer = SnowballStemmer('english')
snowball_stemmed_words = [snowball_stemmer.stem(word) for word in default_word_tokens]
print(snowball_stemmed_words)
print("\n")

# Sample words to compare stemming
test_words = ["running", "runs", "ran", "easily", "fairly", "programming", "programmers"]
print("Stemming comparison for words:", test_words)
print("Porter Stemmer & Snowball Stemmer:")
print("Word\t\tPorter Stem  |  Snowball Stem")
for word in test_words:
    print(f"{word} -> {porter_stemmer.stem(word)} | {snowball_stemmer.stem(word)}")
print("\n" + "="*80 + "\n")

# STEP 3: Lemmatization
print("LEMMATIZATION")

lemmatizer = WordNetLemmatizer()
simple_lemmatized_words = [lemmatizer.lemmatize(word) for word in default_word_tokens]
print(simple_lemmatized_words)
print("\n")
