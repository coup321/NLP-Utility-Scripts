import numpy as np
import nltk
import functools
import string
import re

class BasicTextCleaner:
    """
    This is text preprocesser that can perform the following tasks:
        - remove digits
        - remove puncutation
        - remove stopwords
    Inputs:
         - text -> str - this is the text to be processed
         - lower -> bool - indicate if the string should be made lower or not
                           default is True

    Attrs:
        - self.text
        - self.lower

    Methods:
        - process_text - returns string of fully processed text (str)
        - remove_digits - returns only digit removal (str)
        - remove_punctuation - returns only punctuation removal (str)
        - remove_stopwords - removes only stopword removal (str)

    Basic use example:
    text = 'A 5 inch mouse (!) jumped over the LAZY dog'
    Cleaner = BasicTextCleaner(text, lower=True)
    print(Cleaner.process_text())
    > 'a inch mouse jumped over the lazy dog'

    """
    def __init__(self, text, lower=True):
        self.text = text
        self.lower = lower

    def process_text(self):
        #set string to lower case if self.lower is true
        if self.lower:
            self.text = self.text.lower()

        #uses _compose to create a sequence of functions to be
        #applied to the inputs. Additional functions could
        #easily be added to this sequence of functions
        #they will be applied in order (similar to tf.Sequential)
        composed_fn = self._compose(
            self.remove_digits,
            self.remove_punctuation,
            self.remove_stopwords)

        return composed_fn(self.text)

    def _compose(self, *functions):
        #returns a function that sequentially applies each function to an input
        return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)    

    def remove_stopwords(self):
        stopwords_en = set(nltk.corpus.stopwords.words('english'))
        removed = [word for word in self.text.split() if word not in stopwords_en]
        return ' '.join(removed)

    def remove_digits(self):
        return ''.join([c for c in self.text if not c.isdigit()])

    def remove_punctuation(self):
        return re.sub('[' + re.escape(string.punctuation) + r'\\\n]', ' ', self.text)

class BasicTokenizer:
    """
    Creates a vocabulary dictionary from string input and then 
    generates one-hot-encoding for words in sentences. 

    Input
        - text -> str

    Attrs:
        - self.text - input text
        - self.vocab - initially set to None, returns vocab dict after it is generated
    
    Methods
        - tokenize(self) - builds vocab from self.text and then returns one
                           hot representation of self.text as numpy array
        - _build_vocab(self) - generates vocab based on self.text
        - word2num(word) - generates the word index number (ex: the -> 5)
        - num2word(nuM) - generates the word from a word index number (ex: 5 -> the)

    Use example:
    text = 'A 5 inch mouse (!) jumped over the LAZY dog'
    tok = BasicTokenizer(text)
    one_hot_embedding = tok.tokenize()

    """

    def __init__(self, text):
        self.text = text
        self.vocab = None

    def tokenize(self):
        if not self.vocab:
            self._build_vocab()

        #converts list of words to list of numbers (ints)
        nums = [self.word2num(w) for w in self.text.split(' ')]

        n = len(nums)
        #create zero array of shape(num_words, vocab_size)
        encoding = np.zeros((len(self.vocab),n))
        #for each word, make it's index = 1 (for the one hot encoding)
        for i in range(n):
            encoding[nums[i]][i] = 1
        return encoding #shape(num_words, vocab_size)
    
    def _build_vocab(self):
        words = enumerate(set(self.text.split(' ')))
        self.vocab = {word:num for num, word in words}
    
    def word2num(self, word):
        if not self.vocab:
            self._build_vocab()
        #for each word return the vocab dictionary number 
        if word in self.vocab.keys():
            return self.vocab[word]
        return None
    
    def num2word(self, num):
        if not self.vocab:
            self._build_vocab()
        #iterate through items and return the word when index number is found
        #not a very performant implementation.
        for word, number in self.vocab.items():
            if num == number:
                return word
        return None