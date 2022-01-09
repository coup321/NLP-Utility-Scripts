import numpy as np

class BasicTokenizer:
    def __init__(self, text):
        self.text = text
        self.vocab = None
    
    def build_vocab(self, text):
        words = enumerate(set(text.split(' ')))
        self.vocab = {word:num for num, word in words}
    
    def word2num(self, word):
        if not self.vocab:
            self.build_vocab(self.text)
        if word in self.vocab.keys():
            return self.vocab[word]
        return None
    
    def num2word(self, num):
        if not self.vocab:
            self.build_vocab(self.text)
        for word, number in self.vocab.items():
            if num == number:
                return word
        return None

    def tokenize(self):
        if not self.vocab:
            self.build_vocab(self.text)

        nums = [self.word2num(w) for w in self.text.split(' ')]

        n = len(nums)
        zeros = np.zeros((len(self.vocab),n))
        for i in range(n):
            zeros[nums[i]][i] = 1
        return zeros

sentence1 = 'the quick brown fox jumped over the lazy fox'
sentence2 = 'you are not from around here are you?'
sentence3 = 'you, there, in the mountains'
sentences = [sentence1, sentence2, sentence3]

tok = BasicTokenizer(sentence1)
print(tok.text)
print(tok.tokenize())