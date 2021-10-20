'''
Created on Oct 13, 2021

@author: black
'''
import math
import nltk
from nltk import sent_tokenize
from nltk.tokenize import TweetTokenizer
from collections import Counter
from nltk.util import ngrams

#read text from file
def read_file(file):
    f = open(file, "r")
    text = f.read()
    return text

text = read_file("../train_data.txt");

#split text into sentences
sentences = sent_tokenize(text)
sentences_tokenized = [];

#tokenize sentences
tweet_wt = TweetTokenizer()

for sent in sentences:
    sent_tok = tweet_wt.tokenize(sent);
    sentences_tokenized.append(sent_tok);

#tokenize text 
tokens = tweet_wt.tokenize(text);

#find word frequencies
count = nltk.FreqDist(tokens)

#set vocabulary (distinct words with frequency >10)
vocabulary = set()
for key in tokens:
    if(count[key]>=10):
        vocabulary.add(key);

vocab_size = len(vocabulary)+1 #1 is the unknown token

#replace the unknown words in the training set with the token unknown
for sent in sentences_tokenized:
    for i in range(len(sent)):
        if(sent[i] not in vocabulary):
            sent[i] = 'UNK'

#hash tables with ngrams frequencies
unigram_counter = Counter()
bigram_counter = Counter()
trigram_counter = Counter()

for sent in sentences_tokenized:
    unigram_counter.update([gram for gram in ngrams(sent, 1, pad_left=True, pad_right=True,left_pad_symbol='<s>',right_pad_symbol='<e>') ])
    bigram_counter.update([gram for gram in ngrams(sent,2,pad_left=True, pad_right=True,left_pad_symbol='<s>', right_pad_symbol='</e>')])
    trigram_counter.update([gram for gram in ngrams(sent,3,pad_left=True, pad_right=True,left_pad_symbol='<s>', right_pad_symbol='</e>')])

print(bigram_counter.most_common(10))

alpha = 0.1;

#n gram propability for a sequence //Î•Î´Ï‰ Î¸ÎµÏ‰Ï�ÏŽ Î¿Ï„Î¹ sequence ÎµÎ¯Î½Î±Î¹ Î¼Î¹Î± Ï€Ï�Î¿Ï„Î±ÏƒÎ·
def findSequenceBigramPropability(sequence):
    seq_bigram_propability = 0;
    for i in range(0,len(sequence)-1):
        bigram_prob = (bigram_counter[(sequence[i], sequence[i+1])] + alpha) / (unigram_counter[(sequence[i],)] + alpha*vocab_size)
        seq_bigram_propability += math.log2(bigram_prob);
    return seq_bigram_propability

def findSequenceTrigramPropability(sequence):
    seq_trigram_propability = 0;
    for i in range(0,len(sequence)-2):
        trigram_prob = (trigram_counter[(sequence[i], sequence[i+1], sequence[i+2])] + alpha) / (bigram_counter[(sequence[i],sequence[i+1])] +alpha*vocab_size);
        seq_trigram_propability += math.log2(trigram_prob);
    return seq_trigram_propability

#bigram/trigram propability for the three first sentences in the training set
for i in range(3):
    print(sentences_tokenized[i])
    prob2 = findSequenceBigramPropability(sentences_tokenized[i])
    print("bigram_prob2: {0:.8f} ".format(prob2))
    prob3 = findSequenceTrigramPropability(sentences_tokenized[i])
    print("trigram_prob: {0:.8f} ".format(prob3))

''' upoerotima 2 '''
test_text = read_file("../test_data.txt")

test_sentences = sent_tokenize(test_text)
test_sentences_tokenized = [];

#tokenize sentences
for sent in test_sentences:
    sent=sent+' <e>'
    sent_tok = tweet_wt.tokenize(sent);
    test_sentences_tokenized.append(sent_tok);

count_bigrams =0;
#def findLanguageEntropy():
language_propabity = 0;
for sent in test_sentences_tokenized:
    for i in range(len(sent)):
        if(sent[i] not in vocabulary):
            sent[i] = 'UNK'
    language_propabity += findSequenceBigramPropability(sent)
    count_bigrams += len(sent)-1;

language_entropy = (-1)*language_propabity/count_bigrams
language_perplexity = math.pow(2,language_entropy)

print("language bigram_prob: {0:.8f} ".format(language_propabity))    
print(count_bigrams) 
print("language entropy: {0:.8f} ".format(language_entropy))    
print("language perpexity: {0:.8f} ".format(language_perplexity)) 

  
