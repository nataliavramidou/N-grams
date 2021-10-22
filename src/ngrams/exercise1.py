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

#n gram propability for a sequence //ÃŽâ€¢ÃŽÂ´Ã�â€° ÃŽÂ¸ÃŽÂµÃ�â€°Ã�ï¿½Ã�Å½ ÃŽÂ¿Ã�â€žÃŽÂ¹ sequence ÃŽÂµÃŽÂ¯ÃŽÂ½ÃŽÂ±ÃŽÂ¹ ÃŽÂ¼ÃŽÂ¹ÃŽÂ± Ã�â‚¬Ã�ï¿½ÃŽÂ¿Ã�â€žÃŽÂ±Ã�Æ’ÃŽÂ·
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

'''
#bigram/trigram propability for the three first sentences in the training set
for i in range(3):
    print(sentences_tokenized[i])
    prob2 = findSequenceBigramPropability(sentences_tokenized[i])
    print("bigram_prob2: {0:.8f} ".format(prob2))
    prob3 = findSequenceTrigramPropability(sentences_tokenized[i])
    print("trigram_prob: {0:.8f} ".format(prob3))
'''

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
print("language entropy: {0:.8f} ".format(language_entropy))    
print("language perpexity: {0:.8f} ".format(language_perplexity)) 

'''beam search'''
def LD(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1
       
    res = min([LD(s[:-1], t)+1,
               LD(s, t[:-1])+1, 
               LD(s[:-1], t[:-1]) + cost])

    return res

sequence = input('Give a sequence')
sequence = '<s> '+sequence;
sequArr = tweet_wt.tokenize(sequence)
word = sequArr[1];

edtdistance = dict()
def findWordEditDistances(word):
    for token in vocabulary:
        distance = LD(word, token)
        if(distance<3):
            edtdistance[token] = distance

    return edtdistance 

edtdistance = findWordEditDistances(word);


#πρεπει να βρω και τα λ1,λ2

def findBigramPropability(start, end):
    bigram_prob = (bigram_counter[(start, end)] + alpha) / (unigram_counter[(start,)] + alpha*vocab_size)
    seq_bigram_propability = math.log2(bigram_prob);
    return seq_bigram_propability

'''
most_propable = dict()
for key in edtdistance.keys():
    pEditDistance = 1 / (edtdistance[key]+1)
    LK = -findBigramPropability(sequArr[0], key) -math.log2(pEditDistance)
    #print(key, edtdistance[key])
    #print(LK)
    most_propable[key] = LK

sort_words = sorted(most_propable.items(), key=lambda x: x[1], reverse=False)
word1 = sort_words[0][0]
word2 = sort_words[1][0]
print(word1)
print(word2)
'''

'''έστω ότι κρατάμε τις δύο πρώτες λέξεις.'''
word1 = sequArr[0];
word2 = sequArr[0]
for k in range(1,len(sequArr)):
    edtdistance = findWordEditDistances(sequArr[k]);
    most_propable = dict()
    #most_propable2 = dict()
    for key in edtdistance.keys():
        pEditDistance = 1 / (edtdistance[key]+1)
        LK1 = -(0.6)*findBigramPropability(word1, key) -(0.4)*math.log2(pEditDistance)
        #LK2 = -findBigramPropability(word2, key) -math.log2(pEditDistance)
        #print(key, edtdistance[key])
        #print(LK)
        most_propable[key] = LK1
        #most_propable2[key] = LK2
    sort_words = sorted(most_propable.items(), key=lambda x: x[1], reverse=False)
    #sort_words2 = sorted(most_propable2.items(), key=lambda x: x[1], reverse=False)
    word1 = sort_words[0][0]
    word2 = sort_words[1][0]
    print(word1)
    #print(word2)







   



    







