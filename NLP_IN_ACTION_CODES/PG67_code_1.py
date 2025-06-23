import numpy as np
sentence="Thomas Jefferson began building Monticello at the age of 26."
token_seq=sentence.split() #Primary dirty tokenizer
vocab=sorted(set(token_seq))
print(', '.join(vocab)) #num>Cap alpha>small alpha in sort order
num_tokens=len(token_seq) #number of tokens
vocab_size=len(vocab) # no of unique tokens
onehot_vector=np.zeros((num_tokens,vocab_size),int)
for i,word in enumerate(token_seq):
    onehot_vector[i,vocab.index(word)]=1
print(onehot_vector)
import pandas as pd #to improve visualisation
df=pd.DataFrame(onehot_vector,columns=vocab)
print(df.head(10))
