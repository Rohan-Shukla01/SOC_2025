sentence="Thomas Jefferson began building Monticello at the age of 26."
sentence_bow={}
for token in sentence.split():
    sentence_bow[token]=1
print(sorted(sentence_bow.items()))
#another efficient way is to use pandas series
import pandas as pd
df=pd.DataFrame(pd.Series(dict((token,1) for token in sentence.split())),columns=['sent']).T
print(df)