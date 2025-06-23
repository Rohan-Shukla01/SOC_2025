import pandas as pd
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
import fitz
import string
def extract_text(file_path):
    doc=fitz.open(file_path)
    txt=""
    for page in doc:
        txt=txt+page.get_text()
    return txt
tokenizer=TreebankWordTokenizer()

doc1=extract_text(r'C:\Users\Rohan\Desktop\1st sem\cs101\asmt4.pdf')
doc2=extract_text(r"C:\Users\Rohan\Desktop\1st sem\cs101\asm3.pdf")
doc1_tokenized=tokenizer.tokenize(doc1)
doc2_tokenized=tokenizer.tokenize(doc2)
doc1_=set([i for i in doc1_tokenized if i not in string.punctuation])
doc2_=set([i for i in doc2_tokenized if i not in string.punctuation])
vocab=doc1_.union(doc2_)
d={"d1":{},"d2":{}}
for i in vocab:
    if i in doc1_:
        d["d1"][i]=1
    if i in doc2_:
        d["d2"][i]=1
df=pd.DataFrame(d).fillna(0).astype(int).T
df2=df.T
d1_norm=np.sqrt(df2.d1.dot(df2.d1))
d2_norm=np.sqrt(df2.d2.dot(df2.d2))
print((df2.d1.dot(df2.d2))/(d1_norm*d2_norm))




