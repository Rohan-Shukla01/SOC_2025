sentence="Thomas Jefferson  began building Monticello at the age of 26.\n"
sentence+="Construction was done mostly by local masons and\
    carpenters.\n"
sentence+="He moved into the South Pavilion in 1770.\n"
sentence+="Turning Monticello into a neoclassical masterpiece\
    was Jefferson's obsession."
corpus={}
for i, sent in enumerate(sentence.split('\n')):
    corpus[f'sent{i}']=dict((tok,1) for tok in sent.split())
import pandas as pd
df=pd.DataFrame(corpus).fillna(0).astype(int).T
print(df.iloc[:,0:10])