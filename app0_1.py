import pandas as pd
import sarc_detect2
import sarc_detect1
import streamlit as st
import pickle

df_1 = pd.read_csv('C:/Users/Hp/Documents/vscode_folder/major_project_part2sarc-Copy/dataset_f/india-news-headlines.csv')


df2 = df_1.head(10)
docs1 = df2['headline_text'].values
is_sarc = []

for i in docs1:
    is_sarc.append(sarc_detect2.predict(i))


# docs1 = df2['headline_text'].values
is_sarc2 = []
is_sarc1 = is_sarc
# for i in range(10,31):
#     is_sarc.append(sarc_detect2.predict(docs[i]))



# for i in range(31,101):
#     is_sarc.append(sarc_detect2.predict(docs[i]))

is_sarc1 = is_sarc

# val1 = len(docs)
# for i in range(101,val1-1):
#     is_sarc.append(sarc_detect2.predict(docs[i]))

# st.write(
#     is_sarc
# )
df3 = df_1.head(28619)
docs = df3['headline_text'].values
mydoc = pd.Series(docs)
# for row in mydoc.items():
#     is_sarc2.append(sarc_detect2.predict(row))
#     print(row)
for i,row in mydoc.items():
    is_sarc2.append(sarc_detect2.predict(row))
    print(i)

df3['is_sarcastic'] = is_sarc2
# st.write(
#     df3.head()
# )

dataframe_1 = df3.loc[:, ['publish_date','headline_text','is_sarcastic']]
dataframe_1.to_pickle('saved_dataframe.pkl')