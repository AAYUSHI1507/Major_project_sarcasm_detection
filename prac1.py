import pandas as pd

df = pd.read_json('C:/Users/Hp/Documents/vscode_folder/major_project_part2/dataset_f/Sarcasm_Headlines_Dataset_v2.json', lines=True)
df1 = df.head()
print(df1['headline'])
count = 0
for i in df1.items():
    print(i)
    count = count + 1
    print("#############")
    print("This is ",count)

df1_1 = df.loc[:, ['headline','is_sarcastic']]
print("This is updated dataframe \n",df1_1.head())