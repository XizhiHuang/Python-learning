import pandas as pd

df = pd.read_excel(r"C:/Users/Xizhi Huang/Desktop/53696.xlsx", engine='openpyxl')
print(df.head)
df_sum = df.groupby(['年'])['降雨量'].sum()
df_sum.to_csv('C:/Users/Xizhi Huang/Desktop/tmp912.txt', encoding='utf-8', index=True)
