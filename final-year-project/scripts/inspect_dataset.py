import pandas as pd
path = r'd:\Final Year Project\data\train_001_final.xlsx'
df = pd.read_excel(path)
print(df.columns)
print(df.head())
