import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.DataFrame([
    ['Global', 'PC', 542],
    ['Global', 'Mac', 45],
    ['Global', 'PS5', 23],
    ['Global', 'XBoxSX', 34],
    ['Asia', 'PC', 42],
    ['Asia', 'Mac', np.nan],
    ['Asia', 'PS5', 7],
    ['Asia', 'XBoxSX', 3],
    ['Europe', 'PC', np.nan],
    ['Europe', 'Mac', 23],
    ['Europe', 'PS5', 12],
    ['Europe', 'XBoxSX', 21]
    ],
    columns=['region','platform', 'sales'])
print(df)

print(df.dropna(axis=0))

print(df.dropna(axis=1))

imputer = SimpleImputer(strategy='mean')
print(imputer.fit_transform(df[['sales']]))

imputer = SimpleImputer(strategy='median')
print(imputer.fit_transform(df[['sales']]))