import numpy as np
import pandas as pd

# https://www.kaggle.com/austinreese/craigslist-carstrucks-data?select=vehicles.csv

# df = pd.read_csv('data/vehicles.csv')
df = pd.DataFrame({'name': ['ji', 'kim', 'song', 'yi', 'jang', 'min', 'yang', 'jo'], 
                   'region': ['seoul', 'busan', 'seoul', 'seoul', 'busan', 'gwangju', np.nan, 'kangwon']})
print(df)
# drop_first : one class is decided, so no need to count
# dummy_na : count np.nan
df_region = pd.get_dummies(df['region'], drop_first=True, dummy_na=True)
print(df_region)

df = pd.concat([df.drop('region', axis=1), df_region], axis=1)
print(df)