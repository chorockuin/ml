import numpy as np
import pandas as pd

d = [10,5,62,89]
i = ["Thomas","Emily","Sam","Scott"]
print(pd.Series(d))
print(pd.Series(d, i))
print(pd.Series(d, i, name="score"))
print(pd.Series({k:v for k, v in zip(i, d)}))

d = {"c1": [1,2,3,4], "c2":[5,6,7,8]}
print(pd.DataFrame(d))

d = np.random.rand(10,5)
i = list(range(1,11))
c = ["A","B","C","D","E"]
print(pd.DataFrame(d))
print(pd.DataFrame(d, i))
print(pd.DataFrame(d, i, c))

s1 = pd.Series(["Thomas", 80, "A"], index=["name", "score", "grade"], name=101)
s2 = pd.Series(["Emily", 45, "C"], index=["name", "score", "grade"], name=102)
df = pd.DataFrame([s1, s2])
print(df)
print(df.at[101, "name"])
print(df.at[102, "score"])
print(df.iat[0,0])
print(df.iat[1,1])
print(df["name"])
print(df[["name", "score"]])
print(df["name"][101])
print(df["name"].at[101])
print(df["name"].iat[0])
print(df.loc[101])
print(df.loc[[101, 102]])
print(df.loc[101]["name"])
print(df.loc[101].at["name"])
print(df.loc[101].iat[0])
print(df.iloc[0]["name"])
print(df.iloc[[0,1]]["name"])

d = np.random.rand(10,5)
i = list(range(11, 21))
c = ["A","B","C","D","E"]
df = pd.DataFrame(d, i, c)
print(df)
print(df>0.5)
print(df[df>0.5])
print(df["A"]>0.5)
print(df[df["A"]>0.5])
print(df.loc[11]>0.5)
print(df[(df["A"]>0.5) & (df["B"]>0.2)])

print(df.drop("A", axis=1))
print(df.drop(["A", "B"], axis=1))
print(df.drop(11, axis=0))
print(df.drop([11,12,13], axis=0))
df.drop("B", axis=1, inplace=True)
print(df)

d = np.random.rand(10,5)
d[4,2] = np.nan
d[2,4] = np.nan
d[3,4] = np.nan
d[7,3] = np.nan
i = list(range(101,111))
c = ["A","B","C","D","E"]
df = pd.DataFrame(d,i,c)
print(df.isna())
print(df.notna())
# print(df.fillna(-1))
print(df)
print(df.dropna(axis=0))
print(df.dropna(axis=1))
print(df.dropna(how="all", axis=0))
print(df.dropna(how="any", axis=0))
print(df.dropna(thresh=9, axis=1))
print(df.dropna(subset=["A","C"], axis=0))
print(df.dropna(axis=0, inplace=True))

d = np.arange(0,50).reshape(10,5)
df = pd.DataFrame(data=d, index=i, columns=c)
print(df)
print(df.set_index("A"))
df.set_index("B", inplace=True)
print(df)
print(df.reset_index(inplace=True))

d = {"class": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
"student_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
"math_score": [60, 70, 50, 64, 75, 20, 60, 90, 45],
"eng_score": [66, 56, 90, 34, 55, 56, 62, 44, 49]}
df = pd.DataFrame(d)
print(df)
df.set_index(["class","student_id"], inplace=True)
print(df)
print(df.xs("A"))
print(df.xs(1, level=1))
print(df.xs(1, level="student_id"))
print(df.xs(("A",1)))
print(df["math_score"])

d = {"class": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
"student_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
"math_score": [60, 70, 50, 64, 75, 20, 60, 90, 45],
"eng_score": [66, 56, 90, 34, 55, 56, 62, 44, 49]}
df = pd.DataFrame(data=d)
print(df)
print(df.groupby("class").mean())
print(df.groupby(by="class", as_index=False).mean())
df.set_index(["class", "student_id"], inplace=True)
print(df)
print(df.groupby(level=0).sum())
print(df.groupby(level=1).sum())
print(df.groupby(level="student_id").sum())
df.reset_index(inplace=True)
print(df)
print(df.groupby("class"))
for name, group in df.groupby("class"):
    print(name)
    print(group)
print(df.groupby("class").aggregate(np.sum))
print(df.groupby("class").aggregate(lambda x: sum(map(lambda a: a**2, x))))

d = np.random.randint(0,100,50).reshape(10,5)
i = list(range(101,111))
c = ["A","B","C","D","E"]
df = pd.DataFrame(data=d, index=i, columns=c)
print(df)
df["A"] = df["A"].map(lambda x: x*2)
print(df)
print(df.apply(lambda x: sum(x)/len(x), axis=1))
print(df.apply(lambda x: sum(x)/len(x), axis=0))
print(df.applymap(lambda x: 2*x))

d = {"class": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
"student_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
"math_score": [60, 70, 50, 64, 75, 20, 60, 90, 45],
"eng_score": [66, 56, 90, 34, 55, 56, 62, 44, 49]}
df = pd.DataFrame(data=d)
print(df)
print(df["class"].unique())
print(df["class"].value_counts())
print(df["math_score"].value_counts())
print(df.value_counts())

d1 = {"student_id": [101, 102, 103, 104, 105, 106], "math_score": [60, 70, 50, 64, 75, 20]}
d2 = {"student_id": [107, 108, 109], "math_score": [66, 72, 63]}
d3 = {"student_id": [101, 102, 103, 104, 105, 106], "eng_score": [34, 55, 56, 62, 44, 49]}
d4 = {"student_id": [104, 105, 106, 107, 108, 109], "eng_score": [30, 55, 56, 62, 44, 49]}
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
df3 = pd.DataFrame(d3)
df4 = pd.DataFrame(d4)
df1.set_index("student_id", inplace=True)
df2.set_index("student_id", inplace=True)
df3.set_index("student_id", inplace=True)
df4.set_index("student_id", inplace=True)
print(df1)
print(df2)
print(df3)
print(df4)
print(pd.concat([df1, df2], axis=0))
print(pd.concat([df1, df3], axis=0))
print(pd.concat([df1, df3], join="inner")) # intersection of column
print(pd.concat([df1, df2], join="inner")) # intersection of column
print(pd.merge(df1, df3, left_index=True, right_index=True)) # basically merge is intersection
print(pd.merge(df1.reset_index(), df3.reset_index(), on="student_id"))
print(pd.merge(df1.reset_index(), df3.reset_index(), left_on="student_id", right_on="eng_score"))
print(pd.merge(df1.reset_index(), df3.reset_index()))
print(df1.join(df3))
print(df1.join(df4))
print(df1.join(df4, how="left"))
print(df1.join(df4, how="right"))
print(df1.join(df4, how="outer"))
print(df1.join(df4, how="inner"))

d = {"class": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
"student_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
"math_score": [60, 70, 50, 64, 75, 20, 60, 90, 45],
"eng_score": [66, 56, 90, 34, 55, 56, 62, 44, 49]}
df = pd.DataFrame(data=d)
print(df.sort_values(by="math_score", ascending=True))
print(df.sort_values(by="math_score", ascending=False))
print(df.sort_values(by=["class", "math_score"], ascending=True))
print(df.sort_values(by=["class", "math_score"], ascending=[True,False]))

d = {"class": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
"student_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
"math_score": [60, 70, 50, 64, 75, 20, 60, 90, 45],
"eng_score": [66, 56, 90, 34, 55, 56, 62, 44, 49]}
df = pd.DataFrame(data=d)
print(df.pivot_table(columns="student_id", index="class"))
print(df.pivot_table(columns="student_id", index="class", values="eng_score"))
print(df.pivot_table(columns="student_id", index="class", values=["eng_score", "math_score"]))
print(df.pivot_table(columns="student_id"))
print(df.pivot_table(columns="student_id", aggfunc="mean"))
print(df.pivot_table(columns="student_id", aggfunc="sum"))

print(df)
df.set_index("class", inplace=True)
df.to_csv("test.csv")
df = pd.read_csv("test.csv")
print(df)
print(df.head())
