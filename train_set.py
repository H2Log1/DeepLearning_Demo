import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv(r"D:\MyPython\DeepLearning\DeepLearning_Demo\titanic\train.csv")

# print(df)

train_data = df[:800]
train_data.reset_index(inplace=True)
train_data = train_data.drop(columns=["index"])
train_data.to_csv(r"D:\MyPython\DeepLearning\DeepLearning_Demo\titanic\train1.csv", index=False)

print(train_data)

validation_data = df[800:]
validation_data.reset_index(inplace=True)
validation_data = validation_data.drop(columns=["index"])
validation_data.to_csv(r"D:\MyPython\DeepLearning\DeepLearning_Demo\titanic\validation.csv", index=False)

print(validation_data)