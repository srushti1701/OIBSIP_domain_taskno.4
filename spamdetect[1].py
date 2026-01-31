import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("spam.csv", encoding="latin-1")


df = df.iloc[:, :2]
df.columns = ["label", "message"]

print("✅ First 5 rows:\n", df.head())
print("\n✅ Label counts:\n", df["label"].value_counts())


df["label"] = df["label"].map({"ham": 0, "spam": 1})


X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)


model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("nb", MultinomialNB())
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


acc = accuracy_score(y_test, y_pred)
print("\n✅ Accuracy:", acc*100, "%")


print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
print("\n✅ Confusion Matrix:\n", cm)


print("\n✅ Try your own email text")
msg = input("Enter Email/SMS Text: ")
pred = model.predict([msg])[0]

if pred == 1:
    print("🚨 SPAM Message Detected")
else:
    print("✅ NOT SPAM (Ham)")

