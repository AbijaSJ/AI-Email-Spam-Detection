
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "Win a free prize",
    "You won lottery",
    "Claim your gift",
    "Meeting tomorrow",
    "Send project report",
    "Class starts at 9"
]

labels = [
    "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

msg = input("Enter email: ")

result = model.predict(vectorizer.transform([msg]))

print("Email:", result[0])
