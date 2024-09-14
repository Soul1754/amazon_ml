import pandas as pd
import spacy
from spacy.tokens import DocBin
import re


# Define a function for basic text cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    return text


df = pd.read_csv("train_split.csv")
df = df[:4000]
df1 = pd.read_csv("dataset/train_text.csv")
result = pd.concat([df, df1], axis=1)

print("THIS is result")
print(result.head(3))

nlp = spacy.blank("en")
db = DocBin()

for index, row in result.iterrows():
    text = clean_text(row.get('Text', ''))
    entity_name = clean_text(row.get('entity_name', ''))
    entity_value = clean_text(row.get('entity_value', ''))

    print(text)

    if pd.isna(text) or pd.isna(entity_value):
        continue

    start_idx = text.find(entity_value)
    end_idx = start_idx + len(entity_value)

    if start_idx != -1:
        doc = nlp.make_doc(text)
        ents = [(start_idx, end_idx, entity_name)]

        entities = []
        for start, end, label in ents:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                entities.append(span)

        doc.ents = entities
        db.add(doc)

db.to_disk("./train.spacy")
