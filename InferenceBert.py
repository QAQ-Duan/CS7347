import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('./BertDAModel/')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = model.to(device)
print("device:",device)
print(model)

test_file = './test.tsv'
df = pd.read_csv(test_file, delimiter="\t", quoting=3)
print(df.shape)
ids = df["id"].tolist()
sentences1 = df["sentence1"].tolist()
sentences2 = df["sentence2"].tolist()

predictions = []
category_labels = ['entailment', 'contradiction', 'neutral']
for text1, text2 in tqdm(zip(sentences1, sentences2), total=len(sentences1), desc="Inference"):
    inputs = tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_category = torch.argmax(logits, dim=1).item()
    predicted_label = category_labels[predicted_category]
    predictions.append(predicted_label)

results = pd.DataFrame({"id": ids, "Category": predictions})
results.to_csv("results-BertDA.csv", index=False)

print(results.shape)