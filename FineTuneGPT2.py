
import pandas as pd
import torch
from datasets import Dataset as Dataset1
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv('./train.tsv',  sep='\t', quoting=3)
df.head()

cleaned_df = df.dropna()
print(cleaned_df.shape)

train_df, val_df = train_test_split(cleaned_df, test_size=0.2, random_state=42)

class GPTDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence1 = self.data.iloc[index]['sentence1']
        sentence2 = self.data.iloc[index]['sentence2']
        label = self.data.iloc[index]['label']
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        label = label_map[label]
        
        encoding = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

max_length = 1024

# model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2ForSequenceClassification.from_pretrained('./FineTuneGPT2Model/')
model = model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

train_dataset = GPTDataset(train_df, tokenizer, max_length)
val_dataset = GPTDataset(val_df, tokenizer, max_length)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # 计算准确率
    acc = accuracy_score(labels, preds)

    # 计算精确率、召回率、F1值（针对每个类别）
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# 定义训练参数
training_args = TrainingArguments(
    output_dir='./resultsForGPT2-0525',            # 输出目录
    num_train_epochs=2,                # 训练的轮数
    per_device_train_batch_size=1,     # 训练时的批次大小
    per_device_eval_batch_size=1,      # 验证时的批次大小
    learning_rate=2e-5,                 # 学习率
    weight_decay=0.01,                  # 权重衰减
    logging_dir='./logsForGPT2-0525',               # 日志目录
    logging_steps=5000,                  # 每隔多少步记录一次日志
    report_to="tensorboard",
    save_steps=5000,  # 每隔多少步保存一次检查点
    gradient_accumulation_steps=8,
)

# 定义Trainer实例
trainer = Trainer(
    model=model,                         # 待微调的模型
    args=training_args,                   # 训练参数
    train_dataset=train_dataset,          # 训练数据集
    eval_dataset=val_dataset,             # 验证数据集
    compute_metrics=compute_metrics,      # 自定义的评估指标函数
)

# 开始微调和训练
trainer.train()
print(trainer.predict(test_dataset=val_dataset))
# 保存微调后的模型
trainer.save_model("FineTuneGPT2Trainer-0525")
model.save_pretrained('FineTuneGPT2Model-0525')


