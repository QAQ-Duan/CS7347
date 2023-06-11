import pandas as pd
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv('./trainDA.csv')
print("Data shape:",df.shape)

# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print("train:",train_df.shape)
print("val:",val_df.shape)
# 创建自定义数据集类
class NliDataset(Dataset):
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
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定义模型参数和超参数
model_name = 'bert-base-uncased'
max_length = 512
batch_size = 16
num_epochs = 3

# 加载BertTokenizer和BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 替换为你的类别数

# 创建数据集实例
train_dataset = NliDataset(train_df, tokenizer, max_length)
val_dataset = NliDataset(val_df, tokenizer, max_length)


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
    output_dir='./resultsForBertDA',            # 输出目录
    num_train_epochs=3,                # 训练的轮数
    per_device_train_batch_size=8,     # 训练时的批次大小
    per_device_eval_batch_size=8,      # 验证时的批次大小
    learning_rate=2e-5,                 # 学习率
    weight_decay=0.01,                  # 权重衰减
    logging_dir='./logs',               # 日志目录
    logging_steps=500,                  # 每隔多少步记录一次日志
    report_to="none",
    save_steps=5000,
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

# 开始微调和训练
print(trainer.predict(test_dataset=train_dataset))
print(trainer.predict(test_dataset=val_dataset))
# 保存微调后的模型
trainer.save_model("BertDATrainer")
model.save_pretrained('BertDAModel')