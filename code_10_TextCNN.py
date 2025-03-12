# -*- coding: utf-8 -*-
"""
完全适配 PyTorch 2.5.1+ 和 torchtext 0.16.0+ 的最终修正版
"""
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import spacy

# 初始化配置
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################### 修正的数据处理部分 ###################
# 加载spacy模型
print("正在加载spacy模型...")
nlp = spacy.load("en_core_web_sm")
tokenizer = get_tokenizer("spacy", language="en_core_web_sm")  # 统一使用这个分词器

# 加载数据集
print("正在加载IMDB数据集...")
train_iter, test_iter = IMDB(split=('train', 'test'))
train_data = list(train_iter)
test_data = list(test_iter)


# 构建词汇表
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield [token.lower() for token in tokenizer(text)]


print("正在构建词汇表...")
vocab = build_vocab_from_iterator(
    yield_tokens(train_data),
    max_tokens=25000,
    specials=['<unk>', '<pad>']
)
vocab.set_default_index(vocab['<unk>'])

# 加载预训练词向量
print("正在加载GloVe词向量...")
glove = GloVe(name='6B', dim=100)

# 构建词向量矩阵
print("对齐词向量...")
vocab_size = len(vocab)
embedding_dim = 100
vectors = torch.zeros((vocab_size, embedding_dim))
for i, word in enumerate(vocab.get_itos()):
    if word in glove.stoi:
        vectors[i] = glove[word]
    elif word == '<pad>':
        vectors[i] = torch.zeros(embedding_dim)
    else:  # 处理未登录词
        vectors[i] = torch.randn(embedding_dim) * 0.1

# 数据预处理管道
text_pipeline = lambda x: [vocab[token.lower()] for token in tokenizer(x)]
label_pipeline = lambda x: 1.0 if x == 'pos' else 0.0


# 数据批处理函数
def collate_batch(batch):
    text_list, label_list = [], []
    for (label, text) in batch:
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(label_pipeline(label))

    padded_text = pad_sequence(text_list,
                               batch_first=True,
                               padding_value=vocab['<pad>'])
    return padded_text.to(device), torch.tensor(label_list).to(device)


# 拆分验证集
train_data, valid_data = random_split(
    train_data,
    [len(train_data) - 5000, 5000],
    generator=torch.Generator().manual_seed(1234)
)

# 创建DataLoader
BATCH_SIZE = 64
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)


################### 模型定义 ###################
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.mish = Mish()

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        conved = []
        for conv in self.convs:
            conv_out = self.mish(conv(embedded)).squeeze(3)
            pooled = F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2)
            conved.append(pooled)
        cat = self.dropout(torch.cat(conved, dim=1))
        return self.fc(cat)


# 初始化模型
print("初始化模型...")
PAD_IDX = vocab['<pad>']
model = TextCNN(
    vocab_size=len(vocab),
    embedding_dim=100,
    n_filters=100,
    filter_sizes=[3, 4, 5],
    output_dim=1,
    dropout=0.5,
    pad_idx=PAD_IDX
).to(device)

# 加载预训练权重
model.embedding.weight.data.copy_(vectors.to(device))


################### 训练流程 ###################
def binary_accuracy(preds, y):
    rounded = torch.round(torch.sigmoid(preds))
    correct = (rounded == y).float()
    return correct.sum() / len(correct)


def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    for texts, labels in loader:
        optimizer.zero_grad()
        preds = model(texts).squeeze(1)
        loss = criterion(preds, labels)
        acc = binary_accuracy(preds, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    with torch.no_grad():
        for texts, labels in loader:
            preds = model(texts).squeeze(1)
            loss = criterion(preds, labels)
            acc = binary_accuracy(preds, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)


if __name__ == '__main__':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss().to(device)

    N_EPOCHS = 5
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
        end = time.time()

        mins, secs = int((end - start) / 60), int((end - start) % 60)
        print(f'Epoch: {epoch + 1:02} | Time: {mins}m {secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'textcnn.pt')

    model.load_state_dict(torch.load('textcnn.pt'))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f'\nTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


    # 修正后的预测函数
    def predict(text):
        model.eval()
        # 使用训练时相同的分词器
        tokens = [token.lower() for token in tokenizer(text)]

        # 确保最小长度满足最大卷积核尺寸
        min_len = max([3, 4, 5])
        if len(tokens) < min_len:
            tokens += ['<pad>'] * (min_len - len(tokens))

        # 转换为索引
        indices = [vocab[token] for token in tokens]
        tensor = torch.LongTensor(indices).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()
        return prob


    print("\n预测示例:")
    test_samples = [
        "This movie is terrible",
        "I love this film!",
        "What a brilliant masterpiece"
    ]
    for text in test_samples:
        print(f"{text} -> {predict(text):.4f}")