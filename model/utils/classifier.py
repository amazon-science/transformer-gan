import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import (
    BertConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BertForMaskedLM
)

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from abc import abstractmethod


class Metrics:
    def __init__(self, name='Metric'):
        self.name = name

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class TextDataset(Dataset):
    def __init__(self, test_text, real_text, split, block_size=128, train_size=5000, eval_size=2000
                 ):

        self.real_examples = []
        self.real_labels = []
        self.gen_examples = []
        self.gen_labels = []

        for real_data in real_text:
            tokenized_text = real_data

            # print("processing", path)
            # We ignore last block
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.real_examples.append(
                    tokenized_text[i: i + block_size]
                )
                self.real_labels.append(0)

        for gen_data in test_text:
            tokenized_text = gen_data

            # print("processing", path)
            # We ignore last block
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.gen_examples.append(
                    tokenized_text[i: i + block_size]
                )
                self.gen_labels.append(1)

        if "train" in split:
            self.real_examples = self.real_examples[:int(0.8 * len(self.real_examples))]
            self.real_labels = self.real_labels[:int(0.8 * len(self.real_labels))]
            self.gen_examples = self.gen_examples[:int(0.8 * len(self.gen_examples))]
            self.gen_labels = self.gen_labels[:int(0.8 * len(self.gen_labels))]
            NUM = train_size

        else:
            self.real_examples = self.real_examples[int(0.8 * len(self.real_examples)):]
            self.real_labels = self.real_labels[int(0.8 * len(self.real_labels)):]
            self.gen_examples = self.gen_examples[int(0.8 * len(self.gen_examples)):]
            self.gen_labels = self.gen_labels[int(0.8 * len(self.gen_labels)):]
            NUM = eval_size

        # Shuffle samples

        self.examples = self.real_examples[:NUM] + self.gen_examples[:NUM]
        self.labels = self.real_labels[:NUM] + self.gen_labels[:NUM]
        del self.real_examples
        del self.gen_examples
        del self.real_labels
        del self.gen_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        data = self.examples[item].long()
        labels = torch.tensor([self.labels[item]], dtype=torch.long)
        return data, labels


class Classifier(Metrics):
    def __init__(self, name=None, test_text=None, real_text=None, device='cpu', if_use=False, seq_len=128,
                 batch_size=20, model_name_or_path="spanBERT-512-update/checkpoint-1969000"):
        super(Classifier, self).__init__(name)

        self.if_use = if_use

        if not if_use:
            return
        self.test_text = test_text
        self.real_text = real_text
        self.train_size = 5000
        self.eval_size = 1000
        self.batch_size = batch_size

        self.block_size = seq_len

        self.model_name_or_path = model_name_or_path

        config_class, model_class = (BertConfig, BertForMaskedLM)
        config = config_class.from_pretrained(self.model_name_or_path)
        config.num_labels = 2

        self.model = model_class.from_pretrained(
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=config,
            cache_dir=None,
        )

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def get_score(self):

        if not self.if_use:
            return 0

        train_sampler = SequentialSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
        )

        # Train
        self.model.eval()
        X = np.array([])
        y = np.array([])
        for batch in train_dataloader:
            inputs, labels = batch
            inputs = inputs.to(self.device)

            y = np.concatenate([y, labels.cpu().numpy().squeeze()], 0) if y.size else labels.numpy().squeeze()

            with torch.no_grad():
                outputs = self.model(inputs)[0].cpu().numpy().squeeze()
                outputs = np.max(outputs, axis=1)
                X = np.concatenate([X, outputs], 0) if X.size else outputs

        # Define scikit learn model

        # Normalize data
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        clf = svm.LinearSVC(max_iter=10000, dual=False)
        clf.fit(X, y)

        # Compute train_acc
        y_train = clf.predict(X)
        train_acc = accuracy_score(y, y_train)

        # Evaluate

        eval_sampler = SequentialSampler(self.eval_dataset)
        eval_dataloader = DataLoader(
            self.eval_dataset,
            sampler=eval_sampler,
            batch_size=self.batch_size,
        )

        # Eval!
        X = np.array([])
        y = np.array([])
        for batch in eval_dataloader:
            inputs, labels = batch
            inputs = inputs.to(self.device)

            y = np.concatenate([y, labels.cpu().numpy().squeeze()], 0) if y.size else labels.numpy().squeeze()

            with torch.no_grad():
                outputs = self.model(inputs)[0].cpu().numpy().squeeze()
                outputs = np.max(outputs, axis=1)
                X = np.concatenate([X, outputs], 0) if X.size else outputs

        X = scaler.transform(X)
        y_pred = clf.predict(X)

        eval_acc = accuracy_score(y, y_pred)

        self.test_text = None
        self.real_text = None
        torch.cuda.empty_cache()
        return eval_acc

    def reset(self, test_text=None, real_text=None):
        if test_text is not None:
            self.test_text = test_text

        if real_text is not None:
            self.real_text = real_text

        self.eval_dataset = TextDataset(test_text=self.test_text, real_text=self.real_text, split='eval',
                                        block_size=self.block_size, train_size=self.train_size,
                                        eval_size=self.eval_size)
        self.train_dataset = TextDataset(test_text=self.test_text, real_text=self.real_text, split='train',
                                         block_size=self.block_size, train_size=self.train_size,
                                         eval_size=self.eval_size)
