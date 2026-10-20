import json
import os
from pathlib import Path
import random
from datasets import Dataset
from torch.utils.data import Subset
from datasets import load_from_disk
from torch.utils.data import DataLoader

class Processor:
    def __init__(
        self,
        data_path,
        save_dir,
        tokenizer,
        batch_size,
        max_seq_len,
        train_ratio=0.8,
        test_ratio=0.1,
    ):
        self.data_path = Path(data_path)
        self.save_dir = Path(save_dir)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def process(self):
        """处理数据集并保存"""
        dataset = self._make_dataset()
        dataset = self._split_dataset(dataset)
        for type in ["train", "valid", "test"]:
            dataset[type].save_to_disk(self.save_dir / type)

    def get_dataloader(self, type, max_examples=None):
        """获取保存的数据集，并加载 Dataloader"""
        if not os.path.exists(self.save_dir / type):
            self.process()
        dataset = load_from_disk(self.save_dir / type)
        if max_examples:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            max_examples = min(max_examples, len(dataset))
            indices = indices[:max_examples]
            dataset = Subset(dataset, indices)
        return DataLoader(dataset, self.batch_size, shuffle=(type == "train"))

    def _make_dataset(self):
        """处理数据集"""
        raise NotImplementedError

    def _split_dataset(self, dataset: Dataset):
        """划分数据集"""
        train_size = int(dataset.num_rows * self.train_ratio)
        dataset = dataset.train_test_split(test_size=self.test_ratio, seed=42)
        split = dataset["train"].train_test_split(train_size=train_size, seed=42)
        dataset["train"] = split["train"]
        dataset["valid"] = split["test"]
        return dataset

class AddressTaggingProcessor(Processor):
    def __init__(
        self,
        data_path,
        save_dir,
        tokenizer,
        batch_size,
        label_list,
        max_seq_len=128,
        train_ratio=0.8,
        test_ratio=0.1,
    ):
        super().__init__(
            data_path,
            save_dir,
            tokenizer,
            batch_size,
            max_seq_len,
            train_ratio,
            test_ratio,
        )
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(self.label_list)}

    def _make_dataset(self):
        """处理数据集"""
        dataset = Dataset.from_generator(self._generate_examples)
        dataset = dataset.map(
            self._map_fn, batched=True, remove_columns=["text", "labels"]
        )
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return dataset

    def _generate_examples(self):
        """读取逐字标注 JSONL，并在错误中指出原始数据行号。"""
        with self.data_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                    text, labels = example["text"], example["labels"]
                    if not isinstance(text, list) or not text or not all(
                        isinstance(char, str) and bool(char) for char in text
                    ):
                        raise ValueError("text 必须是非空的字符串列表")
                    if not isinstance(labels, list) or len(text) != len(labels):
                        raise ValueError("text 与 labels 必须等长")
                    chars, char_labels = [], []
                    for word, label in zip(text, labels):
                        if label not in self.label2id:
                            raise ValueError(f"未知标签：{label}")
                        # 数据第 3202 行含多字词“中心”，展开后保持字符级标注。
                        for index, char in enumerate(word):
                            char_label = label
                            if len(word) > 1 and label != "O":
                                prefix, entity = label.split("-", 1)
                                if prefix == "B":
                                    prefix = "B" if index == 0 else "I"
                                elif prefix == "E":
                                    prefix = "E" if index == len(word) - 1 else "I"
                                elif prefix == "S":
                                    prefix = "B" if index == 0 else "E" if index == len(word) - 1 else "I"
                                char_label = f"{prefix}-{entity}"
                            chars.append(char)
                            char_labels.append(self.label2id[char_label])
                    text, label_ids = chars, char_labels
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{self.data_path}:{line_no}: 数据或标签无效：{exc}") from exc
                yield {"text": text, "labels": label_ids}

    def _map_fn(self, examples):
        inputs = self.tokenizer(
            examples["text"],
            is_split_into_words=True,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        all_labels = []
        for i, labels in enumerate(examples["labels"]):
            # 得到每个 token 对应序列中原始词的索引
            word_ids = inputs.word_ids(batch_index=i)
            aligned_labels = self._align_labels_with_tokens(labels, word_ids)
            all_labels.append(aligned_labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": all_labels,
        }

    def _align_labels_with_tokens(self, labels, word_ids):
        """
        标签与 token 对齐
        如果一个词被分为多个子词，则第一个 token 的标签为词的标签，其余为 -100
        """
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is not None and word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx])
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        return aligned_labels
