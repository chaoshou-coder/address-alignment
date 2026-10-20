import torch
import config
from transformers import AutoTokenizer, BertModel

def load_params(model, model_params_path):
    """加载权重；首次训练允许没有权重，结构不匹配则明确报错。"""
    from pathlib import Path

    if model_params_path is None or not Path(model_params_path).is_file():
        return False
    try:
        model.load_state_dict(torch.load(model_params_path, map_location="cpu", weights_only=True))
    except RuntimeError as exc:
        raise ValueError("模型权重与当前标签或结构不匹配，请使用 V1.0.1 数据重新训练") from exc
    return True

class AddressTagging(torch.nn.Module):
    def __init__(self, model_name, label_list, max_seq_len=config.MAX_SEQ_LEN):
        super().__init__()
        self.label_list = label_list
        self.max_seq_len = max_seq_len
        self.num_labels = len(label_list)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        logits = self.classifier(self.dropout(outputs.last_hidden_state))

        loss = 0.0
        if labels is not None:
            loss += self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def predict(self, text: str | list[str], device=None, batch_size=32):
        device = config.DEVICE if device is None else device
        self.eval()
        self.to(device)

        res: list[list[str]] = []
        input_texts = text if isinstance(text, list) else [text]
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i : i + batch_size]
            # 将每个字符串拆成字列表：确保 tokenizer 能按字对齐 word_ids
            batch_words = [list(t) for t in batch_texts]
            inputs = self.tokenizer(
                batch_words,
                is_split_into_words=True,
                max_length=self.max_seq_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs = self(inputs["input_ids"], inputs["attention_mask"])
            preds = torch.argmax(outputs["logits"], dim=-1).detach().cpu()
            # 处理子词
            for batch_idx, pred in enumerate(preds):
                word_ids = inputs.word_ids(batch_index=batch_idx)
                current_word = None
                tokens_pred = [""] * len(batch_texts[batch_idx])
                for idx, word_id in enumerate(word_ids):
                    if word_id is not None and word_id != current_word:
                        current_word = word_id
                        label = self.label_list[pred[idx].item()]
                        tokens_pred[word_id] = label[2:] if label != "O" else ""
                res.append(tokens_pred)
        return res if isinstance(text, list) else res[0]
