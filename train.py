import tqdm
import torch
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support

class Trainer:
    """训练验证与测试"""

    def __init__(self, model, device, epochs, learning_rate, checkpoint_steps=200):
        """
        参数:
        - model: 模型
        - device: 设备
        - epochs: 训练轮数
        - learning_rate: 学习率
        - checkpoint_steps: 多少步之后保存检查点
        """
        self.model = model
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.checkpoint_steps = checkpoint_steps

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def __call__(
        self,
        dataloader: dict,
        model_params_path=None,
        writer=None,
        is_test=False,
    ):
        """
        参数:
        - dataloader: 数据加载器
        - model_params_path: 模型参数保存路径
        - writer: 记录器
        - is_test: 是否执行测试
        """
        self.dataloader = dataloader
        self.model_params_path = model_params_path
        self.writer = writer
        self.is_test = is_test

        self.model.to(self.device)
        self.global_step = 0

        # 测试
        if is_test:
            for k, v in self.run_epoch("test").items():
                print(f"Test {k}:", v)
            return

        # 训练并验证
        assert self.model_params_path is not None, "缺少模型参数保存路径"
        best_valid_metric = 0
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")

            train_metrics = self.run_epoch("train", epoch)
            for k, v in train_metrics.items():
                print(f"Train {k}:", v)

            valid_metrics = self.run_epoch("valid", epoch)
            for k, v in valid_metrics.items():
                print(f"Valid {k}:", v)

            # 保存最佳模型
            if valid_metrics["f1"] >= best_valid_metric:
                best_valid_metric = valid_metrics["f1"]
                torch.save(self.model.state_dict(), self.model_params_path)

    def run_epoch(self, phase, epoch=0):
        self.model.train() if phase == "train" else self.model.eval()
        # 初始化总损失值和总样本数
        total_loss = 0.0
        total_examples = 0
        # 初始化记录
        records = {}

        with torch.set_grad_enabled(phase == "train"):
            for inputs in tqdm.tqdm(self.dataloader[phase], desc=phase):
                # 数据转移到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 前向传播
                outputs, loss = self.forward(inputs)

                # 反向传播和优化（仅训练阶段）
                if phase == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 向 TensorBoard 写入损失
                    if self.writer:
                        self.writer.add_scalar(
                            f"Loss/{phase}", loss.item(), self.global_step
                        )

                    self.global_step += 1

                    # 保存模型参数
                    if (
                        self.checkpoint_steps
                        and self.global_step % self.checkpoint_steps == 0
                    ):
                        checkpoint_path = str(self.model_params_path) + ".checkpoint"
                        torch.save(self.model.state_dict(), checkpoint_path)

                # 记录损失
                current_batch_size = inputs["input_ids"].size(0)
                total_loss += loss.item() * current_batch_size
                total_examples += current_batch_size

                # 更新记录，用于评估
                if phase != "train":
                    self.update_records(inputs, outputs, records)

        # 计算平均损失
        metrics = {"loss": total_loss / total_examples}

        # 计算评估指标
        if phase != "train":
            self.compute_metrics(metrics, records)
            if self.writer:
                for metric_name, value in metrics.items():
                    self.writer.add_scalar(f"{phase}/{metric_name}", value, epoch)
        return metrics

    def forward(self, inputs):
        """前向传播"""
        raise NotImplementedError

    def update_records(self, inputs, outputs, records):
        """更新记录"""
        raise NotImplementedError

    def compute_metrics(self, metrics, records):
        """计算评估指标"""
        raise NotImplementedError

class AddressTaggingTrainer(Trainer):
    def forward(self, inputs):
        """前向传播"""
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        return outputs, outputs["loss"]

    def update_records(self, inputs, outputs, records):
        """更新记录"""
        preds = outputs["logits"].argmax(dim=-1)
        labels = inputs["labels"]
        mask = (inputs["attention_mask"] == 1) & (labels != -100)
        preds = preds[mask].view(-1).detach().cpu()
        labels = labels[mask].view(-1).detach().cpu()
        records.setdefault("preds", []).append(preds)
        records.setdefault("labels", []).append(labels)

    def compute_metrics(self, metrics, records):
        """计算评估指标"""
        all_preds = torch.cat(records["preds"])
        all_labels = torch.cat(records["labels"])
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        metrics.update({"precision": precision, "recall": recall, "f1": f1})
