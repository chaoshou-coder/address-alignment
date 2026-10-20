# 地址对齐

57 类标签，使用 PyTorch、FastAPI 和 SQLite。

## 环境与设备

在项目根目录安装依赖：

```powershell
uv sync
```

在 `config.py` 中选择设备，训练和推理共用：

```python
DEVICE = torch.device("cuda")  # 可改为 "cpu" 或 "mps"
```

## 数据与训练

```powershell
uv run python main.py
```

训练输入为 `data/raw/data.txt`，每行“文字 标签”，空行分隔样本。训练时自动预处理，默认 10 轮、batch 128。

训练后，将选定的 `finetuned/address_tagging-时间戳.pt` 复制为 `finetuned/address_tagging.pt`，供 Web 加载。

## Web 服务

```powershell
uv run python app.py
```

首页：<http://127.0.0.1:8089/>；接口调试：<http://127.0.0.1:8089/docs>。

SQLite 首次启动自动初始化，无需 MySQL。详细说明见 [docs/README.md](docs/README.md)。

## 待决事项

现有 TXT 只覆盖部分细分类，读取时有通用字段映射；细分类标注方案待完善。
