# 地址对齐

57 类标签，使用 PyTorch、FastAPI 和 SQLite。

## 环境与设备

在项目根目录安装依赖：

```bash
uv sync
```

本项目使用自己的 `.venv`。在 PyCharm 中将解释器设置为
`address-alignment/.venv/bin/python`，避免使用 ML08 的共享解释器。
不要将 `UV_PROJECT_ENVIRONMENT` 指向 ML08，也不要对 ML08 使用本项目的 `uv sync --active`。
共享环境里残留的 torchvision/torchaudio 可能与项目固定的 torch 版本不兼容；
本项目只处理文本，不需要这两个包。

同一份配置自动选择设备，训练和推理共用：CUDA → MPS → CPU。
无需在两台机器之间反复修改 `config.py`。启动训练时会打印实际设备。

`uv sync` 在 macOS 安装普通 PyTorch（支持 MPS），在 Linux / Windows 安装
CUDA 13.0 构建；`requirements.txt` 也按平台选择对应构建：

```bash
python -m pip install -r requirements.txt
```

安装源的平台配置依据：[uv 官方 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/)。
Linux / Windows 的 CUDA 加速需要兼容的 NVIDIA GPU 和驱动。可在两台机器上检查：

```bash
uv run python -c "import config; print(config.DEVICE)"
```

迁移权重时需要保持模型结构和标签顺序一致；权重先映射到 CPU，再放到运行设备。
MPS 显存不足时，可调小 `main.py` 中的 `batch_size`。

## 数据与训练

```bash
uv run python main.py
```

训练输入为 `data/raw/data.txt`，每行“文字 标签”，空行分隔样本。训练时自动预处理，默认 10 轮、batch 128。

训练后，将选定的 `finetuned/address_tagging-时间戳.pt` 复制为 `finetuned/address_tagging.pt`，供 Web 加载。

## Web 服务

```bash
uv run python app.py
```

首页：<http://127.0.0.1:8089/>；接口调试：<http://127.0.0.1:8089/docs>。

SQLite 首次启动自动初始化，无需 MySQL。详细说明见 [docs/README.md](docs/README.md)。

## 待决事项

现有 TXT 只覆盖部分细分类，读取时有通用字段映射；细分类标注方案待完善。
