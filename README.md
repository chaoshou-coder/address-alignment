# 地址对齐

当前代码使用 `data/raw/data.jsonl` 和 V1.0.1 的 25 个标签。继续使用本地
`pretrained/bert-base-chinese`，无需重新下载预训练模型。

## 设备

`config.py` 直接指定 MPS，训练、测试和推理共用：

```python
DEVICE = torch.device("mps")
```

## CUDA 13.0 部署依赖

`requirements.txt` 按本项目直接依赖维护，与 `pyproject.toml` 保持一致，
不复制 ML08 环境。目标为 Linux / Windows NVIDIA GPU，沿用项目 Python 3.13 要求。
PyTorch 固定为官方 `2.13.0+cu130` 构建，其他直接依赖声明版本范围；此文件不是完整依赖锁文件。

在目标服务器的独立虚拟环境中安装：

```bash
python -m pip install -r requirements.txt
```

也可使用 `uv sync`，`pyproject.toml` 已为 torch 指定专用 cu130 源。
PyTorch CUDA 构建依据：[官方安装说明](https://pytorch.org/get-started/previous-versions/)。
目标设备必须具备兼容的 NVIDIA GPU 和驱动，安装后实际检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

应显示 cu130 构建、CUDA 13.0，且设备可用结果为 `True`。
部署时将 `config.py` 的设备改为：

```python
DEVICE = torch.device("cuda")
```

本地开发仍可借用 ML08 环境并保留 MPS；无需在 Mac 上安装 CUDA 依赖。
SQLite 和 unittest 使用 Python 标准库，无需额外安装。
当前未在目标 CUDA 服务器执行安装或 GPU 验证。

## 运行

激活运行环境后，在项目根目录依次运行：

```bash
python data_prepare.py
python main.py
```

第一步生成 `data/processed/v1_0_1/` 下的训练、验证、测试数据集，隔离旧版缓存。
若省略第一步，训练入口会自动预处理。JSONL 中多字词会展开为逐字标签。
`main.py` 默认训练 10 轮、批大小 32，测试并打印样例预测；可修改入口的
`model_go(..., epochs=10, batch_size=32)` 调整参数。

验证集最优权重保存到 `finetuned/address_tagging.pt`，已有兼容权重会用于继续训练。
旧版标签训练的权重不可复用，需要重新训练。

地址校验使用 Python 内置 SQLite，无需安装数据库服务或 pymysql。
首次启动服务会从 `data/region.sql` 自动生成 `data/region.db`，已有数据库直接复用。
也可以先运行 `python init_db.py` 单独初始化。
数据库路径可在 `config.py` 的 `SQLITE_DB_PATH` 中修改；训练不依赖地区数据库。

```bash
python app.py
```

启动后访问 <http://127.0.0.1:8089/> 或 <http://127.0.0.1:8089/docs>。
服务必须使用训练后的权重，缺少权重会明确报错。
姓名和电话参与模型标注，但不会拼入接口返回的地址字段。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试使用本地 tokenizer 和小型随机 BERT 验证训练链路，不代表真实模型效果。
