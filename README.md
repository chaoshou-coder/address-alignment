
## 设备

`config.py` 直接指定 MPS，训练、测试和推理共用：

```python
DEVICE = torch.device("mps")
```


```bash
python -m pip install -r requirements.txt
```

也可使用 `uv sync`，`pyproject.toml` 已为 torch 指定专用 cu130 源。
PyTorch CUDA 构建：[官方安装说明](https://pytorch.org/get-started/previous-versions/)。
目标设备必须具备兼容的 NVIDIA GPU 和驱动

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

应显示 cu130 构建、CUDA 13.0，且设备可用结果为 `True`。
部署时将 `config.py` 中的设备配置改为：

```python
DEVICE = torch.device("cuda")
```

SQLite 和 unittest 使用 Python 标准库，无需额外安装。

## 运行

激活运行环境后，在项目根目录依次运行：

```bash
python data_prepare.py
python main.py
```

第一步生成 `data/processed/v1_0_1/` 下的训练、验证、测试数据集。
若省略第一步，训练入口会自动预处理。JSONL 中多字词会展开为逐字标签。
`main.py` 默认训练 10 轮、批大小 32，测试并打印样例预测；可修改入口的
`model_go(..., epochs=10, batch_size=32)` 调整参数。

地址校验使用 Python 内置 SQLite，无需安装数据库服务或 pymysql。
首次启动服务会从 `data/region.sql` 自动生成 `data/region.db`，已有数据库直接复用。
也可以先运行 `python init_db.py` 单独初始化。
数据库路径可在 `config.py` 的 `SQLITE_DB_PATH` 中修改；训练不依赖地区数据库。

项目根目录下使用:

```bash
tensorboard --logdir=logs/
```

监控训练情况。

```bash
python app.py
```

启动后访问 <http://127.0.0.1:8089/> 或 <http://127.0.0.1:8089/docs>。
服务必须使用训练后的权重，缺少权重会明确报错。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试使用本地 tokenizer 和小型随机 BERT 验证训练链路，不代表真实模型效果。
