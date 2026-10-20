import torch
from pathlib import Path

# ==================== 路径设置 ====================
# 项目根目录
BASE_DIR = Path(__file__).resolve().parent
# SQLite 地区数据库及原始地区数据
SQLITE_DB_PATH = BASE_DIR / "data/region.db"
REGION_SQL_PATH = BASE_DIR / "data/region.sql"
# 原始数据路径
RAW_DATA_DIR = BASE_DIR / "data/raw"
# 已处理数据存放路径
PROCESSED_DATA_DIR = BASE_DIR / "data/processed/v1_0_1"
# 本地预训练模型路径
PRETRAINED_DIR = BASE_DIR / "pretrained"
# 模型参数保存路径
FINETUNED_DIR = BASE_DIR / "finetuned"
# TensorBoard 日志保存路径
LOGS_DIR = BASE_DIR / "logs"

# 训练、测试和推理共用 MPS。
DEVICE = torch.device("cuda")

# 与 data.jsonl 和 V1.0.1 文档保持同一标签顺序。
LABELS = [
    "O",
    "B-prov",
    "I-prov",
    "E-prov",
    "B-city",
    "I-city",
    "E-city",
    "B-district",
    "I-district",
    "E-district",
    "S-district",
    "B-town",
    "I-town",
    "E-town",
    "S-town",
    "B-detail",
    "I-detail",
    "E-detail",
    "S-detail",
    "B-name",
    "I-name",
    "E-name",
    "B-phone",
    "I-phone",
    "E-phone",
]

MAX_SEQ_LEN = 128
MODEL_PARAMS_PATH = FINETUNED_DIR / "address_tagging.pt"
