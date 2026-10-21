import torch
from pathlib import Path


# ==================== 路径设置 ====================
# 项目根目录
BASE_DIR = Path(__file__).parent
SQLITE_DB_PATH = BASE_DIR / "data/region.db"
REGION_SQL_PATH = BASE_DIR / "data/region.sql"
# 原始数据路径
RAW_DATA_DIR = BASE_DIR / "data/raw"
# 已处理数据存放路径
PROCESSED_DATA_DIR = BASE_DIR / "data/processed"
# 本地预训练模型路径
PRETRAINED_DIR = BASE_DIR / "pretrained"
# 模型参数保存路径
FINETUNED_DIR = BASE_DIR / "finetuned"
# TensorBoard 日志保存路径
LOGS_DIR = BASE_DIR / "logs"

# 训练和推理共用设备：优先 CUDA，其次 MPS，均不可用时使用 CPU。
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

LABELS = [
    "O",
    "B-assist",
    "I-assist",
    "S-assist",
    "E-assist",
    "B-cellno",
    "I-cellno",
    "E-cellno",
    "B-city",
    "I-city",
    "E-city",
    "B-community",
    "I-community",
    "S-community",
    "E-community",
    "B-devzone",
    "I-devzone",
    "E-devzone",
    "B-district",
    "I-district",
    "S-district",
    "E-district",
    "B-floorno",
    "I-floorno",
    "E-floorno",
    "B-houseno",
    "I-houseno",
    "E-houseno",
    "B-poi",
    "I-poi",
    "S-poi",
    "E-poi",
    "B-prov",
    "I-prov",
    "E-prov",
    "B-road",
    "I-road",
    "E-road",
    "B-roadno",
    "I-roadno",
    "E-roadno",
    "B-subpoi",
    "I-subpoi",
    "E-subpoi",
    "B-town",
    "I-town",
    "E-town",
    "B-intersection",
    "I-intersection",
    "S-intersection",
    "E-intersection",
    "B-distance",
    "I-distance",
    "E-distance",
    "B-village_group",
    "I-village_group",
    "E-village_group",
]
