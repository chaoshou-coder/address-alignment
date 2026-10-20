import config
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from address_alignment import address_alignment
from models_def import AddressTagging, load_params
from init_db import initialize_database

# 首次启动时从本地地区 SQL 初始化 SQLite，已有数据库直接复用。
initialize_database()

# 加载模型
model = AddressTagging(config.PRETRAINED_DIR / "bert-base-chinese", config.LABELS)
if not load_params(model, config.MODEL_PARAMS_PATH):
    raise FileNotFoundError("缺少训练权重，请先运行 main.py")

# 创建 FastAPI 实例
app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "templates")), name="static")

class AddressAlignmentRequest(BaseModel):
    message: str = Field(..., example="地址文本信息")

class AddressAlignmentResponse(BaseModel):
    province: str | None = Field(..., example="省份")
    city: str | None = Field(..., example="城市")
    district: str | None = Field(..., example="区县")
    town: str | None = Field(..., example="乡/镇/街道")
    detail: str | None = Field(..., example="详细地址")

@app.get("/")
async def homepage():
    return FileResponse(config.BASE_DIR / "templates/index.html")

@app.post("/address_alignment")
async def handle_message(request: AddressAlignmentRequest) -> AddressAlignmentResponse:
    user_message = request.message
    address = address_alignment(user_message, model)
    res = AddressAlignmentResponse(
        province=address["省份"],
        city=address["城市"],
        district=address["区县"],
        town=address["街道"],
        detail=address["详细地址"],
    )
    return res

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8089)
