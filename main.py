import config
from train import AddressTaggingTrainer
from preprocess import AddressTaggingProcessor
from models_def import AddressTagging, load_params
from torch.utils.tensorboard.writer import SummaryWriter

device = config.DEVICE
learning_rate = 1e-5

def model_go(train=None, test=None, inference=None, model_params_path=None, epochs=10, batch_size=32):
    print(f"运行设备：{device}")
    model = AddressTagging(config.PRETRAINED_DIR / "bert-base-chinese", config.LABELS)
    processor = AddressTaggingProcessor(
        data_path=config.RAW_DATA_DIR / "data.jsonl",
        save_dir=config.PROCESSED_DATA_DIR,
        tokenizer=model.tokenizer,
        batch_size=batch_size,
        label_list=config.LABELS,
        max_seq_len=config.MAX_SEQ_LEN,
    )
    trainer = AddressTaggingTrainer(model, device, epochs, learning_rate)

    writer = None
    model_params_path = model_params_path or config.MODEL_PARAMS_PATH
    loaded = load_params(model, model_params_path)
    if not train and not loaded:
        raise FileNotFoundError(f"缺少训练权重：{model_params_path}，请先训练")

    if train:
        config.FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(config.LOGS_DIR))
        dataloader = {
            "train": processor.get_dataloader("train"),
            "valid": processor.get_dataloader("valid"),
        }
        trainer(dataloader, model_params_path, writer)
        load_params(model, model_params_path)  # 使用验证集最优权重测试和推理

    if test:
        test_dataloader = processor.get_dataloader("test")
        trainer({"test": test_dataloader}, writer=writer, is_test=True)

    if writer:
        writer.close()

    if inference:
        res = model.predict(text, device)
        if isinstance(res, str):
            print(res)
        elif isinstance(res, list):
            for a_text, a_res in zip(text, res):
                for t, r in zip(a_text, a_res):
                    print(f"{t}-{r}", end="\t")
                print("\n")

text = [
    "中国浙江省杭州市余杭区葛墩路27号楼",
    "北京市通州区永乐店镇27号楼",
    "北京市市辖区高地街道27号楼",
    "新疆维吾尔自治区划阿拉尔市金杨镇27号楼",
    "甘肃省南市文县碧口镇27号楼",
    "陕西省渭南市华阴市罗镇27号楼",
    "西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
    "广州市花都区花东镇27号楼",
]

if __name__ == "__main__":
    model_go(train=True, test=True, inference=True)
