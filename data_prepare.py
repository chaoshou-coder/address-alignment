"""预处理 V1.0.1 JSONL 数据；运行后可直接用于 main.py 训练。"""
import config
from transformers import AutoTokenizer
from preprocess import AddressTaggingProcessor

# 供训练配置和演示共用，避免维护两份标签。
LABELS = config.LABELS

# 样例数据
texts = [
    "中国浙江省杭州市余杭区葛墩路27号楼傅婷15830444519",
    "北京市市辖区通州区永乐店镇27号楼汪明13334219987",
    "高霞 13139243427北京市市辖区东风街道27号楼",
    "新疆维吾尔自治区划阿拉尔市金杨镇27号楼 刘燕 14727827196",
    "甘肃省南市文县碧口镇27号楼陈桂兰 13939269190",
    "陕西省渭南市华阴市罗镇27号楼 赵鑫15687584092",
    "邱金凤18582166250西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
    "广州市花都区花东镇27号楼张荣   18736672007",
]


def prepare_data():
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRETRAINED_DIR / "bert-base-chinese", local_files_only=True
    )
    processor = AddressTaggingProcessor(
        data_path=config.RAW_DATA_DIR / "data.jsonl",
        save_dir=config.PROCESSED_DATA_DIR,
        tokenizer=tokenizer,
        batch_size=32,
        label_list=LABELS,
        max_seq_len=config.MAX_SEQ_LEN,
    )
    processor.process()
    print(f"预处理完成：{config.PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    prepare_data()
