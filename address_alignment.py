import config
import sqlite3
from contextlib import closing
from init_db import initialize_database
from models_def import AddressTagging, load_params

def address_alignment(text, model):
    tagging = model.predict(text)

    # 填充地址信息
    # 标签到地区类别 id 映射表
    label_map = {
        "": 0,
        "prov": 2,
        "city": 3,
        "district": 4,
        "road": 5,
        "intersection": 5,
        "town": 5,
        "roadno": 6,
        "cellno": 6,
        "community": 6,
        "houseno": 6,
        "poi": 6,
        "subpoi": 6,
        "assist": 6,
        "distance": 6,
        "village_group": 6,
        "floorno": 6,
        "devzone": 6,
    }
    address = {
        2: None,
        3: None,
        4: None,
        5: None,
        6: None,
    }
    start_pos = 0
    tag_len = len(tagging)
    for end_pos in range(tag_len):
        # 如果到结尾、或 end_pos 的下一个位置不是同一类
        if (end_pos == tag_len - 1) or (
            label_map[tagging[end_pos + 1]] != label_map[tagging[start_pos]]
        ):
            if label_map[tagging[start_pos]] != 0:
                # 添加片段
                address[label_map[tagging[start_pos]]] = text[start_pos : end_pos + 1]
            start_pos = end_pos + 1
    # print("标注后填充结果:", address)

    # 校验地址信息
    for region_type_id in [2, 3, 4, 5]:
        if not address[region_type_id]:
            continue
        check_address(region_type_id, address)

    # print("校验后填充结果:", address)
    return {
        "省份": address[2],
        "城市": address[3],
        "区县": address[4],
        "街道": address[5],
        "详细地址": address[6],
    }

def check_address(region_type_id, address, parent_id=None):
    """
    向上校验地址信息
    如果没有查询到结果，当前地址信息改为None
    如果是顶层，填充本级
    如果上级不为None，且校验正确，填充当前地址信息
    如果上级不为None，且校验不正确，将当前地址信息改为None
    如果上级为None，使用父级地址填充上级，并继续校验上级。直到顶层，或者
        如果校验到不为None且正确的，结束校验
        如果校验到不为None且不正确的，撤销当前到该上级的所有的填充
    """
    res = query_parent(region_type_id, address[region_type_id], parent_id)
    # print(address)

    # 无结果
    if not res:
        address[region_type_id] = None
        return True

    # 顶层
    if region_type_id == 2:
        address[region_type_id] = res["name"]
        return True

    # 上级不为None，且校验正确
    if (
        address[region_type_id - 1]
        and address[region_type_id - 1] == res["parent_name"]
    ):
        address[region_type_id] = res["name"]
        return True

    # 上级不为None，且校验不正确
    if (
        address[region_type_id - 1]
        and address[region_type_id - 1] != res["parent_name"]
    ):
        address[region_type_id] = None
        return False

    # 上级为None，使用父级地址填充上级，并继续校验上级
    if not address[region_type_id - 1]:
        address[region_type_id] = res["name"]
        address[region_type_id - 1] = res["parent_name"]
        done = check_address(region_type_id - 1, address, res["parent_id"])
        if done:
            return True
        address[region_type_id] = None
        address[region_type_id - 1] = None
        return False

def query_parent(region_type_id, region_name, region_id=None):
    """Query a region and its parent using the local SQLite database."""
    database_uri = config.SQLITE_DB_PATH.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        sql = (
            "select region_parent.id as parent_id, "
            "region_parent.name as parent_name, region.name as name "
            "from region "
            "join region as region_parent on region_parent.id = region.parent_id "
            "where region.region_type = ? and region.name like ?"
        )
        params = [region_type_id, f"%{region_name}%"]
        if region_id:
            sql += " and region.id = ?"
            params.append(region_id)
        results = connection.execute(sql, params).fetchall()
        if results:
            for row in results:
                if region_name == row["name"]:
                    return dict(row)
            return dict(results[0])
        return None

if __name__ == "__main__":
    initialize_database()
    model = AddressTagging(config.PRETRAINED_DIR / "bert-base-chinese", config.LABELS)
    load_params(model, config.FINETUNED_DIR / "address_tagging.pt")
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
    for i in text:
        print(address_alignment(i, model))
