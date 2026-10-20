import config
import sqlite3
from contextlib import closing
from models_def import AddressTagging, load_params

def address_alignment(text, model):
    tagging = model.predict(text)

    # 姓名、电话和 O 标签不进入地址字段。
    label_map = {"prov": 2, "city": 3, "district": 4, "town": 5, "detail": 6}
    address = dict.fromkeys(range(2, 7))
    for char, label in zip(text, tagging):
        region_type_id = label_map.get(label)
        if region_type_id is not None:
            address[region_type_id] = (address[region_type_id] or "") + char

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
    """
    查询自己和父级地址信息
    如果是首次查询，只有地址名称可作为过滤信息
    如果是后续查询，可以使用先前查询的 parent_id 作为当前 id 来进行过滤，防止重名地址的问题
    """
    # 只读连接，避免路径错误时悄悄创建空数据库。
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
        if region_id is not None:
            sql += " and region.id = ?"
            params.append(region_id)
        # 多个候选时优先精确匹配，其余按原地区编号稳定排序。
        sql += " order by (region.name = ?) desc, region.id limit 1"
        params.append(region_name)
        row = connection.execute(sql, params).fetchone()
        return dict(row) if row is not None else None


if __name__ == "__main__":
    from init_db import initialize_database
    initialize_database()
    model = AddressTagging(config.PRETRAINED_DIR / "bert-base-chinese", config.LABELS)
    if not load_params(model, config.MODEL_PARAMS_PATH):
        raise FileNotFoundError("缺少训练权重，请先运行 main.py")
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
