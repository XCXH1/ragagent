import json
import os


def load_key(provider_name: str) -> str:
    """
    根据传入的服务商名称，从同目录下的 Keys.json 文件中读取并返回对应的 API Key。

    :param provider_name: 需要读取的 key 名称 (例如: 'tongyi', 'zhipu')
    :return: 对应的 API Key 字符串
    """
    # 🌟 核心魔法：获取当前代码文件 (Loader_key.py) 所在的绝对文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 拼接出同目录下 Keys.json 的完整绝对路径
    file_path = os.path.join(current_dir, "Keys.json")

    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到配置文件: '{file_path}'，请检查该文件是否存在。")

    # 2. 读取并解析 JSON 文件
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            keys_data = json.load(file)
    except json.JSONDecodeError:
        raise ValueError(f"❌ 文件 '{file_path}' 不是有效的 JSON 格式，请检查语法。")

    # 3. 查找对应的 Key
    if provider_name not in keys_data:
        raise KeyError(f"❌ 在 Keys.json 中未找到名为 '{provider_name}' 的配置项！")

    return keys_data[provider_name]


# ==========================================
# 本地测试代码 (右键直接运行此文件时测试用)
# ==========================================
# if __name__ == "__main__":
#     try:
#         print("测试读取 zhipu:", load_key("zhipu"))
#         print("测试读取 tongyi:", load_key("tongyi"))
#     except Exception as e:
#         print(e)