# tools/retrievalEnhanceTool.py

import re
from typing import Any, Dict, List

import jieba
from rank_bm25 import BM25Okapi



# 统一定义检索结果的数据结构
# 每个检索片段都用 dict 表示
RetrievedDoc = Dict[str, Any]


def tokenize_text(text: str) -> List[str]:
    """
    对中英文混合文本进行分词。

    Hybrid Search 中的 BM25 属于关键词检索，
    中文文本不能像英文一样直接按空格切分，
    因此这里使用 jieba 对中文进行分词，
    同时使用正则提取英文和数字。

    :param text: 输入文本
    :return: 分词后的 token 列表
    """

    if not text:
        return []

    text = str(text).lower()

    # 中文分词
    jieba_tokens = jieba.lcut(text)

    # 提取英文、数字、下划线等内容
    english_tokens = re.findall(r"[a-zA-Z0-9_]+", text)

    tokens = []

    for token in jieba_tokens + english_tokens:
        token = token.strip()

        if not token:
            continue

        # 过滤纯标点符号，只保留包含中文、英文或数字的 token，即至少包含上述要求的其中一个
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", token):
            continue

        tokens.append(token)

    return tokens


def min_max_normalize(values: List[float], reverse: bool = False) -> List[float]:
    """
    对分数进行 min-max 归一化。

    作用：
    将不同来源的分数（chromadb向量距离分数，bm25关键词相关分数）统一到 0~1 范围，便于后续融合。

    ChromaDB 返回的 distance 通常是距离，距离越小表示越相关。
    因此对 distance 做归一化时，需要设置 reverse=True，
    把“小距离”转换成“大分数”。

    :param values: 原始分数列表
    :param reverse: 是否反转分数
    :return: 归一化后的分数列表
    """

    if not values:
        return []

    cleaned_values = []

    # 把none值转换为0.0，并将所有数转为浮点数
    for value in values:
        if value is None:
            cleaned_values.append(0.0)
        else:
            cleaned_values.append(float(value))

    # 如果是 distance，则越小越好，所以取负数反转
    if reverse:
        cleaned_values = [-value for value in cleaned_values]

    min_value = min(cleaned_values)
    max_value = max(cleaned_values)

    # 如果所有分数都一样，则统一给 1.0，避免除零
    if abs(max_value - min_value) < 1e-12:
        return [1.0 for _ in cleaned_values]

    # 归一化所有数据
    return [
        (value - min_value) / (max_value - min_value)
        for value in cleaned_values
    ]


def get_first_query_result(value):
    """
    取出 ChromaDB 返回结果中第一个 query 对应的结果。

    ChromaDB query 的返回通常是二维列表，例如：

    documents = [
        ["文本1", "文本2", "文本3"]
    ]

    因为当前系统一次只查询一个用户问题，
    所以这里只取第一个 query 的结果，即 value[0]。

    :param value: ChromaDB 返回的某个字段
    :return: 第一个 query 对应的结果列表
    """

    if not value:
        return []

    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]

    if isinstance(value, list):
        return value

    return []


def parse_chroma_query_results(search_results: Dict[str, Any]) -> List[RetrievedDoc]:
    """
    将 ChromaDB 向量检索的原始结果解析成统一格式，即将多个结果单独列出。

    ChromaDB 原始格式一般是：

    {
        "ids": [["id1", "id2"]],
        "documents": [["文本1", "文本2"]],
        "distances": [[0.23, 0.31]],
        "metadatas": [[{}, {}]]
    }

    本函数会转换成：

    [
        {
            "id": "id1",
            "document": "文本1",
            "metadata": {},
            "distance": 0.23,
            "vector_score": 1.0
        }
    ]

    :param search_results: ChromaDB 原始检索结果
    :return: 统一格式的向量检索结果
    """

    if not isinstance(search_results, dict):
        return []

    # 获取向量检索买个字典的值
    documents = get_first_query_result(search_results.get("documents"))
    ids = get_first_query_result(search_results.get("ids"))
    distances = get_first_query_result(search_results.get("distances"))
    metadatas = get_first_query_result(search_results.get("metadatas"))

    if not documents:
        return []

    # ChromaDB distance 越小越相关，所以这里 reverse=True
    if distances:
        vector_scores = min_max_normalize(distances, reverse=True)
    else:
        # 如果没有 distance，就根据原始排序构造一个简单分数
        total = len(documents)
        vector_scores = [
            1.0 - index / max(total, 1)
            for index in range(total)
        ]

    parsed_docs = []

    for index, document in enumerate(documents):
        # 跳过处理空白数据
        if not document or not str(document).strip():
            continue
        
        # 将每个检索结果组合成一个集合添加到列表中。
        parsed_docs.append(
            {
                "id": ids[index] if index < len(ids) else None,
                "document": str(document).strip(),
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "distance": distances[index] if index < len(distances) else None,
                "vector_score": vector_scores[index] if index < len(vector_scores) else 0.0,
            }
        )

    return parsed_docs

# bm25 检索出相关片段
def bm25_search_all(query: str, collection, top_k: int = 10) -> List[RetrievedDoc]:
    """
    在 ChromaDB collection 的所有文档片段上执行 BM25 关键词检索。

    :param query: 用户问题
    :param collection: ChromaDB collection 对象
    :param top_k: BM25 返回的候选数量
    :return: BM25 检索结果
    """

    if not query:
        return []

    # 对用户问题分词
    query_tokens = tokenize_text(query)

    if not query_tokens:
        return []

    # 从 ChromaDB 中取出所有文档片段
    all_data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = all_data.get("documents") or []
    ids = all_data.get("ids") or []
    metadatas = all_data.get("metadatas") or []

    all_docs = []

    for index, document in enumerate(documents):
        if not document or not str(document).strip():
            continue

        all_docs.append(
            {
                "id": ids[index] if index < len(ids) else None,
                "document": str(document).strip(),
                "metadata": metadatas[index] if index < len(metadatas) else {},
            }
        )

    if not all_docs:
        return []

    # 构建 BM25 语料库
    corpus_tokens = [
        tokenize_text(item["document"])
        for item in all_docs
    ]

    # 实例化 BM25Okapi
    bm25 = BM25Okapi(corpus_tokens)

    # 计算每个文档片段相对于 query 的 BM25 分数
    bm25_scores = bm25.get_scores(query_tokens).tolist()

    # 归一化 BM25 分数
    bm25_norm_scores = min_max_normalize(bm25_scores)

    bm25_docs = []

    for item, raw_score, norm_score in zip(all_docs, bm25_scores, bm25_norm_scores):
        new_item = dict(item) # 拷贝一份原数据，防止影响原列表
        new_item["bm25_score"] = float(raw_score)
        new_item["bm25_norm_score"] = float(norm_score)
        bm25_docs.append(new_item)

    # 按 BM25 归一化分数从高到低排序
    bm25_docs.sort(
        key=lambda x: x.get("bm25_norm_score", 0.0),
        reverse=True
    )

    return bm25_docs[:top_k]


def hybrid_fusion(
    vector_docs: List[RetrievedDoc],
    bm25_docs: List[RetrievedDoc],
    alpha: float = 0.7,
    top_k: int = 3
) -> List[RetrievedDoc]:
    """
    融合向量检索结果和 BM25 检索结果。

    融合公式：
    hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score

    其中：
    alpha 越大，越依赖向量语义检索；
    alpha 越小，越依赖 BM25 关键词检索。

    :param vector_docs: ChromaDB 向量检索结果
    :param bm25_docs: BM25 关键词检索结果
    :param alpha: 向量检索权重
    :param top_k: 最终返回的片段数量
    :return: 融合排序后的文档列表
    """

    merged = {}

    # def get_key(item: RetrievedDoc) -> str:
    #     """
    #     用于去重。

    #     优先使用 ChromaDB 的 id；
    #     如果没有 id，则使用 document 文本内容。
    #     """

    #     if item.get("id"):
    #         return str(item["id"])

    #     return item.get("document", "")

    # 加入向量检索结果
    for item in vector_docs:
        key = str(item["id"])

        if not key:
            continue

        merged[key] = dict(item)
        merged[key]["vector_score"] = float(item.get("vector_score", 0.0))
        merged[key].setdefault("bm25_norm_score", 0.0)

    # 加入 BM25 检索结果
    for item in bm25_docs:
        key = str(item["id"])

        if not key:
            continue
        
        # ChromaDB 没找到，但 BM25 找到了
        if key not in merged:
            merged[key] = dict(item)
            merged[key].setdefault("vector_score", 0.0)

        merged[key]["bm25_score"] = float(item.get("bm25_score", 0.0))
        merged[key]["bm25_norm_score"] = float(item.get("bm25_norm_score", 0.0))
    
    # merged中每个元素是一个字典类型检索结果，key是对应检索结果id的str类型

    fused_docs = []

    # merged.values直接取每个键值对的值。
    for item in merged.values():
        vector_score = float(item.get("vector_score", 0.0))
        bm25_score = float(item.get("bm25_norm_score", 0.0))

        # 计算每个片段的融合分数，并添加到列表中
        item["hybrid_score"] = alpha * vector_score + (1 - alpha) * bm25_score

        fused_docs.append(item)

    # 按融合分数从高到低排序
    fused_docs.sort(
        key=lambda x: x.get("hybrid_score", 0.0),
        reverse=True
    )

    return fused_docs[:top_k]


def format_docs_for_context(docs: List[RetrievedDoc]) -> str:
    """
    将融合排序后的检索结果格式化成字符串。

    该字符串会被写入 LangGraph 的 state["retrieved_context"]，
    再传给 report_node 生成健康报告。

    :param docs: 融合排序后的文档列表
    :return: 拼接后的检索上下文
    """

    if not docs:
        return "未从健康档案库中检索到相关内容。"

    full_text_parts = []

    for index, item in enumerate(docs, start=1):
        document = item.get("document", "")

        if not document:
            continue

        hybrid_score = item.get("hybrid_score")
        vector_score = item.get("vector_score")
        bm25_score = item.get("bm25_norm_score")

        title = (
            f"【检索片段{index} | "
            f"hybrid={float(hybrid_score):.4f}, "
            f"vector={float(vector_score):.4f}, "
            f"bm25={float(bm25_score):.4f}】"
        )

        full_text_parts.append(
            f"{title}\n{document.strip()}"
        )

    if not full_text_parts:
        return "检索结果为空。"

    return "\n\n".join(full_text_parts)