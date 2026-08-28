# tools/vectorSearchTool.py

# 引入相关库
import os
import logging
from pathlib import Path
from typing import List

from openai import OpenAI
import chromadb
from config.Loader_key import load_key
import pprint

from tools.retrievalEnhanceTool import (
    parse_chroma_query_results,
    bm25_search_all,
    hybrid_fusion,
    format_docs_for_context,
)


# 设置日志模版
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================
# Hybrid Search 配置
# =========================================================
# 向量检索初筛数量
VECTOR_CANDIDATE_TOP_K = 10

# BM25 关键词检索候选数量
BM25_CANDIDATE_TOP_K = 10

# 最终传给大模型的片段数量
FINAL_TOP_K = 3

# Hybrid Search 中向量检索的权重
# 0.7 表示 70% 依赖向量语义检索，30% 依赖 BM25 关键词检索
HYBRID_ALPHA = 0.7


# =========================
# 模型设置相关
# =========================

# openai: 调用 OpenAI-compatible 的 embedding 接口
# oneapi: 调用 oneapi 方案支持的 embedding 接口
API_TYPE = "openai"

# openai模型相关配置 根据自己的实际情况进行调整
OPENAI_API_BASE = "https://api.z.ai/api/paas/v4/"
OPENAI_EMBEDDING_API_KEY = load_key("zhipu")
OPENAI_EMBEDDING_MODEL = "embedding-2"

# oneapi相关配置 根据自己的实际情况进行调整
ONEAPI_API_BASE = "https://api.z.ai/api/paas/v4/"
ONEAPI_EMBEDDING_API_KEY = load_key("zhipu")
ONEAPI_EMBEDDING_MODEL = "embedding-2"

# =========================
# ChromaDB 配置
# =========================

# 如果你仍然使用原来的绝对路径，可以保留下面这个：
CHROMADB_DIRECTORY = r"G:\桌面\Agent\Projects\AgentVQA\src\vectorSaveTest\chromaDB"

# 待查询的 ChromaDB 集合名称
CHROMADB_COLLECTION_NAME = "demo001"

# =========================
# Embedding 向量生成函数
# =========================

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    计算文本向量。

    :param texts: 文本列表
    :return: 向量列表
    """

    global API_TYPE
    global ONEAPI_API_BASE, ONEAPI_EMBEDDING_API_KEY, ONEAPI_EMBEDDING_MODEL
    global OPENAI_API_BASE, OPENAI_EMBEDDING_API_KEY, OPENAI_EMBEDDING_MODEL

    # 过滤空文本，避免 embedding 接口报错
    texts = [text.strip() for text in texts if text and text.strip()]

    if not texts:
        raise ValueError("待向量化文本为空，无法生成 embedding。")

    if API_TYPE == "oneapi":
        api_base = ONEAPI_API_BASE
        api_key = ONEAPI_EMBEDDING_API_KEY
        model_name = ONEAPI_EMBEDDING_MODEL

    elif API_TYPE == "openai":
        api_base = OPENAI_API_BASE
        api_key = OPENAI_EMBEDDING_API_KEY
        model_name = OPENAI_EMBEDDING_MODEL

    else:
        raise ValueError(f"不支持的 API_TYPE: {API_TYPE}")

    if not api_key:
        raise ValueError(
            f"{API_TYPE} 的 embedding API Key 未设置，请先配置环境变量。"
        )

    try:
        client = OpenAI(
            base_url=api_base,
            api_key=api_key
        )

        response = client.embeddings.create(
            input=texts,
            model=model_name
        )

        embeddings = [item.embedding for item in response.data]

        if not embeddings:
            raise RuntimeError("embedding 接口返回空向量。")

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"embedding 数量与文本数量不一致，文本数量={len(texts)}，向量数量={len(embeddings)}。"
            )

        return embeddings

    except Exception as e:
        logger.exception(f"生成向量时出错: {e}")
        raise


# =========================
# 分批生成向量,一次性过多传入负载大
# =========================

def generate_vectors(data: List[str], max_batch_size: int = 25) -> List[List[float]]:
    """
    对文本按批次进行向量计算。

    :param data: 文本列表
    :param max_batch_size: 每批处理的最大文本数量
    :return: 向量列表
    """

    data = [text.strip() for text in data if text and text.strip()]

    if not data:
        raise ValueError("待生成向量的数据为空。")

    results = []

    # 索引从0，到文本数，步长为 max_batch_size。
    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]

        # 调用向量生成 get_embeddings 方法
        response = get_embeddings(batch)

        if not response:
            raise RuntimeError("当前批次 embedding 结果为空。")

        # extend用于在列表一次性追加另一个序列中的多个值
        results.extend(response)

    if len(results) != len(data):
        raise RuntimeError(
            f"向量数量与文本数量不一致，文本数量={len(data)}，向量数量={len(results)}。"
        )

    return results


# =========================
# 封装 ChromaDB 向量数据库连接类
# =========================

class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        """
        初始化 ChromaDB 连接器。

        :param collection_name: ChromaDB 集合名称
        :param embedding_fn: embedding 处理函数
        """

        global CHROMADB_DIRECTORY

        # 实例化一个 chromadb 对象
        # 连接一个持久化的 ChromaDB 数据库
        chroma_client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)

        # 获取一个现有的向量集合，如果该集合不存在，则创建一个新的集合
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name
        )

        # embedding处理函数
        self.embedding_fn = embedding_fn

    def search(self, query: str, top_n: int):
        """
        检索向量数据库，返回最相似的文本片段。

        :param query: 查询文本
        :param top_n: 返回与查询向量最相似的前 n 个结果
        :return: ChromaDB 查询结果
        """

        try:
            if not query or not query.strip():
                raise ValueError("查询文本为空，无法检索向量数据库。")

            # 检查集合中是否有数据
            collection_count = self.collection.count()
            logger.info(f"当前 ChromaDB collection 数量: {collection_count}")

            if collection_count == 0:
                return {
                    "documents": [[]],
                    "distances": [[]],
                    "metadatas": [[]]
                }

            # 防止 top_n 大于数据库中的文档数量
            top_n = min(top_n, collection_count)

            # 计算查询文本的向量
            query_embeddings = self.embedding_fn([query])

            if not query_embeddings:
                raise RuntimeError("查询文本向量生成失败，query_embeddings为空。")

            # 将查询向量在向量数据库中进行相似度检索
            # 返回通常是个字典:
            # { "ids": [["id1", "id2"]],
            # "documents": [["张三九既往体检记录显示血压偏高。","张三九颈椎检查提示轻度退行性改变。"]],
            # "metadatas": [[None, None]],
            # "distances": [[0.23, 0.41]]}
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_n
            )

            return results

        except Exception as e:
            logger.exception(f"检索向量数据库时出错: {e}")
            raise


# =========================
# LangGraph 调用的检索函数
# =========================

def vectorSearch(user_query: str) -> str:
    """
    根据用户的问题，从健康档案库中检索相关内容。

    当前使用 Hybrid Search：
    1. 使用 ChromaDB 向量检索召回候选片段；
    2. 使用 BM25 关键词检索召回候选片段；
    3. 对两路结果进行分数融合；
    4. 返回融合排序后的 top-k 片段。

    返回值必须是字符串，方便写入 LangGraph 的 state["retrieved_context"]。

    :param user_query: 用户的问题
    :return: 检索到的健康档案文本，或检索失败说明
    """

    global CHROMADB_COLLECTION_NAME

    try:
        # 兼容某些调用方式传入 dict 的情况
        if isinstance(user_query, dict):
            user_query = user_query.get("user_query", "")

        # 若查询不是字符串，则转为字符串
        if not isinstance(user_query, str):
            user_query = str(user_query)

        user_query = user_query.strip()

        if not user_query:
            return "检索失败：用户问题为空。"

        logger.info(f"开始 Hybrid Search 检索健康档案，用户问题: {user_query}")

        # 初始化向量数据库连接器
        vector_db = MyVectorDBConnector(
            CHROMADB_COLLECTION_NAME,
            generate_vectors
        )

        # =========================================================
        # 第一步：ChromaDB 向量检索
        # =========================================================
        vector_search_results = vector_db.search(
            user_query,
            VECTOR_CANDIDATE_TOP_K
        )

        logger.info(f"ChromaDB 向量检索原始结果: {vector_search_results}")

        # 将 ChromaDB 向量检索的原始结果解析成统一格式，即将多个结果单独列出。
        vector_docs = parse_chroma_query_results(vector_search_results)

        logger.info(f"解析后的向量检索候选数量: {len(vector_docs)}")

        # =========================================================
        # 第二步：BM25 关键词检索
        # =========================================================
        bm25_docs = bm25_search_all(
            query=user_query,
            collection=vector_db.collection,
            top_k=BM25_CANDIDATE_TOP_K
        )

        logger.info(f"BM25 关键词检索候选数量: {len(bm25_docs)}")

        # =========================================================
        # 第三步：向量检索结果与 BM25 结果融合
        # 每个融合结果数据结构如下：
        # {'bm25_norm_score': 1.0,
        # 'bm25_score': 3.5609322108769668,
        # 'distance': 1.1164121627807617,
        # 'document': '健康档案...',
        # 'hybrid_score': 0.9398250467498676,
        # 'id': '8de1ef99-4b1f-4486-bf94-4e288d7e8073',
        # 'metadata': None,
        # 'vector_score': 0.9140357810712393}
        # =========================================================
        hybrid_docs = hybrid_fusion(
            vector_docs=vector_docs,
            bm25_docs=bm25_docs,
            alpha=HYBRID_ALPHA,
            top_k=FINAL_TOP_K
        )

        # # 使用 pprint 打印
        # print("=========== 融合后的数据结构 ===========")
        # pprint.pprint(hybrid_docs, indent=4, width=80)

        if not hybrid_docs:
            return "未从健康档案库中检索到相关内容。"

        # =========================================================
        # 第四步：格式化检索结果，返回给 LangGraph
        # =========================================================
        full_text = format_docs_for_context(hybrid_docs)

        logger.info(f"Hybrid Search 最终检索结果: {full_text}")

        return full_text

    except Exception as e:
        logger.exception(f"vectorSearch 执行失败: {e}")
        return f"检索工具执行失败：{str(e)}"
