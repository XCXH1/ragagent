import requests
import json
import logging
import os


# 设置日志模版
# 日志时间,当前模块名,志级别，例如 INFO / ERROR,具体日志内容
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 请求的 FastAPI 服务地址
# 需要先启动 main.py：
# python -X utf8 main.py
url = os.getenv("AGENTVQA_API_URL", "http://localhost:8012/agentvqa")
headers = {"Content-Type": "application/json"}


# 构造消息体
# 默认非流式输出 True or False
stream_flag = False

# 用户输入
content = "张三九身高175cm，体重82kg，请计算他的BMI，并结合健康档案给出建议。"

data = {
    "session_id": "p1",
    "messages": [
        {
            "role": "user",
            "content": content
        }
    ],
    "stream": stream_flag,
}


# 接收流式输出，且默认为 SSE 格式
if stream_flag:
    try:
        # 发送 POST 请求
        # stream=True 表示启用流式接收
        # timeout 设置大一些，因为 LangGraph + RAG + LLM 可能执行较久
        with requests.post(
            url,
            stream=True,
            headers=headers,
            json=data,
            timeout=300
        ) as response:

            # 如果服务端返回 4xx / 5xx，这里会直接抛出异常
            response.raise_for_status()

            # 按行读取服务端返回的 SSE 数据
            for line in response.iter_lines():
                if not line:
                    continue

                # requests收到的是bytes类型，将 bytes 解码为字符串
                line_str = line.decode("utf-8").strip()

                # LangGraph版 main.py 中，流式响应采用标准 SSE 格式：
                # data: {...}
                # data: [DONE]
                if not line_str.startswith("data: "):
                    logger.info(f"收到非SSE格式数据，跳过: {line_str}")
                    continue

                # 已确认是sse格式，因此去掉前len个字符即去掉 data: 前缀
                json_str = line_str[len("data: "):].strip()

                # 判断流式响应是否结束
                if json_str == "[DONE]":
                    logger.info("流式输出接收结束")
                    break

                # 检查是否为空
                if not json_str:
                    logger.info("收到空字符串，跳过...")
                    continue

                try:
                    chunk = json.loads(json_str)

                    choice = chunk["choices"][0]
                    print(f'**** {choice}')

                    # 如果 finish_reason 为 stop，说明模型输出结束
                    if choice.get("finish_reason") == "stop":
                        logger.info("接收JSON数据结束")
                        continue

                    # 获取流式内容
                    delta = choice.get("delta", {})
                    text = delta.get("content", "")

                    if text:
                        logger.info(f"流式输出，响应内容是: {text}")

                except json.JSONDecodeError as e:
                    logger.info(f"JSON解析错误: {e}, 原始内容: {json_str}")
                except Exception as e:
                    logger.info(f"解析流式数据时出错: {e}, 原始内容: {json_str}")

    except requests.exceptions.RequestException as e:
        logger.error(f"请求服务时出错: {e}")
    except Exception as e:
        logger.error(f"发生未知错误: {e}")


# 接收非流式输出处理
else:
    try:
        # 发送 POST 请求
        # json=data 会自动完成 json.dumps 和 Content-Type 处理
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=300
        )

        # 如果状态码不是 2xx，打印错误内容
        if response.status_code != 200:
            logger.error(f"请求失败，状态码: {response.status_code}")
            logger.error(f"错误响应内容: {response.text}")
            response.raise_for_status()

        # 将响应转成 JSON
        result = response.json()

        # logger.info(f"接收到返回的响应原始内容: {result}\n")

        # 兼容 OpenAI Chat Completions 格式
        content = result["choices"][0]["message"]["content"]

        logger.info(f"非流式输出，响应内容是: {content}\n")

    except requests.exceptions.RequestException as e:
        logger.error(f"请求服务时出错: {e}")
    except KeyError as e:
        logger.error(f"响应格式不符合预期，缺少字段: {e}")
        logger.error(f"响应原始内容: {response.text if 'response' in locals() else ''}")
    except Exception as e:
        logger.error(f"发生未知错误: {e}")

