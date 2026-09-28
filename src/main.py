# 导入依赖包
import os
import uuid
import time
import json
import asyncio
import traceback
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from langchain_openai import ChatOpenAI
from config.api_keys import TONGYI_API_KEY, ZHIPU_API_KEY
from health_graph import build_health_graph
from langgraph.checkpoint.memory import InMemorySaver


# =========================
# 全局配置
# =========================

OPENAI_API_BASE = "https://api.z.ai/api/paas/v4/"
OPENAI_CHAT_API_KEY = ZHIPU_API_KEY
OPENAI_CHAT_MODEL = "glm-5.1"

ONEAPI_API_BASE = "https://api.z.ai/api/paas/v4/"
ONEAPI_CHAT_API_KEY = ZHIPU_API_KEY
ONEAPI_CHAT_MODEL = "glm-5.1"

TONGYI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TONGYI_CHAT_API_KEY = TONGYI_API_KEY
TONGYI_CHAT_MODEL = "qwen-plus"

OLLAMA_API_BASE = "http://192.168.2.9:11434/v1"
OLLAMA_CHAT_API_KEY = "NA"
OLLAMA_CHAT_MODEL = "llama3.1:latest"

PORT = int(os.getenv("AGENTVQA_PORT", "8012"))
MODEL_TYPE = os.getenv("AGENTVQA_MODEL_TYPE", "tongyi").lower()
SAVE_PDF = True

model = None
graph_app = None
checkpointer = None


# =========================
# 请求与响应模型
# =========================

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    session_id: Optional[str] = "default"


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


# =========================
# 外部功能函数
# =========================

def create_chat_model(model_type: str) -> ChatOpenAI:
    """
    根据 MODEL_TYPE 初始化不同的大模型。
    """

    if model_type == "tongyi":
        if not TONGYI_CHAT_API_KEY:
            raise ValueError("TONGYI_CHAT_API_KEY 未设置。")

        return ChatOpenAI(
            base_url=TONGYI_API_BASE,
            api_key=TONGYI_CHAT_API_KEY,
            model=TONGYI_CHAT_MODEL,
            temperature=0.7,
        )

    if model_type == "oneapi":
        if not ONEAPI_CHAT_API_KEY:
            raise ValueError("ONEAPI_CHAT_API_KEY 未设置。")

        return ChatOpenAI(
            base_url=ONEAPI_API_BASE,
            api_key=ONEAPI_CHAT_API_KEY,
            model=ONEAPI_CHAT_MODEL,
            temperature=0.7,
        )

    if model_type == "ollama":
        os.environ["OPENAI_API_KEY"] = "NA"

        return ChatOpenAI(
            base_url=OLLAMA_API_BASE,
            api_key=OLLAMA_CHAT_API_KEY,
            model=OLLAMA_CHAT_MODEL,
            temperature=0.7,
        )

    if not OPENAI_CHAT_API_KEY:
        raise ValueError("OPENAI_CHAT_API_KEY 未设置。")

    return ChatOpenAI(
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_CHAT_API_KEY,
        model=OPENAI_CHAT_MODEL,
        temperature=0.7,
    )

# 创建健康图
def create_health_graph(
    chat_model: ChatOpenAI,
    save_pdf: bool = True,
    checkpointer=None
):
    """
    初始化 LangGraph 健康档案工作流。

    :param chat_model: ChatOpenAI 或其他 LangChain ChatModel
    :param save_pdf: 是否保存 PDF
    :param checkpointer: LangGraph 记忆检查点，用于保存多轮对话状态
    :return: 编译后的 LangGraph 应用
    """

    return build_health_graph(
        model=chat_model,
        save_pdf=save_pdf,
        checkpointer=checkpointer
    )

# 检验并提取请求中的最后一条问题文本和 session_id
def validate_and_extract_query(request: ChatCompletionRequest):
    """
    校验请求，并提取用户最后一条消息作为当前问题，同时提取 session_id。

    session_id 用于区分不同会话。
    同一个 session_id 会共享同一份 LangGraph chat_history。
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages不能为空")

    query_prompt = request.messages[-1].content.strip()
    session_id = request.session_id or "default"

    if not query_prompt:
        raise HTTPException(status_code=400, detail="用户问题不能为空")

    return query_prompt, session_id

# 异步执行 langgraph 工作流
async def run_health_graph(query_prompt: str, session_id: str) -> dict:
    """
    执行 LangGraph 工作流。

    graph_app.invoke 是同步函数。
    当前 FastAPI 接口是 async def。
    因此使用 asyncio.to_thread 避免阻塞事件循环。

    session_id 会作为 LangGraph 的 thread_id，
    用于让 checkpointer 保存和恢复同一个会话的 chat_history。
    """
    if graph_app is None:
        raise HTTPException(status_code=500, detail="LangGraph工作流未初始化")

    initial_state = {
        "query": query_prompt
    }

    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    result_state = await asyncio.to_thread(
        graph_app.invoke,
        initial_state,
        config
    )

    return result_state

# 格式化返回 LangGraph 中的状态
def format_graph_result(result_state: dict) -> str:
    """
    将 LangGraph 最终状态整理成返回给用户的文本。
    """
    report = result_state.get("report", "").strip()
    pdf_result = result_state.get("pdf_result", "").strip()

    if not report:
        report = "未生成有效回答。"

    if pdf_result:
        return f"{report}\n\n{pdf_result}"

    return report

# 处理流式响应
async def generate_stream_response(formatted_response: str):
    """
    生成 OpenAI 风格的流式 SSE 响应。
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

    lines = formatted_response.split("\n")

    for line in lines:
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": line + "\n"
                    },
                    "finish_reason": None
                }
            ]
        }

        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.05)

    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    
    # 发送最终 JSON 数据块，说明本次流式回答已结束
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    # 发送特殊结束标记，说明后面不会再有数据
    yield "data: [DONE]\n\n"

# 处理非流式响应
def build_non_stream_response(formatted_response: str) -> JSONResponse:
    """
    构造非流式 JSON 响应。
    """
    response = ChatCompletionResponse(
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=Message(
                    role="assistant",
                    content=formatted_response
                ),
                finish_reason="stop"
            )
        ]
    )

    return JSONResponse(content=response.model_dump())


def build_stream_response(formatted_response: str) -> StreamingResponse:
    """
    构造流式响应。
    """

    # media_type设置为text/event-stream以符合SSE(Server-SentEvents) 格式
    # StreamingResponse用于持续打开http，一边生成数据，一边用yiedl发送给客户端
    # text/event-streamg 告诉客户端这个是 SSE 事件流
    return StreamingResponse(
        generate_stream_response(formatted_response),
        media_type="text/event-stream"
    )


# =========================
# FastAPI 生命周期
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, graph_app, checkpointer
    # 若部署到生产环境，需要替换为 PostgreSQL 或 Redis 等持久化 Checkpointer
    checkpointer = InMemorySaver()

    try:
        print("正在初始化模型")
        model = create_chat_model(MODEL_TYPE)
        print("LLM初始化完成")

        print("正在初始化 LangGraph 健康档案工作流")
        graph_app = create_health_graph(
            chat_model=model,
            save_pdf=SAVE_PDF,
            checkpointer=checkpointer
        )
        print("LangGraph 健康档案工作流初始化完成")

    except Exception as e:
        print(f"初始化过程中出错: {str(e)}")
        traceback.print_exc()
        raise

    yield

    print("正在关闭...")


app = FastAPI(lifespan=lifespan)


# =========================
# API 接口
# =========================

@app.post("/agentvqa")
async def chat_completions(request: ChatCompletionRequest):
    global model, graph_app

    if model is None:
        raise HTTPException(status_code=500, detail="模型服务未初始化")

    if graph_app is None:
        raise HTTPException(status_code=500, detail="LangGraph工作流未初始化")

    try:
        query_prompt, session_id = validate_and_extract_query(request)

        print(f"用户问题是: {query_prompt}")
        print(f"当前会话ID是: {session_id}")

        result_state = await run_health_graph(
            query_prompt=query_prompt,
            session_id=session_id
        )

        # 从 LangGraph 的最终状态中提取并格式化回答文本
        formatted_response = format_graph_result(result_state)

        print(f"LLM最终回复结果: {formatted_response}")

        if request.stream:
            return build_stream_response(formatted_response)

        return build_non_stream_response(formatted_response)

    except HTTPException:
        raise

    except Exception as e:
        print("处理聊天完成时出错:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 启动服务
# =========================

if __name__ == "__main__":
    print(f"在端口 {PORT} 上启动服务器")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
