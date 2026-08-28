# health_graph.py

import time
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

import uuid
from typing import List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools.vectorSearchTool import vectorSearch
from tools.savePdfTool import saveText2Pdf
from tools.medicalAgentTools import get_medical_agent_tools


class HealthGraphState(TypedDict, total=False):
    """
    LangGraph 全局状态。

    query: 当前用户问题
    chat_history: 多轮对话历史，每轮结束后追加用户问题和助手回答
    retrieved_context: 当前问题检索到的健康档案内容
    messages: 当前问题内部的工具调用消息
    tool_results: 当前问题的工具调用结果
    tool_round_count: 当前问题内的工具调用轮数
    report: 当前问题的最终回答或报告
    pdf_result: PDF保存结果
    error: 错误信息
    """

    query: str

    # 多轮对话历史，跨轮保存
    chat_history: Annotated[List[BaseMessage], add_messages]

    retrieved_context: str

    # 当前问题内部的 tool calling 消息，不作为长期对话历史
    messages: Annotated[List[BaseMessage], add_messages]

    tool_results: str
    tool_round_count: int
    report: str
    pdf_result: str
    error: str
    
class HealthGraphBuilder:
    """
    健康档案 LangGraph 工作流构建器。

    这个类负责：
    1. 保存模型对象 model
    2. 保存是否生成 PDF 的配置 save_pdf
    3. 定义各个 LangGraph 节点
    4. 构建并编译 LangGraph 工作流
    """

    def __init__(
        self,
        model,
        save_pdf: bool = True,
        chat_history_max_tokens: int = 2000,
    ):
        """
        初始化工作流构建器。

        :param model: ChatOpenAI 或其他 LangChain ChatModel
        :param save_pdf: 是否保存 PDF
        :param chat_history_max_tokens: chat_history 超过该阈值时触发摘要压缩
        """

        self.model = model
        self.save_pdf = save_pdf
        self.chat_history_max_tokens = chat_history_max_tokens

        # 医学工具列表
        self.medical_tools = get_medical_agent_tools()
        # 保存 ToolNode，后面用自定义 medical_tools_node 包装它
        self.medical_tool_node = ToolNode(self.medical_tools)

        # 绑定工具调用能力
        try:
            self.tool_model = model.bind_tools(self.medical_tools)
            self.enable_tool_calling = True
        except Exception as e:
            print(f"当前模型不支持 bind_tools，已跳过工具调用能力。错误信息: {e}")
            self.tool_model = model
            self.enable_tool_calling = False

    # 给 messages 补id
    def ensure_message_id(self, message: BaseMessage) -> BaseMessage:
        """
        确保单条 message 一定有 id。

        如果模型或 ToolNode 已经自动生成了 id，则保留原 id；
        如果没有 id，则使用 uuid 手动补一个。
        """
        
        prefix = ""

        if getattr(message, "id", None):
            return message

        if isinstance(message, HumanMessage):
            prefix = "human"
        if isinstance(message, AIMessage):
            prefix = "ai"
        if isinstance(message, SystemMessage):
            prefix = "system"
        if isinstance(message, ToolMessage):
            prefix = "tool"

        new_id = f"{prefix}_{uuid.uuid4().hex}"

        # 大多数 LangChain Message 可以直接赋值 id
        try:
            message.id = new_id
            return message

        # 兼容部分不可变对象或 Pydantic 对象
        except Exception:
            if hasattr(message, "model_copy"):
                return message.model_copy(update={"id": new_id})

            return message.copy(update={"id": new_id})

    def prepare_turn_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        每轮对话开始前的准备节点。

        作用：
        1. 清空上一轮工具调用 messages；
        2. 重置当前轮状态；
        3. 检查 chat_history 是否超过阈值；
        4. 如果超过阈值，则调用 LLM 总结成一条摘要消息；
        5. 压缩后的 chat_history 中只保留这一条摘要消息。
        """

        print("========== LangGraph Node: 对话轮次准备节点 ==========")

        chat_history = state.get("chat_history", [])
        messages = state.get("messages", [])

        updates = {
            "tool_round_count": 0,
            "tool_results": "",
            "retrieved_context": "",
            "report": "",
            "pdf_result": "",
        }

        # =========================================================
        # 1. 清空上一轮工具调用 messages
        # =========================================================
        remove_tool_messages = []

        for message in messages:
            message_id = getattr(message, "id", None)

            if message_id:
                remove_tool_messages.append(
                    RemoveMessage(id=message_id)
                )

        if remove_tool_messages:
            updates["messages"] = remove_tool_messages

        # =========================================================
        # 2. 估算 chat_history token 数
        #    这里用字符数 / 2 近似 token 数
        # =========================================================
        total_chars = 0

        for message in chat_history:
            total_chars += len(str(message.content or ""))

        estimated_tokens = max(1, total_chars // 2)

        print(f"当前 chat_history 估算 token 数: {estimated_tokens}")

        # 未超出阈值，直接返回更新后的messages
        if estimated_tokens <= self.chat_history_max_tokens:
            return updates

        print("chat_history 超过阈值，开始压缩为一条摘要消息。")

        # =========================================================
        # 3. 调用 LLM 总结历史
        # =========================================================
        summarize_messages = [
            SystemMessage(
                content=(
                    "你是一个对话历史摘要助手。"
                    "请将下面的历史对话压缩成一条简洁摘要，用于后续医学健康问答。"
                    "摘要必须保留对后续对话有用的信息，例如患者姓名、症状、体检结果、"
                    "用户关注点、已经计算过的指标、助手已经给出的重要结论。"
                    "不要添加历史中没有的信息。"
                    f"摘要长度必须尽量小于 {self.chat_history_max_tokens} 个 token。"
                )
            )
        ]

        summarize_messages.extend(chat_history)

        summarize_messages.append(
            HumanMessage(
                content=(
                    "请基于以上历史对话生成一条压缩摘要。"
                    "只输出摘要内容，不要输出解释。"
                )
            )
        )

        # 临时对话，用于压缩，因此不用设置id，不会进入langgraph状态中。
        summary_response = self.model.invoke(summarize_messages)
        summary_text = summary_response.content.strip()

        # =========================================================
        # 4. 删除旧 chat_history，并写入一条新的摘要消息
        # =========================================================
        remove_chat_history = []

        # 为每一条旧消息创建一个 RemoveMessage(id=message_id)。
        # 当 LangGraph 的 Reducer 接收到 RemoveMessage 时，它会去状态列表中找到对应 ID 的消息并将其销毁。
        for message in chat_history:
            message_id = getattr(message, "id", None)

            if message_id:
                remove_chat_history.append(
                    RemoveMessage(id=message_id)
                )
        # 将生成的摘要文本包装成一个新的 SystemMessage，并利用 uuid.uuid4().hex 为其赋予一个全新的全局唯一 ID。
        summary_message = SystemMessage(
            content=f"历史对话摘要：{summary_text}",
            id=f"summary_{uuid.uuid4().hex}"
        )

        updates["chat_history"] = remove_chat_history + [summary_message]

        return updates

    def retrieve_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        对应原来的 retrieval_agent：
        健康档案检索专家。

        该节点负责：
        1. 从 state 中读取用户问题 query
        2. 调用 vectorSearch 从健康档案向量库中检索相关内容
        3. 将检索结果写入 retrieved_context
        """

        query = state.get("query", "").strip()

        if not query:
            return {
                "retrieved_context": "",
                "error": "用户问题为空，无法检索健康档案。"
            }

        print("========== LangGraph Node: 健康档案检索节点 ==========")
        print(f"检索问题: {query}")

        retrieved_context = vectorSearch(query)

        print(f"检索结果: {retrieved_context}")

        return {
            "retrieved_context": retrieved_context
        }
    
    def has_valid_context(self, context: str) -> bool:
        """
        判断检索到的健康档案内容是否有效。

        context: retrieved_context
        return: True 表示有有效检索结果，False 表示无有效检索结果
        """

        invalid_signals = [
            "未从健康档案库中检索到相关内容",
            "检索失败",
            "检索工具执行失败",
            "检索结果为空",
            "用户问题为空"
        ]

        if not context:
            return False

        for signal in invalid_signals:
            if signal in context:
                return False

        return True

    def medical_tool_agent_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        多工具 Agent 节点。
        chat_history: 跨轮历史对话，可能是完整历史，也可能是压缩后的摘要消息
        messages: 当前轮工具调用过程

        该节点负责：
        1. 读取用户问题 query；
        2. 读取健康档案检索结果 retrieved_context；
        3. 判断是否需要调用医学计算工具；
        4. 如果需要，则由模型生成 tool_calls；
        5. 如果不需要，则直接进入后续报告生成流程。
        """

        query = state.get("query", "")
        retrieved_context = state.get("retrieved_context", "")
        chat_history = state.get("chat_history", [])
        messages = state.get("messages", [])
        tool_round_count = state.get("tool_round_count", 0)

        print("========== LangGraph Node: 多工具 Agent 判断节点 ==========")

        # 如果当前模型不支持工具调用，则跳过工具调用
        # 这里设置id 是为了下一轮清空 messages 时提供id。于chat_history无关。
        if not self.enable_tool_calling:
            return {
                "messages": [
                    AIMessage(
                        content="当前模型不支持工具调用，已跳过医学计算工具。",
                        id=f"ai_{uuid.uuid4().hex}"
                        )
                ],
                "tool_round_count": tool_round_count + 1
            }

        # 当前用户问题第一次进入工具判断节点
        if tool_round_count == 0:
            current_tool_messages = [
                SystemMessage(
                    content=(
                        "你是一个医学健康助手中的工具调度 Agent。"
                        "你的任务不是直接生成最终健康报告，而是判断是否需要调用工具。"
                        "如果用户问题涉及 BMI、身高体重、血压平均值、血压参考分层等数值计算，"
                        "你应调用合适的工具。"
                        "如果用户问题不需要计算工具，则不要调用工具，只需回复“不需要调用额外工具”。"
                        "你不能直接做医学诊断，工具结果只能作为健康报告的参考信息。"
                    ),
                    id=f"system_{uuid.uuid4().hex}"
                )
            ]

            # 直接把历史对话消息传入
            current_tool_messages.extend(chat_history)

            # 传入人类提问信息
            current_tool_messages.append(
                HumanMessage(
                    content=(
                        f"用户问题：\n{query}\n\n"
                        f"已检索到的健康档案内容：\n{retrieved_context}\n\n"
                        "请判断是否需要调用医学计算工具。不要重复调用已经完成的工具。如果仍缺少必要计算，只调用尚未完成的工具。"
                    ),
                    id=f"human_{uuid.uuid4().hex}"
                )
            )

            response = self.tool_model.invoke(current_tool_messages)
            # 确保模型返回的 Message 有 id
            response = self.ensure_message_id(response)

            # 返回tool_call字段
            return {
                "messages": current_tool_messages + [response],
                "tool_round_count": tool_round_count + 1
            }

        # 如果是工具执行后再次回到该节点，则把已有 messages 继续交给模型
        response = self.tool_model.invoke(messages)
        # 确保模型返回的 Message 有 id
        response = self.ensure_message_id(response)

        return {
            "messages": [response],
            "tool_round_count": tool_round_count + 1
        }
    
    # 工具执行
    def medical_tools_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        包装 LangGraph 的 ToolNode。
        保证 ToolMessage 一定有 id，
        """

        tool_output = self.medical_tool_node.invoke(state)

        # ToolNode 通常返回 {"messages": [ToolMessage(...)]}
        if isinstance(tool_output, dict) and "messages" in tool_output:
            tool_output["messages"] = [self.ensure_message_id(message) for message in tool_output.get("messages", [])]

        return tool_output
    
    def collect_tool_results_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        收集工具调用结果节点。

        ToolNode 执行工具后，会把工具返回值保存为 ToolMessage。
        本节点负责从 messages 中提取所有 ToolMessage，
        并拼接成 tool_results，供后续 report_node 使用。
        """

        print("========== LangGraph Node: 工具结果收集节点 ==========")

        messages = state.get("messages", [])

        tool_messages = [
            message for message in messages
            if isinstance(message, ToolMessage)
        ]

        if not tool_messages:
            return {
                "tool_results": "未调用额外医学计算工具。"
            }

        tool_result_parts = []

        for index, message in enumerate(tool_messages, start=1):
            tool_result_parts.append(
                f"【工具结果{index}】\n{message.content}"
            )

        tool_results = "\n\n".join(tool_result_parts)

        print(f"工具调用结果: {tool_results}")

        return {
            "tool_results": tool_results
        }

    def route_after_tools(self, state: HealthGraphState) -> Literal["report", "no_context_report"]:
        """
        工具调用完成后的条件边。

        如果存在有效健康档案内容，或者存在有效工具结果，则进入 report 节点；
        如果二者都没有，则进入 no_context_report 节点。
        """

        context = state.get("retrieved_context", "")
        tool_results = state.get("tool_results", "")

        has_context = self.has_valid_context(context)

        has_tool_results = (
            bool(tool_results.strip())
            and "未调用额外医学计算工具" not in tool_results
        )

        # 有检索结果或有工具调用结果
        if has_context or has_tool_results:
            return "report"

        return "no_context_report"

    def no_context_report_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        没有检索到健康档案内容时，生成保守回答。

        该节点的目标是：
        1. 不编造健康档案中不存在的信息
        2. 不做确定性医学诊断
        3. 明确说明当前档案证据不足
        """

        query = state.get("query", "")
        context = state.get("retrieved_context", "")

        print("========== LangGraph Node: 无档案证据回答节点 ==========")

        messages = [
            SystemMessage(
                content=(
                    "你是一个健康档案问答助手。"
                    "当前没有可靠的健康档案证据时，不能编造事实，不能做医学诊断。"
                    "你需要明确说明目前无法判断，并建议补充检查记录或咨询医生。"
                )
            ),
            HumanMessage(
                content=(
                    f"医生询问的健康问题：{query}\n\n"
                    f"健康档案检索结果：{context}\n\n"
                    "请给出谨慎、简洁的回答。"
                )
            )
        ]

        # 没有返回 messages，因此不会进入langgraph状态，不用设置id。
        response = self.model.invoke(messages)

        return {
            "report": response.content
        }

    def report_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        对应原来的 report_agent：
        健康报告撰写专家。

        该节点负责：
        1. 读取用户问题 query
        2. 读取检索到的健康档案 retrieved_context
        3. 调用大模型生成结构化健康报告
        4. 将报告写入 report 字段
        """

        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        retrieved_context = state.get("retrieved_context", "")
        tool_results = state.get("tool_results", "")

        print("========== LangGraph Node: 健康报告撰写节点 ==========")

        report_messages = [
            SystemMessage(
                content=(
                    "你是一位医学背景的健康报告撰写专家，擅长分析健康档案并生成专业、"
                    "简洁、易于医生理解的健康建议报告。"
                    "你需要基于检索到的健康档案内容回答医生的问题。"
                    "回答必须有医学依据，但不能替代医生诊断。"
                    "如果证据不足，需要明确说明证据不足。"
                )
            ) 
        ]

        # 直接把历史对话或历史摘要传入
        report_messages.extend(chat_history)

        report_messages.append(
            HumanMessage(
                content=(
                    f"医生询问的健康问题：\n{query}\n\n"
                    f"从健康档案中检索到的相关记录：\n{retrieved_context}\n\n"
                    f"医学计算工具结果：\n{tool_results if tool_results else '未调用额外医学计算工具。'}\n\n"
                    "请结合上述健康档案内容和工具计算结果，撰写一份简洁明了、富有医学依据的健康建议报告。\n"
                    "报告需要包括以下部分：\n"
                    "一、问题概述\n"
                    "二、检索到的相关健康档案信息\n"
                    "三、医学计算工具结果\n"
                    "四、与当前问题的可能关联分析\n"
                    "五、证据不足或需要注意的地方\n"
                    "六、建议\n\n"
                    "注意：\n"
                    "1. 不能编造健康档案中没有的信息；\n"
                    "2. 不能直接给出确定诊断；\n"
                    "3. 工具计算结果只能作为辅助参考，不能替代医生诊断；\n"
                    "4. 如果没有调用工具，需要说明未使用额外医学计算工具；\n"
                    "5. 可以使用“可能相关”“需要进一步检查”“建议医生结合临床表现判断”等谨慎表述。"
                )
            )
        )

        response = self.model.invoke(report_messages)

        return {
            "report": response.content
        }

    def save_pdf_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        保存 PDF 节点。

        该节点负责：
        1. 从 state 中读取最终报告 report
        2. 判断是否需要保存 PDF
        3. 调用 saveText2Pdf 将报告保存到 output 目录
        4. 将 PDF 保存结果写入 pdf_result 字段
        """

        if not self.save_pdf:
            return {}

        report = state.get("report", "")

        if not report:
            return {
                "pdf_result": "PDF未保存：报告内容为空。"
            }

        filename = f"zhangsanjiu_health_report_{int(time.time())}.pdf"
        pdf_result = saveText2Pdf(report, filename)

        print("========== LangGraph Node: PDF保存节点 ==========")
        print(pdf_result)

        return {
            "pdf_result": pdf_result
        }

    def update_chat_history_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        每轮对话结束后，将当前用户问题和最终回答写入 chat_history。

        因为 chat_history 使用 add_messages，
        所以这里返回 HumanMessage 和 AIMessage 后会自动追加到历史中。
        """

        print("========== LangGraph Node: 更新对话历史节点 ==========")

        query = state.get("query", "")
        report = state.get("report", "")

        history_updates = []

        # 将当前轮的问题和最终报告写入chat_history
        if query:
            history_updates.append(
                HumanMessage(
                    content=query,
                    id=f"human_{uuid.uuid4().hex}"
                )
            )

        if report:
            history_updates.append(
                AIMessage(
                    content=report,
                    id=f"ai_{uuid.uuid4().hex}"
                )
            )

        if not history_updates:
            return {}

        return {
            "chat_history": history_updates
        }

    def build(self, checkpointer=None):
        """
        START
        ↓
        prepare_turn
        ↓
        retrieve
        ↓
        medical_tool_agent
        ↓
        medical_tools / collect_tool_results
        ↓
        report 或 no_context_report
        ↓
        save_pdf
        ↓
        update_chat_history
        ↓
        END
        """

        # 实例化图对象
        graph_builder = StateGraph(HealthGraphState)

        # 添加节点
        graph_builder.add_node("prepare_turn", self.prepare_turn_node)
        graph_builder.add_node("retrieve", self.retrieve_node)

        # 多工具 Agent 判断节点
        graph_builder.add_node("medical_tool_agent", self.medical_tool_agent_node)

        # 真正执行工具的节点
        graph_builder.add_node("medical_tools", self.medical_tools_node)

        # 收集工具执行结果的节点
        graph_builder.add_node("collect_tool_results", self.collect_tool_results_node)

        graph_builder.add_node("report", self.report_node)
        graph_builder.add_node("no_context_report", self.no_context_report_node)
        graph_builder.add_node("save_pdf", self.save_pdf_node)
        graph_builder.add_node("update_chat_history", self.update_chat_history_node)

        graph_builder.add_edge(START, "prepare_turn")
        graph_builder.add_edge("prepare_turn", "retrieve")

        graph_builder.add_edge("retrieve", "medical_tool_agent")

        # medical_tool_agent 执行后，判断是否产生工具调用
        # 工具条件边:根据最后一条 AIMessage 是否有 tool_calls决策，产生 tool_calls，则进入 medical_tools,否则进入 collect_tool_results
        graph_builder.add_conditional_edges(
            "medical_tool_agent",
            tools_condition,
            {
                "tools": "medical_tools",
                END: "collect_tool_results",
            }
        )

        # 工具执行完成后，回到 medical_tool_agent
        # 让模型读取工具结果，并判断是否还需要继续调用工具
        graph_builder.add_edge("medical_tools", "medical_tool_agent")

        # 工具结果收集完成后，判断是进入 report 还是 no_context_report
        graph_builder.add_conditional_edges(
            "collect_tool_results",
            self.route_after_tools,
            {
                "report": "report",
                "no_context_report": "no_context_report"
            }
        )

        # 普通边
        graph_builder.add_edge("report", "save_pdf")
        graph_builder.add_edge("no_context_report", "save_pdf")
        graph_builder.add_edge("save_pdf", "update_chat_history")
        graph_builder.add_edge("update_chat_history", END)

        # 编译并返回 LangGraph 应用
        return graph_builder.compile(checkpointer=checkpointer)


def build_health_graph(model, save_pdf: bool = True, checkpointer=None):
    """
    对外暴露的构建函数。

    main.py 中仍然可以保持原来的调用方式：

    graph_app = build_health_graph(model, save_pdf=SAVE_PDF)

    :param model: ChatOpenAI 或其他 LangChain ChatModel
    :param save_pdf: 是否保存 PDF
    :return: 编译后的 LangGraph 应用
    """

    return HealthGraphBuilder(
        model=model,
        save_pdf=save_pdf,
        chat_history_max_tokens=2000,
    ).build(checkpointer=checkpointer)