# health_graph.py

import time
import uuid
import json
import re
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

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
    多智能体健康档案问答系统的全局状态。

    query: 当前用户问题
    chat_history: 跨轮历史对话
    messages: 当前轮工具调用消息
    retrieved_context: 健康档案检索结果
    tool_results: 工具调用结果
    report: 最终报告
    pdf_result: PDF 保存结果
    """

    query: str

    # 跨轮保存的长期历史
    chat_history: Annotated[List[BaseMessage], add_messages]

    # 当前轮工具调用过程消息，不作为长期历史
    messages: Annotated[List[BaseMessage], add_messages]

    # Supervisor Agent 决策结果
    next_agent: str
    need_retrieval: bool
    need_tools: bool
    supervisor_reason: str

    # Retrieval Agent 输出
    retrieval_query: str
    retrieved_context: str

    # Medical Tool Agent 输出
    tool_results: str
    tool_round_count: int

    # Report Agent 输出
    draft_report: str
    report: str

    # Review Agent 输出
    review_result: str
    review_feedback: str
    review_round_count: int

    pdf_result: str
    error: str


class HealthGraphBuilder:
    """
    多智能体健康档案 LangGraph 工作流构建器。

    多智能体角色：
    1. Memory / Prepare Agent: 负责清理当前轮消息、压缩历史
    2. Supervisor Agent: 负责判断任务流转
    3. Retrieval Agent: 负责健康档案检索
    4. Medical Tool Agent: 负责医学工具调用
    5. Report Agent: 负责生成健康报告
    6. Review Agent: 负责医学安全审核
    7. PDF Agent: 负责保存报告
    """

    def __init__(
        self,
        model,
        save_pdf: bool = True,
        chat_history_max_tokens: int = 2000,
        keep_recent_messages: int = 6,
        max_tool_rounds: int = 3,
        max_review_rounds: int = 1,
    ):
        self.model = model
        self.save_pdf = save_pdf
        self.chat_history_max_tokens = chat_history_max_tokens
        self.keep_recent_messages = keep_recent_messages
        self.max_tool_rounds = max_tool_rounds
        self.max_review_rounds = max_review_rounds

        # 医学工具列表
        self.medical_tools = get_medical_agent_tools()

        # ToolNode 用于真正执行工具
        self.medical_tool_node = ToolNode(self.medical_tools)

        # 绑定工具调用能力
        try:
            self.tool_model = model.bind_tools(self.medical_tools)
            self.enable_tool_calling = True
        except Exception as e:
            print(f"当前模型不支持 bind_tools，已跳过工具调用能力。错误信息: {e}")
            self.tool_model = model
            self.enable_tool_calling = False

    # =========================================================
    # 通用工具函数
    # =========================================================

    def ensure_message_id(self, message: BaseMessage) -> BaseMessage:
        """
        确保进入 LangGraph state 的 message 一定有 id。

        chat_history 和 messages 都使用 add_messages，
        所以 message id 对后续追加、替换、删除非常重要。
        """

        if getattr(message, "id", None):
            return message

        prefix = "msg"

        if isinstance(message, HumanMessage):
            prefix = "human"
        elif isinstance(message, AIMessage):
            prefix = "ai"
        elif isinstance(message, SystemMessage):
            prefix = "system"
        elif isinstance(message, ToolMessage):
            prefix = "tool"

        new_id = f"{prefix}_{uuid.uuid4().hex}"

        try:
            message.id = new_id
            return message
        except Exception:
            if hasattr(message, "model_copy"):
                return message.model_copy(update={"id": new_id})

            return message.copy(update={"id": new_id})

    def parse_json_from_text(self, text: str) -> dict:
        """
        从大模型输出中解析 JSON。

        兼容以下情况：
        1. 直接输出 JSON
        2. 输出 ```json ... ```
        3. JSON 前后带解释文字
        """

        if not text:
            return {}

        text = text.strip()

        # 去掉 Markdown JSON 代码块,re用于匹配text中特定开头或结尾的字符，替换为空。
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

        # 先直接解析
        try:
            return json.loads(text)
        except Exception:
            # 有异常说明去掉 Markdown 标记后仍无法直接转为json
            pass

        # 直接匹配 '{}' 部分内容，尝试提取第一个 JSON 对象
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}

        return {}

    def has_valid_context(self, context: str) -> bool:
        """
        判断检索到的健康档案内容是否有效。
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

    # =========================================================
    # 1. Memory / Prepare Agent
    # =========================================================

    def prepare_turn_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        Memory / Prepare Agent。

        作用：
        1. 清空上一轮工具调用 messages；
        2. 重置当前轮中间状态；
        3. 检查 chat_history 是否过长；
        4. 如果过长，则压缩为摘要消息。
        """

        print("========== Agent: Memory / Prepare ==========")

        chat_history = state.get("chat_history", [])
        tool_messages = state.get("messages", [])

        updates = {
            "tool_round_count": 0,
            "tool_results": "",
            "retrieved_context": "",
            "retrieval_query": "",
            "draft_report": "",
            "report": "",
            "pdf_result": "",
            "review_result": "",
            "review_feedback": "",
            "review_round_count": 0,
            "next_agent": "",
            "need_retrieval": False,
            "need_tools": False,
            "supervisor_reason": "",
        }

        # 1. 根据id清空上一轮工具调用 messages
        remove_tool_messages = []

        for message in tool_messages:
            message_id = getattr(message, "id", None)
            if message_id:
                remove_tool_messages.append(RemoveMessage(id=message_id))

        if remove_tool_messages:
            updates["messages"] = remove_tool_messages

        # 2. 估算 chat_history token 数
        total_chars = 0
        for message in chat_history:
            total_chars += len(str(message.content or ""))

        estimated_tokens = max(1, total_chars // 2)

        print(f"当前 chat_history 估算 token 数: {estimated_tokens}")

        if estimated_tokens <= self.chat_history_max_tokens:
            return updates

        print("chat_history 超过阈值，开始压缩为摘要消息。")

        # 3. 压缩历史
        summarize_messages = [
            SystemMessage(
                content=(
                    "你是一个对话历史摘要助手。"
                    "请将下面的历史对话压缩成一条简洁摘要，用于后续医学健康问答。"
                    "摘要必须保留对后续有用的信息，例如患者姓名、症状、体检结果、"
                    "用户关注点、已经计算过的指标、助手已经给出的重要结论。"
                    "不要添加历史中没有的信息。"
                    f"摘要长度尽量小于 {self.chat_history_max_tokens} 个 token。"
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

        summary_response = self.model.invoke(summarize_messages)
        summary_text = summary_response.content.strip()

        # 4. 删除旧历史，写入摘要
        remove_chat_history = []

        for message in chat_history:
            message_id = getattr(message, "id", None)
            if message_id:
                remove_chat_history.append(RemoveMessage(id=message_id))

        summary_message = SystemMessage(
            content=f"历史对话摘要：{summary_text}",
            id=f"summary_{uuid.uuid4().hex}"
        )

        updates["chat_history"] = remove_chat_history + [summary_message]

        return updates

    # =========================================================
    # 2. Supervisor Agent
    # =========================================================

    def supervisor_agent_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        Supervisor Agent。

        作用：
        判断当前问题应该交给哪个智能体处理。
        它不直接回答用户问题，只负责调度。
        """

        print("========== Agent: Supervisor ==========")

        query = state.get("query", "")
        chat_history = state.get("chat_history", [])

        messages = [
            SystemMessage(
                content=(
                    "你是一个健康档案多智能体系统的 Supervisor Agent。"
                    "你的任务不是回答用户问题，而是判断下一步应该交给哪个智能体处理。\n\n"

                    "系统中有以下智能体：\n"
                    "1. retrieval_agent：负责检索健康档案、体检记录、既往健康数据。\n"
                    "2. medical_tool_agent：负责调用医学计算工具，例如 BMI、血压平均值、健康指标计算等。\n"
                    "3. report_agent：在已有足够信息后，负责生成最终健康建议报告。\n"
                    "4. no_context_agent：当没有健康档案证据，且无法通过工具得到有效信息时，给出保守回答。\n\n"

                    "判断规则：\n"
                    "1. 如果问题要求结合健康档案、体检记录、既往记录，need_retrieval=true。\n"
                    "2. 如果问题涉及 BMI、身高体重、血压、数值计算、指标判断，need_tools=true。\n"
                    "3. 如果既需要档案又需要计算，优先进入 retrieval_agent。\n"
                    "4. 如果只需要计算，进入 medical_tool_agent。\n"
                    "5. 如果既不需要档案也不需要计算，但可以基于常识谨慎回答，进入 report_agent。\n"
                    "6. 如果没有信息支撑，进入 no_context_agent。\n\n"

                    "请只输出 JSON，不要输出解释。格式如下：\n"
                    "{\n"
                    '  "need_retrieval": true,\n'
                    '  "need_tools": true,\n'
                    '  "next_agent": "retrieval_agent",\n'
                    '  "reason": "简要说明调度理由"\n'
                    "}"
                )
            )
        ]

        messages.extend(chat_history)

        messages.append(
            HumanMessage(
                content=(
                    f"当前用户问题：\n{query}\n\n"
                    "请判断下一步应该由哪个智能体处理。"
                )
            )
        )

        response = self.model.invoke(messages)
        decision = self.parse_json_from_text(response.content) # 提取llm返回的json内容

        need_retrieval = bool(decision.get("need_retrieval", True))
        need_tools = bool(decision.get("need_tools", False))
        next_agent = decision.get("next_agent", "")
        reason = decision.get("reason", "")

        valid_agents = {
            "retrieval_agent",
            "medical_tool_agent",
            "report_agent",
            "no_context_agent"
        }

        if next_agent not in valid_agents:
            if need_retrieval:
                next_agent = "retrieval_agent"
            elif need_tools:
                next_agent = "medical_tool_agent"
            else:
                next_agent = "report_agent"

        print(f"Supervisor 决策: {next_agent}, need_retrieval={need_retrieval}, need_tools={need_tools}")

        return {
            "need_retrieval": need_retrieval,
            "need_tools": need_tools,
            "next_agent": next_agent,
            "supervisor_reason": reason
        }

    def route_by_supervisor(
        self,
        state: HealthGraphState
    ) -> Literal["retrieval_agent", "medical_tool_agent", "report_agent", "no_context_agent"]:
        """
        根据 Supervisor Agent 的决策进行路由。
        """

        next_agent = state.get("next_agent", "retrieval_agent")

        if next_agent == "medical_tool_agent":
            return "medical_tool_agent"

        if next_agent == "report_agent":
            return "report_agent"

        if next_agent == "no_context_agent":
            return "no_context_agent"

        return "retrieval_agent"

    # =========================================================
    # 3. Retrieval Agent
    # =========================================================

    def retrieval_agent_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        Retrieval Agent。

        作用：
        1. 从 state 中读取 query；
        2. 检索健康档案；
        3. 将检索结果写入 retrieved_context。

        默认调用 vectorSearch(query)。
        """

        print("========== Agent: Retrieval ==========")

        query = state.get("query", "").strip()

        if not query:
            return {
                "retrieval_query": "",
                "retrieved_context": "",
                "error": "用户问题为空，无法检索健康档案。"
            }

        print(f"Retrieval Agent 检索问题: {query}")

        try:
            retrieved_context = vectorSearch(query)
        except Exception as e:
            retrieved_context = f"检索失败: {str(e)}"

        print(f"Retrieval Agent 检索结果: {retrieved_context}")

        return {
            "retrieval_query": query,
            "retrieved_context": retrieved_context
        }

    def route_after_retrieval(
        self,
        state: HealthGraphState
    ) -> Literal["medical_tool_agent", "report_agent", "no_context_agent"]:
        """
        Retrieval Agent 后的路由。

        如果问题需要工具，则进入 Medical Tool Agent。
        如果不需要工具但有有效档案，则进入 Report Agent。
        如果既没有有效档案，也不需要工具，则进入 No Context Agent。
        """

        need_tools = state.get("need_tools", False)
        context = state.get("retrieved_context", "")

        # 如果需要工具，就调用工具agent
        if need_tools:
            return "medical_tool_agent"
        
        # 若检索到合法的信息，就生成报告
        if self.has_valid_context(context):
            return "report_agent"

        return "no_context_agent"

    # =========================================================
    # 4. Medical Tool Agent
    # =========================================================

    def medical_tool_agent_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        Medical Tool Agent。

        作用：
        1. 判断当前问题是否需要调用医学计算工具；
        2. 如果需要，生成 tool_calls；
        3. 如果不需要，说明不需要调用工具。
        """

        print("========== Agent: Medical Tool ==========")

        query = state.get("query", "")
        retrieved_context = state.get("retrieved_context", "")
        chat_history = state.get("chat_history", [])
        messages = state.get("messages", [])
        tool_round_count = state.get("tool_round_count", 0)

        # 防止工具调用死循环
        if tool_round_count >= self.max_tool_rounds:
            return {
                "messages": [
                    AIMessage(
                        content="工具调用轮数已达到上限，不再继续调用工具。",
                        id=f"ai_{uuid.uuid4().hex}"
                    )
                ],
                "tool_round_count": tool_round_count + 1
            }

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

        # 当前轮第一次进入工具 Agent
        if tool_round_count == 0:
            current_tool_messages = [
                SystemMessage(
                    content=(
                        "你是健康档案多智能体系统中的 Medical Tool Agent。"
                        "你的任务不是生成最终报告，而是判断是否需要调用医学计算工具。\n\n"
                        "当用户问题涉及以下内容时，应调用工具：\n"
                        "1. BMI、身高、体重；\n"
                        "2. 血压平均值、血压分层；\n"
                        "3. 明确的数值计算；\n"
                        "4. 工具列表中支持的医学计算。\n\n"
                        "如果不需要工具，请不要调用工具，只回复“不需要调用额外工具”。"
                        "不要直接做医学诊断。工具结果只作为后续报告参考。"
                    ),
                    id=f"system_{uuid.uuid4().hex}"
                )
            ]

            # 加入跨轮历史，让 Tool Agent 理解上下文
            current_tool_messages.extend(chat_history)

            current_tool_messages.append(
                HumanMessage(
                    content=(
                        f"用户问题：\n{query}\n\n"
                        f"健康档案检索结果：\n{retrieved_context}\n\n"
                        "请判断是否需要调用医学计算工具。"
                        "不要重复调用已经完成的工具。"
                        "如果仍缺少必要计算，只调用尚未完成的工具。"
                    ),
                    id=f"human_{uuid.uuid4().hex}"
                )
            )

            # 执行工具调用，并给返回中的工具字段添加id，方便后续清理。
            response = self.tool_model.invoke(current_tool_messages)
            response = self.ensure_message_id(response)

            return {
                "messages": current_tool_messages + [response],
                "tool_round_count": tool_round_count + 1
            }

        # 工具执行后再次回到 Tool Agent，agent会根据现存信息给出哪些工具以调用，还需要调用哪些工具
        response = self.tool_model.invoke(messages)
        response = self.ensure_message_id(response)

        return {
            "messages": [response],
            "tool_round_count": tool_round_count + 1
        }

    def medical_tools_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        真正执行工具的节点。

        这里包装 ToolNode，确保 ToolMessage 一定有 id。
        """

        print("========== ToolNode: Medical Tools ==========")

        tool_output = self.medical_tool_node.invoke(state)

        if isinstance(tool_output, dict):
            if "messages" in tool_output:
                tool_output["messages"] = [
                    self.ensure_message_id(message)
                    for message in tool_output.get("messages", [])
                ]

            return tool_output

        if isinstance(tool_output, list):
            return {
                "messages": [
                    self.ensure_message_id(message)
                    for message in tool_output
                ]
            }

        return {}

    def collect_tool_results_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        收集工具调用结果。
        """

        print("========== Node: Collect Tool Results ==========")

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

    def route_after_tools(
        self,
        state: HealthGraphState
    ) -> Literal["report_agent", "no_context_agent"]:
        """
        工具结果收集完成后的路由。
        """

        context = state.get("retrieved_context", "")
        tool_results = state.get("tool_results", "")
        
        # 确认是否有检索结果
        has_context = self.has_valid_context(context)

        has_tool_results = (
            bool(tool_results.strip())
            and "未调用额外医学计算工具" not in tool_results
        )

        # 有检索结果或工具调用结果就转到报告生成agent
        if has_context or has_tool_results:
            return "report_agent"

        return "no_context_agent"

    # =========================================================
    # 5. Report Agent
    # =========================================================

    def report_agent_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        Report Agent。

        基于：
        1. 用户问题；
        2. 历史对话；
        3. 健康档案检索结果；
        4. 医学工具结果；

        生成结构化健康建议报告。
        """

        print("========== Agent: Report ==========")

        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        retrieved_context = state.get("retrieved_context", "")
        tool_results = state.get("tool_results", "")
        supervisor_reason = state.get("supervisor_reason", "")

        report_messages = [
            SystemMessage(
                content=(
                    "你是健康档案多智能体系统中的 Report Agent。"
                    "你擅长基于健康档案、医学计算工具结果和用户问题生成结构化健康建议报告。\n\n"
                    "要求：\n"
                    "1. 必须基于检索到的健康档案内容和工具结果回答；\n"
                    "2. 不能编造健康档案中没有的信息；\n"
                    "3. 不能直接给出确定性医学诊断；\n"
                    "4. 如果证据不足，需要明确说明证据不足；\n"
                    "5. 工具计算结果只能作为辅助参考，不能替代医生诊断；\n"
                    "6. 表述要谨慎，可以使用“可能相关”“建议进一步检查”等说法。"
                )
            )
        ]

        report_messages.extend(chat_history)

        report_messages.append(
            HumanMessage(
                content=(
                    f"用户问题：\n{query}\n\n"
                    f"Supervisor 调度理由：\n{supervisor_reason}\n\n"
                    f"健康档案检索结果：\n"
                    f"{retrieved_context if retrieved_context else '未检索到有效健康档案。'}\n\n"
                    f"医学计算工具结果：\n"
                    f"{tool_results if tool_results else '未调用额外医学计算工具。'}\n\n"
                    "请生成一份结构化健康建议报告，包括：\n"
                    "一、问题概述\n"
                    "二、检索到的相关健康档案信息\n"
                    "三、医学计算工具结果\n"
                    "四、与当前问题的可能关联分析\n"
                    "五、证据不足或需要注意的地方\n"
                    "六、建议"
                )
            )
        )

        response = self.model.invoke(report_messages)
        report = response.content.strip()

        return {
            "draft_report": report,
            "report": report
        }

    # =========================================================
    # 6. Review Agent
    # =========================================================

    def review_agent_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        Review Agent。

        负责审核报告：
        1. 是否编造档案中不存在的信息；
        2. 是否存在过度诊断；
        3. 是否忽略证据不足；
        4. 是否需要更谨慎表达。
        """

        print("========== Agent: Review ==========")

        query = state.get("query", "")
        retrieved_context = state.get("retrieved_context", "")
        tool_results = state.get("tool_results", "")
        report = state.get("report", "")
        review_round_count = state.get("review_round_count", 0)

        review_messages = [
            SystemMessage(
                content=(
                    "你是健康档案多智能体系统中的 Review Agent。"
                    "你的任务是审核 Report Agent 生成的报告是否安全、谨慎、符合证据。\n\n"

                    "重点检查：\n"
                    "1. 是否编造健康档案中不存在的信息；\n"
                    "2. 是否把工具计算结果解释成确定诊断；\n"
                    "3. 是否遗漏证据不足说明；\n"
                    "4. 是否存在过度医疗建议；\n"
                    "5. 是否需要更加谨慎的医学表达。\n\n"

                    "请只输出 JSON，不要输出解释。\n"
                    "如果通过，输出：\n"
                    '{"review_result": "pass", "review_feedback": ""}\n\n'
                    "如果需要修改，输出：\n"
                    '{"review_result": "need_revision", "review_feedback": "具体修改意见"}'
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：\n{query}\n\n"
                    f"健康档案检索结果：\n{retrieved_context}\n\n"
                    f"医学工具结果：\n{tool_results}\n\n"
                    f"待审核报告：\n{report}\n\n"
                    "请审核该报告。"
                )
            )
        ]

        # 审核报告并提取返回结果
        response = self.model.invoke(review_messages)
        result = self.parse_json_from_text(response.content)

        review_result = result.get("review_result", "pass")
        review_feedback = result.get("review_feedback", "")
        
        # 如果返回结果不在要求范围内，默认报告无误
        if review_result not in {"pass", "need_revision"}:
            review_result = "pass"
            review_feedback = ""

        print(f"Review 结果: {review_result}, feedback: {review_feedback}")

        return {
            "review_result": review_result,
            "review_feedback": review_feedback,
            "review_round_count": review_round_count + 1
        }

    def route_after_review(
        self,
        state: HealthGraphState
    ) -> Literal["revise_report_agent", "save_pdf"]:
        """
        Review Agent 后的路由。

        如果审核不通过，且未超过最大修改次数，则进入 revise_report_agent。
        否则保存 PDF。
        """

        review_result = state.get("review_result", "pass")
        review_round_count = state.get("review_round_count", 0)

        if (
            review_result == "need_revision"
            and review_round_count <= self.max_review_rounds
        ):
            return "revise_report_agent"

        # 不需要返稿或超过修改次数则直接返还结果
        return "save_pdf"

    def revise_report_agent_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        Revise Report Agent。

        根据 Review Agent 的反馈，对报告进行一次修改。
        """

        print("========== Agent: Revise Report ==========")

        query = state.get("query", "")
        retrieved_context = state.get("retrieved_context", "")
        tool_results = state.get("tool_results", "")
        report = state.get("report", "")
        review_feedback = state.get("review_feedback", "")

        revise_messages = [
            SystemMessage(
                content=(
                    "你是健康报告修改智能体。"
                    "你需要根据审核反馈修改报告，使其更加符合健康档案证据，"
                    "避免编造信息、避免确定性诊断，并保持谨慎医学表达。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：\n{query}\n\n"
                    f"健康档案检索结果：\n{retrieved_context}\n\n"
                    f"医学工具结果：\n{tool_results}\n\n"
                    f"原始报告：\n{report}\n\n"
                    f"审核反馈：\n{review_feedback}\n\n"
                    "请输出修改后的完整报告。"
                )
            )
        ]

        # 获取llm返回结果
        response = self.model.invoke(revise_messages)
        revised_report = response.content.strip()

        return {
            "report": revised_report
        }

    # =========================================================
    # 7. No Context Agent
    # =========================================================

    def no_context_agent_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        No Context Agent。

        当没有可靠健康档案证据，也没有有效工具结果时，生成保守回答。
        """

        print("========== Agent: No Context ==========")

        query = state.get("query", "")
        context = state.get("retrieved_context", "")

        messages = [
            SystemMessage(
                content=(
                    "你是健康档案问答系统中的 No Context Agent。"
                    "当前没有可靠的健康档案证据时，不能编造事实，不能做确定性医学诊断。"
                    "你需要明确说明目前证据不足，并建议补充检查记录或咨询医生。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户健康问题：\n{query}\n\n"
                    f"健康档案检索结果：\n{context}\n\n"
                    "请给出谨慎、简洁的回答。"
                )
            )
        ]

        response = self.model.invoke(messages)

        return {
            "report": response.content.strip()
        }

    # =========================================================
    # 8. PDF Agent
    # =========================================================

    def save_pdf_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        PDF Agent。

        将最终报告保存为 PDF。
        """

        print("========== Agent: PDF Save ==========")

        if not self.save_pdf:
            return {}

        report = state.get("report", "")

        if not report:
            return {
                "pdf_result": "PDF未保存：报告内容为空。"
            }

        filename = f"zhangsanjiu_health_report_{int(time.time())}.pdf"
        pdf_result = saveText2Pdf(report, filename)

        print(pdf_result)

        return {
            "pdf_result": pdf_result
        }

    # =========================================================
    # 9. Update Chat History Agent
    # =========================================================

    def update_chat_history_node(self, state: HealthGraphState) -> HealthGraphState:
        """
        每轮对话结束后，将当前用户问题和最终回答写入 chat_history。
        """

        print("========== Agent: Update Chat History ==========")

        query = state.get("query", "")
        report = state.get("report", "")

        history_updates = []

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

    # =========================================================
    # 构建多智能体图
    # =========================================================

    def build(self, checkpointer=None):
        """
        多智能体工作流：

        START
        ↓
        prepare_turn
        ↓
        supervisor_agent
        ↓
        retrieval_agent / medical_tool_agent / report_agent / no_context_agent
        ↓
        report_agent
        ↓
        review_agent
        ↓
        revise_report_agent / save_pdf
        ↓
        update_chat_history
        ↓
        END
        """

        graph_builder = StateGraph(HealthGraphState)

        # 1. 节点注册
        graph_builder.add_node("prepare_turn", self.prepare_turn_node)
        graph_builder.add_node("supervisor_agent", self.supervisor_agent_node)

        graph_builder.add_node("retrieval_agent", self.retrieval_agent_node)
        graph_builder.add_node("medical_tool_agent", self.medical_tool_agent_node)
        graph_builder.add_node("medical_tools", self.medical_tools_node)
        graph_builder.add_node("collect_tool_results", self.collect_tool_results_node)

        graph_builder.add_node("report_agent", self.report_agent_node)
        graph_builder.add_node("review_agent", self.review_agent_node)
        graph_builder.add_node("revise_report_agent", self.revise_report_agent_node)

        graph_builder.add_node("no_context_agent", self.no_context_agent_node)
        graph_builder.add_node("save_pdf", self.save_pdf_node)
        graph_builder.add_node("update_chat_history", self.update_chat_history_node)

        # 2. 起始流程
        graph_builder.add_edge(START, "prepare_turn")
        graph_builder.add_edge("prepare_turn", "supervisor_agent")

        # 3. Supervisor 路由
        graph_builder.add_conditional_edges(
            "supervisor_agent",
            self.route_by_supervisor,
            {
                "retrieval_agent": "retrieval_agent",
                "medical_tool_agent": "medical_tool_agent",
                "report_agent": "report_agent",
                "no_context_agent": "no_context_agent",
            }
        )

        # 4. 检索后路由
        graph_builder.add_conditional_edges(
            "retrieval_agent",
            self.route_after_retrieval,
            {
                "medical_tool_agent": "medical_tool_agent",
                "report_agent": "report_agent",
                "no_context_agent": "no_context_agent",
            }
        )

        # 5. Medical Tool Agent 工具调用路由
        # 工具条件函数会根据medical_tool_agent输出中是否函授tool_call字段判断是否转入工具执行节点
        graph_builder.add_conditional_edges(
            "medical_tool_agent",
            tools_condition,
            {
                "tools": "medical_tools",
                END: "collect_tool_results",
            }
        )

        # 工具执行后回到 Medical Tool Agent，以防还需要调用其他工具
        graph_builder.add_edge("medical_tools", "medical_tool_agent")

        # 工具结果收集后进入报告或无上下文回答
        graph_builder.add_conditional_edges(
            "collect_tool_results",
            self.route_after_tools,
            {
                "report_agent": "report_agent",
                "no_context_agent": "no_context_agent",
            }
        )

        # 6. 报告生成后进入审核
        graph_builder.add_edge("report_agent", "review_agent")

        # 7. 审核后决定是否修改
        graph_builder.add_conditional_edges(
            "review_agent",
            self.route_after_review,
            {
                "revise_report_agent": "revise_report_agent",
                "save_pdf": "save_pdf",
            }
        )

        # 修改后再次审核
        graph_builder.add_edge("revise_report_agent", "review_agent")

        # 无上下文回答直接保存
        graph_builder.add_edge("no_context_agent", "save_pdf")

        # 8. 收尾
        graph_builder.add_edge("save_pdf", "update_chat_history")
        graph_builder.add_edge("update_chat_history", END)

        # 注意：这里必须传入 checkpointer，main.py 中的 session_id/thread_id 才能生效
        return graph_builder.compile(checkpointer=checkpointer)


def build_health_graph(model, save_pdf: bool = True, checkpointer=None):
    """
    对外暴露的构建函数。
    """

    return HealthGraphBuilder(
        model=model,
        save_pdf=save_pdf,
        chat_history_max_tokens=2000,
        keep_recent_messages=6,
        max_tool_rounds=3,
        max_review_rounds=1,
    ).build(checkpointer=checkpointer)


