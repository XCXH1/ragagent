# tools/medicalAgentTools.py

import re
from typing import List, Tuple

from langchain_core.tools import tool


@tool
def calculate_bmi(height_cm: float, weight_kg: float) -> str:
    """
    根据身高和体重计算 BMI。

    参数：
    height_cm: 身高，单位为厘米，例如 175
    weight_kg: 体重，单位为千克，例如 82

    返回：
    BMI 计算结果和简要说明。
    """

    try:
        height_cm = float(height_cm)
        weight_kg = float(weight_kg)

        if height_cm <= 0 or weight_kg <= 0:
            return "BMI计算失败：身高和体重必须大于0。"

        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)

        return (
            f"BMI计算结果：{bmi:.2f}。\n"
            f"计算公式：BMI = 体重kg / 身高m² = {weight_kg:.2f} / ({height_m:.2f}²)。\n"
            "说明：BMI只能作为体重状态的粗略参考，不能单独作为医学诊断依据，"
            "还需要结合年龄、性别、肌肉量、腰围、基础疾病和医生评估。"
        )

    except Exception as e:
        return f"BMI计算失败：{str(e)}"


@tool
def average_blood_pressure(readings: str) -> str:
    """
    计算多次血压读数的平均值。

    参数：
    readings: 血压读数字符串，格式示例：
              "120/80, 130/85, 125/82"

    返回：
    平均收缩压和平均舒张压。
    """

    try:
        if not readings or not str(readings).strip():
            return "血压平均值计算失败：未提供血压读数。"

        readings = str(readings)

        # 匹配形如 120/80、130 / 85 的血压读数
        pairs = re.findall(r"(\d{2,3})\s*/\s*(\d{2,3})", readings)

        if not pairs:
            return (
                "血压平均值计算失败：没有识别到有效血压读数。"
                "请使用类似 120/80, 130/85 的格式。"
            )

        valid_pairs: List[Tuple[float, float]] = []

        for systolic, diastolic in pairs:
            systolic_value = float(systolic)
            diastolic_value = float(diastolic)

            # 简单过滤明显不合理的输入
            if systolic_value < 50 or systolic_value > 260:
                continue

            if diastolic_value < 30 or diastolic_value > 160:
                continue

            valid_pairs.append((systolic_value, diastolic_value))

        if not valid_pairs:
            return "血压平均值计算失败：输入的血压读数超出合理范围。"

        avg_systolic = sum(item[0] for item in valid_pairs) / len(valid_pairs)
        avg_diastolic = sum(item[1] for item in valid_pairs) / len(valid_pairs)

        return (
            f"共识别到 {len(valid_pairs)} 组有效血压读数。\n"
            f"平均收缩压：{avg_systolic:.1f} mmHg。\n"
            f"平均舒张压：{avg_diastolic:.1f} mmHg。\n"
            "说明：单次或少量血压读数不能直接作为诊断依据，"
            "应结合多日家庭血压、诊室血压或动态血压监测结果综合判断。"
        )

    except Exception as e:
        return f"血压平均值计算失败：{str(e)}"


@tool
def classify_blood_pressure_reference(systolic: float, diastolic: float) -> str:
    """
    根据收缩压和舒张压给出成人血压参考分层。

    参数：
    systolic: 收缩压，单位 mmHg，例如 135
    diastolic: 舒张压，单位 mmHg，例如 88

    返回：
    成人血压参考分层说明。
    """

    try:
        systolic = float(systolic)
        diastolic = float(diastolic)

        if systolic <= 0 or diastolic <= 0:
            return "血压参考分层失败：收缩压和舒张压必须大于0。"

        if systolic < 50 or systolic > 260 or diastolic < 30 or diastolic > 160:
            return "血压参考分层失败：输入血压值超出常见生理范围，请检查输入。"

        if systolic >= 180 or diastolic >= 120:
            level = "明显升高范围"
            note = "建议尽快结合症状和医生评估，必要时及时就医。"
        elif systolic >= 140 or diastolic >= 90:
            level = "高血压范围参考"
            note = "建议结合多次测量、既往病史和医生评估进一步判断。"
        elif systolic >= 120 or diastolic >= 80:
            level = "偏高范围参考"
            note = "建议规律监测血压，并关注生活方式因素。"
        else:
            level = "通常参考范围内"
            note = "仍需结合个体情况和医生建议综合判断。"

        return (
            f"输入血压：{systolic:.0f}/{diastolic:.0f} mmHg。\n"
            f"参考分层：{level}。\n"
            f"提示：{note}\n"
            "说明：该工具只提供规则化参考，不构成医学诊断。"
        )

    except Exception as e:
        return f"血压参考分层失败：{str(e)}"


def get_medical_agent_tools():
    """
    返回医学健康助手可调用的工具列表。

    health_graph.py 中会通过该函数获取工具，
    然后绑定到支持 tool calling 的大模型上。
    """

    return [
        calculate_bmi,
        average_blood_pressure,
        classify_blood_pressure_reference,
    ]