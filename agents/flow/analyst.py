"""Analyst 图：SQL 结果 -> 统计分析 -> 图表 + 文字报告。"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from agents.flow.state import AnalystState
from agents.model.chat_model import get_chat_model
from agents.tool.analyst_tools.parser import parse_sql_result
from agents.tool.analyst_tools.statistics import compute_statistics
from agents.tool.analyst_tools.chart import generate_chart_config
from agents.config.settings import settings


async def parse_data(state: AnalystState) -> dict:
    """解析 SQL 结果。"""
    parsed = parse_sql_result(state["sql_result"])
    return {"parsed_data": parsed}


async def analyze_data(state: AnalystState) -> dict:
    """计算统计指标。"""
    stats = compute_statistics(state["parsed_data"])
    return {"statistics": stats}


async def generate_report(state: AnalystState) -> dict:
    """LLM 生成文字分析报告。"""
    model = get_chat_model(settings.chat_model_type)
    data = state["parsed_data"]
    stats = state["statistics"]

    response = await model.ainvoke([
        HumanMessage(content=f"""请基于以下数据生成一段简洁的分析报告（200字以内）：

列: {data.get('columns', [])}
样本数据: {data.get('sample_rows', '')}
统计指标: {stats}
""")
    ])
    return {"text_analysis": response.content}


async def generate_chart(state: AnalystState) -> dict:
    """生成 ECharts 配置。"""
    chart_config = generate_chart_config(state["parsed_data"])
    return {"chart_config": chart_config}


async def merge_result(state: AnalystState) -> dict:
    """合并分析结果。"""
    return {
        "analysis_result": {
            "text_analysis": state["text_analysis"],
            "chart_config": state["chart_config"],
            "statistics": state["statistics"],
        }
    }


def build_analyst_graph():
    """构建 Analyst 图。generate_report 和 generate_chart 并行执行。"""
    graph = StateGraph(AnalystState)

    graph.add_node("parse_data", parse_data)
    graph.add_node("analyze_data", analyze_data)
    graph.add_node("generate_report", generate_report)
    graph.add_node("generate_chart", generate_chart)
    graph.add_node("merge_result", merge_result)

    graph.add_edge(START, "parse_data")
    graph.add_edge("parse_data", "analyze_data")
    graph.add_edge("analyze_data", "generate_report")
    graph.add_edge("analyze_data", "generate_chart")  # 并行分支
    graph.add_edge("generate_report", "merge_result")
    graph.add_edge("generate_chart", "merge_result")
    graph.add_edge("merge_result", END)

    return graph.compile()
