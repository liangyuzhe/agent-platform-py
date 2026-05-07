"""LLM 文档预处理：提取元数据 + 生成摘要/假设性问题/关键事实。

核心原则：不要相信用户上传的原始文档能直接被机器完美理解。
"拒绝噪声，提高信息密度"是向量检索的第一原则。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from agents.model.chat_model import get_chat_model
from agents.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """LLM 预处理结果。"""
    category: str = ""                      # 文档分类
    tags: list[str] = field(default_factory=list)       # 标签
    entities: list[str] = field(default_factory=list)    # 关键实体
    summary: str = ""                       # 文档摘要（200 字以内）
    hypothetical_questions: list[str] = field(default_factory=list)  # 假设性问题
    key_facts: list[str] = field(default_factory=list)   # 关键事实


_SYSTEM_PROMPT = """你是一个文档分析专家。请分析以下文档内容，返回 JSON 格式的结构化信息。

要求：
1. category: 文档分类（如：财务制度、技术文档、合同协议、操作手册、报告、其他）
2. tags: 3-8 个关键标签（简短词语）
3. entities: 文档中提到的关键实体（人名、组织、产品、术语等）
4. summary: 文档摘要，200 字以内，概括核心内容
5. hypothetical_questions: 5-10 个假设性问题，即用户可能会用什么问题来查询这篇文档
6. key_facts: 3-8 个关键事实，文档中的重要数据点或结论

只返回 JSON，不要其他内容。格式：
{
  "category": "...",
  "tags": ["...", "..."],
  "entities": ["...", "..."],
  "summary": "...",
  "hypothetical_questions": ["...", "..."],
  "key_facts": ["...", "..."]
}"""


async def preprocess_document(text: str, filename: str = "") -> PreprocessResult:
    """用 LLM 预处理文档，提取元数据、摘要、假设性问题。

    Parameters
    ----------
    text:
        文档全文（截取前 8000 字避免超出上下文）。
    filename:
        文件名，作为上下文提示。

    Returns
    -------
    PreprocessResult
        结构化预处理结果。LLM 调用失败时返回空结果（不阻塞索引）。
    """
    # 截取前 8000 字，避免超出 LLM 上下文
    truncated = text[:8000]
    if len(text) > 8000:
        truncated += f"\n\n... (文档共 {len(text)} 字，已截取前 8000 字)"

    user_msg = f"文件名: {filename}\n\n文档内容:\n{truncated}" if filename else f"文档内容:\n{truncated}"

    try:
        model = get_chat_model(settings.chat_model_type)
        response = await model.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        raw = response.content.strip()
        # 提取 JSON（处理可能的 markdown 代码块包裹）
        if raw.startswith("```"):
            lines = raw.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            raw = "\n".join(json_lines)

        data = json.loads(raw)

        result = PreprocessResult(
            category=data.get("category", ""),
            tags=data.get("tags", []),
            entities=data.get("entities", []),
            summary=data.get("summary", ""),
            hypothetical_questions=data.get("hypothetical_questions", []),
            key_facts=data.get("key_facts", []),
        )
        logger.info(
            "Document preprocessed: category=%s, tags=%d, questions=%d, facts=%d",
            result.category, len(result.tags), len(result.hypothetical_questions), len(result.key_facts),
        )
        return result

    except Exception as e:
        logger.warning("Document preprocessing failed, using empty result: %s", e)
        return PreprocessResult()


def enrich_chunk_content(chunk_text: str, summary: str, questions: list[str]) -> str:
    """将摘要和假设性问题组合到 chunk 的 page_content 中。

    组合后的格式：
        [摘要] {summary}
        [原文] {chunk_text}
        [相关问题] {questions}

    这样 embedding 时同时编码了语义摘要和原文，检索时无论是用摘要式提问
    还是原文关键词都能命中。
    """
    parts = []
    if summary:
        parts.append(f"[摘要] {summary}")
    parts.append(f"[原文] {chunk_text}")
    if questions:
        # 取 2-3 个与此 chunk 相关的问题（简单策略：取前几个）
        parts.append(f"[相关问题] {'; '.join(questions[:3])}")
    return "\n\n".join(parts)
