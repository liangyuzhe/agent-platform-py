"""文档上传端点。"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import Literal

from agents.rag.indexing import build_indexing_graph, build_parent_indexing_graph
from agents.config.settings import settings

router = APIRouter()


class InsertDocumentResponse(BaseModel):
    success: bool
    message: str
    doc_ids: list[str] = []
    chunk_count: int = 0
    parent_count: int = 0


@router.post("/insert", response_model=InsertDocumentResponse)
async def insert_document(
    file: UploadFile = File(...),
    rag_mode: Literal["traditional", "parent"] | None = Form(None),
    source: Literal["user_document", "business_knowledge", "agent_knowledge"] = Form("user_document"),
    session_id: str = Form("default_user"),
):
    """上传文档并索引到统一知识库。

    source 决定文档分类：
    - user_document: 用户上传的一般文档（默认），按 session_id 隔离
    - business_knowledge: 业务知识（术语、公式、规则）
    - agent_knowledge: 智能体知识（SQL Q&A、操作指南）

    上传前会经过 LLM 预处理：提取元数据、生成摘要和假设性问题，
    丰富 chunk 内容以提高检索质量。
    """
    suffix = os.path.splitext(file.filename or ".txt")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    mode = rag_mode or settings.rag.mode

    try:
        if mode == "parent":
            index_fn = build_parent_indexing_graph(source=source, session_id=session_id)
            result = await index_fn(tmp_path)
            return InsertDocumentResponse(
                success=True,
                message=f"文档索引成功（{source}，Parent 模式），共 {result.get('parent_count', 0)} 个父分块，{result['chunk_count']} 个子分块",
                doc_ids=result["doc_ids"],
                chunk_count=result["chunk_count"],
                parent_count=result.get("parent_count", 0),
            )
        else:
            index_fn = build_indexing_graph(source=source, session_id=session_id)
            result = await index_fn(tmp_path)
            return InsertDocumentResponse(
                success=True,
                message=f"文档索引成功（{source}），共 {result['chunk_count']} 个分块",
                doc_ids=result["doc_ids"],
                chunk_count=result["chunk_count"],
            )
    except Exception as e:
        return InsertDocumentResponse(
            success=False,
            message=f"索引失败: {str(e)}",
        )
    finally:
        os.unlink(tmp_path)
