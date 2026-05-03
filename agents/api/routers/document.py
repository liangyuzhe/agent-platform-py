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
):
    """上传文档并索引。"""
    # 保存到临时文件
    suffix = os.path.splitext(file.filename or ".txt")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    mode = rag_mode or settings.rag.mode

    try:
        if mode == "parent":
            index_fn = build_parent_indexing_graph()
            result = index_fn(tmp_path)
            return InsertDocumentResponse(
                success=True,
                message=f"文档索引成功（Parent 模式），共 {result['parent_count']} 个父分块，{result['chunk_count']} 个子分块",
                doc_ids=result["doc_ids"],
                chunk_count=result["chunk_count"],
                parent_count=result["parent_count"],
            )
        else:
            index_fn = build_indexing_graph()
            result = index_fn(tmp_path)
            return InsertDocumentResponse(
                success=True,
                message=f"文档索引成功，共 {result['chunk_count']} 个分块",
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
