from dataclasses import dataclass, asdict

from pymilvus import AsyncMilvusClient, DataType, AnnSearchRequest, WeightedRanker

import config
from logger import logger
from plugins import Plugin


@dataclass()
class ChunkInserted:
    original_file_url: str
    file_name: str
    title: str
    content: str
    header_index: int
    dense_vector: list[float]
    sparse_vector: dict[int, float]


@dataclass()
class ChunkHit:
    id: int | str
    distance: float
    title: str
    content: str
    file_name: str
    original_file_url: str


class Milvus(Plugin):
    def __init__(self) -> None:
        scheme = "https" if config.MILVUS_SECURE else "http"
        self.client = AsyncMilvusClient(
            uri=f"{scheme}://{config.MILVUS_ENDPOINT}",
            token=config.MILVUS_TOKEN,
            db_name=config.MILVUS_DATABASE,
        )

    async def _add_chunk_collection(self) -> None:
        """按需创建 chunk 集合与索引；已存在则原样复用，不删除、不重建。"""
        collection_name = "chunk"
        if await self.client.has_collection(collection_name):
            logger.info("Milvus chunk 集合已存在，跳过创建")
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 主键
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        # 原文链接
        schema.add_field(field_name="original_file_url", datatype=DataType.VARCHAR, max_length=1000)
        # 文件名
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=255)
        # 标题
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        # 内容
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        # chunk索引号（标题在原文中的行号，可能远超 INT8 的 127 上限）
        schema.add_field(field_name="header_index", datatype=DataType.INT32)
        # 稠密向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,
                         dim=config.QWEN_EMBEDDING_DIMENSIONS)
        # 稀疏向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 创建索引
        index_params = self.client.prepare_index_params()
        # 为稠密向量构建索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        # 为稀疏向量构建索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            # IP：内积
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        await self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

    async def add_chunks(self, *chunks: ChunkInserted) -> None:
        collection_name = "chunk"
        batch_size = 10
        for index in range(0, len(chunks), batch_size):
            await self.client.insert(
                collection_name=collection_name,
                data=[asdict(chunk) for chunk in chunks[index:index + batch_size]]
            )
        await self.client.load_collection(collection_name)

    async def find_chunks(self,
                          dense_vectors: list[list[float]],
                          sparse_vectors: list[dict[int, float]],
                          limit: int = 10) -> list[ChunkHit]:
        collection_name = "chunk"
        # 构建稠密向量搜索请求
        a = AnnSearchRequest(
            data=dense_vectors,
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit
        )
        # 构建稀疏向量搜索请求
        b = AnnSearchRequest(
            data=sparse_vectors,
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=limit
        )

        ranker = WeightedRanker(0.5, 0.5, norm_score=True)
        response = await self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[a, b],
            ranker=ranker,
            limit=limit,
            output_fields=["id", "title", "content", "file_name", "original_file_url"]
        )
        items = []
        for chunks in response:
            for chunk in chunks:
                items.append(ChunkHit(
                    id=chunk["id"],
                    distance=chunk["distance"],
                    title=chunk["entity"]["title"],
                    content=chunk["entity"]["content"],
                    file_name=chunk["entity"]["file_name"],
                    original_file_url=chunk["entity"]["original_file_url"]
                ))
        return items

    async def register(self) -> None:
        """确保 chunk 集合与索引就绪。

        幂等：仅创建缺失的集合，已存在的集合原样复用、不删除也不重建，因此
        重启不会清空已有数据；集合的字段与索引需与代码保持一致，schema 变更
        需人工迁移。
        """
        await self._add_chunk_collection()

    async def close(self) -> None:
        await self.client.close()
