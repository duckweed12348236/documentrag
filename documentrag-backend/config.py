import os
import tomllib
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent
PYPROJECT_FILE = BASE_DIR / "pyproject.toml"

# --------------------------------------------------------------------------- #
# 日志配置
# --------------------------------------------------------------------------- #
LOG_DIR = BASE_DIR / "logs"
LOG_LEVEL = "INFO"
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
# 控制台输出
LOG_CONSOLE_ENABLED = True
LOG_CONSOLE_LEVEL = LOG_LEVEL
LOG_CONSOLE_COLORIZE = True
# 文件输出
# 进程级后缀：Celery worker 子进程由 plugins/celery/__init__.py 注入
# MAS_LOG_SUFFIX=".worker"，与 API 进程分文件写入，避免跨进程轮转互相干扰
LOG_FILE_SUFFIX = os.environ.get("MAS_LOG_SUFFIX", "")
LOG_FILE_NAME = f"{{time:YYYY-MM-DD}}{LOG_FILE_SUFFIX}.log"
LOG_FILE_LEVEL = LOG_LEVEL
LOG_FILE_ROTATION = "00:00"  # 每天零点创建新日志文件
LOG_FILE_RETENTION = None  # 永久保留，不过期
LOG_FILE_COMPRESSION = "zip"
LOG_FILE_ENCODING = "utf-8"
LOG_FILE_ENQUEUE = True  # 异步写入，保证多线程/多进程安全
# 公共项
LOG_BACKTRACE = True
LOG_DIAGNOSE = False
# 接管标准库 logging（uvicorn / fastapi 等第三方库日志统一走 loguru）
LOG_INTERCEPT_ENABLED = True
LOG_INTERCEPT_ROOT_LEVEL = LOG_LEVEL
LOG_INTERCEPT_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
)

POSTGRES_HOST = ""
POSTGRES_PORT = ""
POSTGRES_USER = ""
POSTGRES_PASSWORD = ""
POSTGRES_DATABASE = ""
POSTGRES_POOL_MIN_SIZE = 1
POSTGRES_POOL_MAX_SIZE = 10

REDIS_HOST = ""
REDIS_PORT = ""
REDIS_DB = 0
REDIS_PASSWORD = ""

# 短期记忆（checkpoint）缓存的过期时间（秒）：Redis 中的副本按此 TTL 独立
# 续期，过期后由 PostgreSQL 回源并回填，因此调小只会降低命中率
SHORT_TERM_MEMORY_TTL = 3600

# --------------------------------------------------------------------------- #
# Celery 配置
# --------------------------------------------------------------------------- #
# 消息代理与结果后端连接串；留空时回退为按上面 Redis 配置拼装的 redis:// 连接串
CELERY_BROKER_URL = ""
CELERY_RESULT_URL = ""

DEEPSEEK_MODEL_URL = ""
DEEPSEEK_MODEL_KEY = ""
DEEPSEEK_MODEL_NAME = ""

# --------------------------------------------------------------------------- #
# Qwen 文本向量模型配置（阿里云百炼 DashScope 原生接口）
# --------------------------------------------------------------------------- #
# 一次调用同时产出稠密 + 稀疏向量：稠密向量入 Milvus 的 dense_vector 字段、稀疏向量入
# sparse_vector 字段（混合检索两路都用本模型），也供 Ragas 等按 Embeddings 复用
# 原生接口根地址（插件在其后拼 /services/embeddings/text-embedding/text-embedding）；
# 使用百炼 MaaS 工作空间网关时改为
# https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1
QWEN_EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1"
QWEN_EMBEDDING_KEY = ""
# 模型名：qwen3.7-text-embedding / qwen3.7-text-embedding-flash
QWEN_EMBEDDING_NAME = "qwen3.7-text-embedding"
# 原生接口请求参数 parameters.dimension 取值：2560 / 2048 / 1536 / 1024 / 768 / 512 / 256
# 默认 1024，与 plugins/milvus.py 中 dense_vector 字段维度一致
QWEN_EMBEDDING_DIMENSIONS = 1024

# --------------------------------------------------------------------------- #
# Qwen3-Reranker 文本重排模型配置（硅基流动 OpenAI 兼容 rerank 端点）
# --------------------------------------------------------------------------- #
# 完整请求地址（OpenAI 协议没有重排端点，各家路径不统一，故整条地址由配置给出）
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
# 模型名：Qwen/Qwen3-Reranker-8B
RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"
# 硅基流动 API Key
RERANK_MODEL_KEY = ""
# 任务说明字段名：硅基流动为 instruction，百炼 qwen3-rerank 为 instruct
RERANK_INSTRUCT_FIELD = "instruction"

# --------------------------------------------------------------------------- #
# Milvus 向量数据库配置
# --------------------------------------------------------------------------- #
# gRPC 服务端点，形如 "host:port"（不带协议前缀，由 MILVUS_SECURE 决定 http / https）
MILVUS_ENDPOINT = "localhost:19530"
# 是否启用 TLS：本地 docker-compose 未启用证书，默认关闭
MILVUS_SECURE = False
# 连接使用的数据库名（Milvus 内置库为 default）
MILVUS_DATABASE = "default"
# 认证 token，形如 "user:password"；服务端未开启鉴权时留空
MILVUS_TOKEN = ""

# --------------------------------------------------------------------------- #
# Tortoise ORM 配置
# --------------------------------------------------------------------------- #
# 连接名 / 应用名，供生成配置字典时复用
TORTOISE_CONNECTION = "default"
TORTOISE_APP = "models"
# 业务模型所在模块，Tortoise 以 app 为单位组织模型
TORTOISE_MODELS = ["models"]
# 迁移文件所在模块：必须显式声明——makemigrations 会自行推导并注入，
# 但 migrate 直接读取本配置，缺此键时会被当成「无迁移的 app」而报 No migrations to apply
TORTOISE_MIGRATIONS = "models.migrations"
# 引擎：使用 psycopg3 异步驱动，与项目的 psycopg 依赖保持一致
TORTOISE_ENGINE = "tortoise.backends.psycopg"
# 是否在读写时间字段时按 TORTOISE_TIMEZONE 做时区转换
TORTOISE_USE_TZ = True
TORTOISE_TIMEZONE = "Asia/Shanghai"
TORTOISE_ORM = {
    "connections": {
        TORTOISE_CONNECTION: {
            "engine": TORTOISE_ENGINE,
            "credentials": {
                "host": POSTGRES_HOST,
                "port": POSTGRES_PORT,
                "user": POSTGRES_USER,
                "password": POSTGRES_PASSWORD,
                "database": POSTGRES_DATABASE,
                "minsize": POSTGRES_POOL_MIN_SIZE,
                "maxsize": POSTGRES_POOL_MAX_SIZE,
            },
        }
    },
    "apps": {
        TORTOISE_APP: {
            "models": TORTOISE_MODELS,
            "default_connection": TORTOISE_CONNECTION,
            "migrations": TORTOISE_MIGRATIONS,
        }
    },
    "use_tz": TORTOISE_USE_TZ,
    "timezone": TORTOISE_TIMEZONE,
}

SNOWFLAKE_DATACENTER_ID = 0
SNOWFLAKE_WORKER_ID = 0
SNOWFLAKE_SEQUENCE = 0

# --------------------------------------------------------------------------- #
# Tavily 联网搜索配置
# --------------------------------------------------------------------------- #
# API Key：https://app.tavily.com 注册后获取；留空时联网搜索不可用
# （离线调用与评测不受影响，检索流水线会跳过联网搜索环节）
TAVILY_KEY = ""
# 单次检索返回的网页片段数量上限
TAVILY_MAX_RESULTS = 5
# 检索深度：basic（默认，快且便宜）/ advanced（更全，更贵）
TAVILY_SEARCH_DEPTH = "basic"

# --------------------------------------------------------------------------- #
# MinerU 客户端配置
# --------------------------------------------------------------------------- #
# v4 精准解析 API 根地址，文档：https://mineru.net/apiManage/docs
MINERU_BASE_URL = "https://mineru.net/api/v4"
# API 管理页面创建的 token，调用时拼成 ``Authorization: Bearer <token>``
MINERU_TOKEN = ""
# 解析产物 Markdown 的存储根目录
STORAGE_DIR = BASE_DIR / "storage"

# --------------------------------------------------------------------------- #
# MinIO 客户端配置
# --------------------------------------------------------------------------- #
# 服务端点，形如 "host:port"（不带协议前缀，由 MINIO_SECURE 决定 http/https）
MINIO_ENDPOINT = "localhost:9000"
# 访问密钥（用户名）
MINIO_ACCESS_KEY = "minioadmin"
# 秘密密钥（密码）
MINIO_SECRET_KEY = "minioadmin"
# 是否走 TLS：本地 docker-compose 未启用证书，默认关闭
MINIO_SECURE = False
# 默认存储桶；不存在时由插件在启动阶段自动创建
MINIO_BUCKET = "productrag"
# 拼接对外访问 URL 时使用的端点（形如 "host:port"，也可带协议前缀）；
# 留空则回退 MINIO_ENDPOINT。容器内走内网端点、浏览器需走外网地址时用得到
MINIO_PUBLIC_ENDPOINT = ""
# 是否把存储桶设为匿名只读（公开读）：开启后对象 URL 无需签名即可被外部
# 直接访问（仅放行 GetObject，不能列举、不能写入）；关闭则只有持有密钥的
# 客户端能访问，外部链接需改用预签名 URL
MINIO_PUBLIC_READ = True

READY = False
if PYPROJECT_FILE.is_file() and not READY:
    READY = True
    try:
        with PYPROJECT_FILE.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        pass

    tool_cfg: dict = data.get("tool") or {}

    # [tool.postgres]
    postgres_cfg: dict = tool_cfg.get("postgres") or {}
    if postgres_cfg:
        globals().update(
            POSTGRES_HOST=postgres_cfg.get("host", POSTGRES_HOST),
            POSTGRES_PORT=postgres_cfg.get("port", POSTGRES_PORT),
            POSTGRES_USER=postgres_cfg.get("user", POSTGRES_USER),
            POSTGRES_PASSWORD=postgres_cfg.get("password", POSTGRES_PASSWORD),
            POSTGRES_DATABASE=postgres_cfg.get("database", POSTGRES_DATABASE),
        )

    # [tool.redis]
    redis_cfg: dict = tool_cfg.get("redis") or {}
    if redis_cfg:
        globals().update(
            REDIS_HOST=redis_cfg.get("host", REDIS_HOST),
            REDIS_PORT=redis_cfg.get("port", REDIS_PORT),
            REDIS_DB=redis_cfg.get("db", REDIS_DB),
            REDIS_PASSWORD=redis_cfg.get("password", REDIS_PASSWORD),
        )

    # [tool.model.deepseek]
    model_cfg: dict = tool_cfg.get("model") or {}
    deepseek_cfg: dict = model_cfg.get("deepseek") or {}
    if deepseek_cfg:
        globals().update(
            DEEPSEEK_MODEL_URL=deepseek_cfg.get("url", DEEPSEEK_MODEL_URL),
            DEEPSEEK_MODEL_KEY=deepseek_cfg.get("key", DEEPSEEK_MODEL_KEY),
            DEEPSEEK_MODEL_NAME=deepseek_cfg.get("name", DEEPSEEK_MODEL_NAME),
        )

    # [tool.model.qwen-embedding]
    qwen_cfg: dict = model_cfg.get("qwen-embedding") or {}
    if qwen_cfg:
        globals().update(
            QWEN_EMBEDDING_URL=qwen_cfg.get("url", QWEN_EMBEDDING_URL),
            QWEN_EMBEDDING_KEY=qwen_cfg.get("key", QWEN_EMBEDDING_KEY),
            QWEN_EMBEDDING_NAME=qwen_cfg.get("name", QWEN_EMBEDDING_NAME),
            QWEN_EMBEDDING_DIMENSIONS=qwen_cfg.get(
                "dimensions", QWEN_EMBEDDING_DIMENSIONS
            ),
        )

    # [tool.model.qwen-rerank]
    rerank_cfg: dict = model_cfg.get("qwen-rerank") or {}
    if rerank_cfg:
        globals().update(
            RERANK_URL=rerank_cfg.get("url", RERANK_URL),
            RERANK_MODEL=rerank_cfg.get("model", RERANK_MODEL),
            RERANK_MODEL_KEY=rerank_cfg.get("key", RERANK_MODEL_KEY),
            RERANK_INSTRUCT_FIELD=rerank_cfg.get(
                "instruct_field", RERANK_INSTRUCT_FIELD
            ),
        )

    # [tool.celery]
    celery_cfg: dict = tool_cfg.get("celery") or {}
    if celery_cfg:
        globals().update(
            CELERY_BROKER_URL=celery_cfg.get("broker_url", CELERY_BROKER_URL),
            CELERY_RESULT_URL=celery_cfg.get("result_url", CELERY_RESULT_URL),
        )

    # [tool.tavily]
    tavily_cfg: dict = tool_cfg.get("tavily") or {}
    if tavily_cfg:
        globals().update(
            TAVILY_KEY=tavily_cfg.get("key", TAVILY_KEY),
            TAVILY_MAX_RESULTS=tavily_cfg.get("max_results", TAVILY_MAX_RESULTS),
            TAVILY_SEARCH_DEPTH=tavily_cfg.get("search_depth", TAVILY_SEARCH_DEPTH),
        )

    # [tool.mineru]
    mineru_cfg: dict = tool_cfg.get("mineru") or {}
    if mineru_cfg:
        globals().update(
            MINERU_TOKEN=mineru_cfg.get("token", MINERU_TOKEN),
            MINERU_BASE_URL=mineru_cfg.get("base_url", MINERU_BASE_URL),
        )

    # [tool.minio]
    minio_cfg: dict = tool_cfg.get("minio") or {}
    if minio_cfg:
        globals().update(
            MINIO_ENDPOINT=minio_cfg.get("endpoint", MINIO_ENDPOINT),
            MINIO_ACCESS_KEY=minio_cfg.get("access_key", MINIO_ACCESS_KEY),
            MINIO_SECRET_KEY=minio_cfg.get("secret_key", MINIO_SECRET_KEY),
            MINIO_SECURE=minio_cfg.get("secure", MINIO_SECURE),
            MINIO_BUCKET=minio_cfg.get("bucket", MINIO_BUCKET),
            MINIO_PUBLIC_ENDPOINT=minio_cfg.get(
                "public_endpoint", MINIO_PUBLIC_ENDPOINT
            ),
            MINIO_PUBLIC_READ=minio_cfg.get("public_read", MINIO_PUBLIC_READ),
        )

    # [tool.milvus]
    milvus_cfg: dict = tool_cfg.get("milvus") or {}
    if milvus_cfg:
        globals().update(
            MILVUS_ENDPOINT=milvus_cfg.get("endpoint", MILVUS_ENDPOINT),
            MILVUS_SECURE=milvus_cfg.get("secure", MILVUS_SECURE),
            MILVUS_DATABASE=milvus_cfg.get("database", MILVUS_DATABASE),
            MILVUS_TOKEN=milvus_cfg.get("token", MILVUS_TOKEN),
        )

    # 就地刷新 Tortoise ORM 配置字典，使其反映本次加载的 PostgreSQL 配置
    TORTOISE_ORM["connections"][TORTOISE_CONNECTION]["credentials"].update(
        {
            "host": POSTGRES_HOST,
            "port": POSTGRES_PORT,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "database": POSTGRES_DATABASE,
            "minsize": POSTGRES_POOL_MIN_SIZE,
            "maxsize": POSTGRES_POOL_MAX_SIZE,
        }
    )
