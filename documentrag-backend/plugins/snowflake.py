# Twitter's Snowflake algorithm implementation which is used to generate distributed IDs.
# https://github.com/twitter-archive/snowflake/blob/snowflake-2010/src/main/scala/com/twitter/service/snowflake/IdWorker.scala

import time

from config import SNOWFLAKE_WORKER_ID, SNOWFLAKE_DATACENTER_ID, SNOWFLAKE_SEQUENCE
from plugins import Singleton

# 64位ID的划分
WORKER_ID_BITS = 5
DATACENTER_ID_BITS = 5
SEQUENCE_BITS = 12

# 最大取值计算
MAX_WORKER_ID = -1 ^ (-1 << WORKER_ID_BITS)  # 2**5-1 0b11111
MAX_DATACENTER_ID = -1 ^ (-1 << DATACENTER_ID_BITS)

# 移位偏移计算
WOKER_ID_SHIFT = SEQUENCE_BITS
DATACENTER_ID_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS
TIMESTAMP_LEFT_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS + DATACENTER_ID_BITS

# 序号循环掩码
SEQUENCE_MASK = -1 ^ (-1 << SEQUENCE_BITS)

# Twitter元年时间戳
TWEPOCH = 1288834974657


class InvalidSystemClock(Exception):
    """
    时钟回拨异常
    """
    pass


class Snowflake:
    """
    用于生成IDs（雪花算法，64 位整型）。

    机器标识、序号与上次时间戳都以类字段持有，生成入口 :meth:`get_id` 为类方法，
    直接 ``Snowflake.get_id()`` 调用即可，无需实例化。
    """

    #: 数据中心（机器区域）ID
    datacenter_id: int = SNOWFLAKE_DATACENTER_ID
    #: 机器 ID
    worker_id: int = SNOWFLAKE_WORKER_ID
    #: 当前毫秒内的序号
    sequence: int = SNOWFLAKE_SEQUENCE
    #: 上次生成 ID 的毫秒时间戳，-1 表示尚未生成
    last_timestamp: int = -1

    @staticmethod
    def _gen_timestamp() -> int:
        """
        生成整数时间戳
        :return:int timestamp
        """
        return int(time.time() * 1000)

    @classmethod
    def get_id(cls) -> int:
        """
        获取新ID
        :return:int id
        """
        timestamp = cls._gen_timestamp()

        # 时钟回拨
        if timestamp < cls.last_timestamp:
            raise InvalidSystemClock

        if timestamp == cls.last_timestamp:
            cls.sequence = (cls.sequence + 1) & SEQUENCE_MASK
            if cls.sequence == 0:
                timestamp = cls._til_next_millis(cls.last_timestamp)
        else:
            cls.sequence = 0

        cls.last_timestamp = timestamp

        new_id = ((timestamp - TWEPOCH) << TIMESTAMP_LEFT_SHIFT) | (cls.datacenter_id << DATACENTER_ID_SHIFT) | \
                 (cls.worker_id << WOKER_ID_SHIFT) | cls.sequence
        return new_id

    @classmethod
    def _til_next_millis(cls, last_timestamp: int) -> int:
        """
        等到下一毫秒
        :return:int timestamp
        """
        timestamp = cls._gen_timestamp()
        while timestamp <= last_timestamp:
            timestamp = cls._gen_timestamp()
        return timestamp
