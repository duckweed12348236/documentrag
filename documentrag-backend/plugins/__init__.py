from abc import ABCMeta, abstractmethod, ABC
from collections.abc import Collection

from metaclasses import SingletonMeta


class Singleton(metaclass=SingletonMeta):
    pass


plugin_classes = []


class PluginMeta(SingletonMeta, ABCMeta):
    def __new__(cls, name, bases, attrs):
        sub_cls = super().__new__(cls, name, bases, attrs)
        if name != "Plugin":
            plugin_classes.append(sub_cls)
        return sub_cls


class Plugin(ABC, metaclass=PluginMeta):
    @abstractmethod
    async def register(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


plugins = []


async def register_plugins(*, exclude: Collection[type[Plugin]] = ()) -> None:
    """注册全部已导入的插件。

    Args:
        exclude: 需要跳过的插件类——Celery worker 子进程在启动信号里复用同一套
            插件时用它排除 ``CeleryWorker``，避免子进程再拉起一个 worker。
    """
    for plugin_cls in plugin_classes:
        if plugin_cls in exclude:
            continue
        plugin = plugin_cls()
        await plugin.register()
        plugins.append(plugin)


async def close_plugins() -> None:
    """按注册顺序注销插件（与 :func:`register_plugins` 一一对应）。"""
    for plugin in plugins:
        await plugin.close()
