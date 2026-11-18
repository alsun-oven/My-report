"""Python 文件配置，支持 `_base_` 继承和 `_delete_` 覆盖。"""
import copy
import inspect
import os
import runpy
from typing import Any, Dict


class ConfigDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def __deepcopy__(self, memo):
        return ConfigDict({k: copy.deepcopy(v, memo) for k, v in self.items()})


def _to_config_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_config_dict(v) for v in obj)
    return obj


def merge_dict(base: Dict, new: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and v.get("_delete_", False):
            out[k] = {kk: vv for kk, vv in v.items() if kk != "_delete_"}
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    @staticmethod
    def _load(path: str) -> Dict:
        path = os.path.abspath(path)
        ns = runpy.run_path(path)
        cfg = {
            k: v for k, v in ns.items()
            if not k.startswith("__") and k != "_base_"
            and not inspect.ismodule(v) and not inspect.isfunction(v) and not inspect.isclass(v)
        }
        bases = ns.get("_base_", [])
        if isinstance(bases, str):
            bases = [bases]
        merged: Dict = {}
        for b in bases:
            merged = merge_dict(merged, Config._load(os.path.join(os.path.dirname(path), b)))
        return merge_dict(merged, cfg)

    @staticmethod
    def fromfile(path: str) -> ConfigDict:
        return _to_config_dict(Config._load(path))
