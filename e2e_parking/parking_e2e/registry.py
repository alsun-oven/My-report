"""自实现的注册器：配置里写 dict(type="ClassName", ...)，按名字查表构建。

层级规则：
- 子注册表挂在父注册表下，`MODELS.build` 能构建任意子表（HEADS、BACKBONES ...）里的类。
- 子表查不到时回退到父表。
- 同名类出现在两个子表里时，用 "heads.SDFHead" 这样的前缀消歧。
"""
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Union


class Registry:
    def __init__(self, name: str, parent: Optional["Registry"] = None):
        self.name = name
        self.parent = parent
        self.children: Dict[str, "Registry"] = {}
        self._module_dict: Dict[str, Callable] = {}
        if parent is not None:
            if name in parent.children:
                raise KeyError(f"registry {name} already exists under {parent.name}")
            parent.children[name] = self

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        return f"Registry(name={self.name}, items={sorted(self._module_dict)})"

    @property
    def module_dict(self) -> Dict[str, Callable]:
        return dict(self._module_dict)

    def all_names(self) -> List[str]:
        names = list(self._module_dict)
        for child in self.children.values():
            names += [f"{child.name}.{n}" for n in child.all_names()]
        return names

    # ------------------------------------------------------------------ register
    def _register(self, obj: Callable, name: Union[None, str, Iterable[str]], force: bool) -> None:
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            raise TypeError(f"only class or function can be registered, got {type(obj)}")
        if name is None:
            keys = [obj.__name__]
        elif isinstance(name, str):
            keys = [name]
        else:
            keys = list(name)
        for key in keys:
            if not force and key in self._module_dict:
                old = self._module_dict[key]
                raise KeyError(
                    f"{key} is already registered in {self.name} by {old.__module__}; "
                    f"new one from {obj.__module__}")
            self._module_dict[key] = obj

    def register_module(self, name: Union[None, str, Iterable[str]] = None,
                        force: bool = False, module: Optional[Callable] = None):
        """装饰器用法 `@REG.register_module()`，或函数用法 `REG.register_module(module=cls)`。"""
        if module is not None:
            self._register(module, name, force)
            return module

        def _decorator(obj):
            self._register(obj, name, force)
            return obj

        return _decorator

    # ------------------------------------------------------------------ lookup
    def _find_child(self, name: str) -> Optional["Registry"]:
        if name == self.name:
            return self
        for child in self.children.values():
            found = child._find_child(name)
            if found is not None:
                return found
        return None

    def _search_down(self, key: str) -> List[Callable]:
        hits = [self._module_dict[key]] if key in self._module_dict else []
        for child in self.children.values():
            hits += child._search_down(key)
        return hits

    def _root(self) -> "Registry":
        reg = self
        while reg.parent is not None:
            reg = reg.parent
        return reg

    def get(self, key: str) -> Optional[Callable]:
        if not isinstance(key, str):
            return None
        if "." in key:
            scope, cls_name = key.split(".", 1)
            reg = self._root()._find_child(scope)
            return None if reg is None else reg._module_dict.get(cls_name)

        hits = self._search_down(key)
        if len(hits) > 1 and any(h is not hits[0] for h in hits):
            raise KeyError(f"{key} is ambiguous under {self.name}; use 'scope.{key}'")
        if hits:
            return hits[0]
        if self.parent is not None:
            return self.parent.get(key)
        return None

    # ------------------------------------------------------------------ build
    def build(self, cfg: Optional[Dict[str, Any]], **default_args):
        return build_from_cfg(cfg, self, default_args)


def build_from_cfg(cfg: Optional[Dict[str, Any]], registry: Registry,
                   default_args: Optional[Dict[str, Any]] = None):
    if cfg is None:
        return None
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise TypeError(f"cfg must be a dict with key 'type', got {cfg}")
    args = dict(cfg)
    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f"{obj_type} is not in the {registry.name} registry. Make sure the file defining "
                f"it is imported. Registered: {sorted(registry.all_names())}")
    elif callable(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be str or callable, got {type(obj_type)}")

    for k, v in (default_args or {}).items():
        args.setdefault(k, v)
    try:
        return obj_cls(**args)
    except TypeError as e:
        raise TypeError(f"{obj_cls.__name__}({', '.join(args)}): {e}") from e


# ---------------------------------------------------------------------- registries
MODELS = Registry("models")
BACKBONES = Registry("backbones", parent=MODELS)
NECKS = Registry("necks", parent=MODELS)
VIEW_TRANSFORMS = Registry("view_transforms", parent=MODELS)
SENSOR_ENCODERS = Registry("sensor_encoders", parent=MODELS)   # lidar / radar / navigation
FUSERS = Registry("fusers", parent=MODELS)                     # 多模态融合、时序对齐
BEV_ENCODERS = Registry("bev_encoders", parent=MODELS)         # BEV 与 GOD 体素编码
HEADS = Registry("heads", parent=MODELS)
LOSSES = Registry("losses", parent=MODELS)

MATCHERS = Registry("matchers")
CAMERA_MODELS = Registry("camera_models")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
