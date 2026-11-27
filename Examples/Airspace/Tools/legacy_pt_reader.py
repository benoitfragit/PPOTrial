#!/usr/bin/env python3
"""Pure-Python reader of the legacy LibTorch Airspace archives (RLMission, RLSNA model.pt).

No torch needed: data.pkl is unpickled with stand-ins for the torch classes and the raw
little-endian storages are decoded with numpy (FloatStorage = float32, LongStorage = int64).
The Adam state of torch::optim::Adam::save() is resolved in optimizer parameter order.

Archive layout (torch::serialize::OutputArchive, written by legacy ppo.cpp PPO::Save:
`policy->save(archive); optimizer->save(archive); archive.save_to(path)`):

    <root>/data.pkl            pickled module tree (protocol 2)
    <root>/data/<key>          one raw storage per tensor
    <root>/code/__torch__/...  TorchScript stubs (types only)

The pickled root holds:
    <module>.weight / .bias    one sub-module per registered torch::nn::Linear, in
                               register_module order (= policy->parameters() order)
    pytorch_version            str
    state                      {"<TensorImpl address>": {step:int, exp_avg, exp_avg_sq}}
    param_groups               {"param_groups/size": LongTensor[] (scalar),
                                "param_groups/0": {"params/size": LongTensor scalar,
                                                   "params/<i>": "<TensorImpl address>",
                                                   "options": {lr, betas, eps,
                                                               weight_decay, amsgrad}}}

The state keys are the addresses of the parameters' TensorImpl at save time; the
group's params/<i> list gives them in optimizer parameter order, which is
policy->parameters() order (legacy ppo.cpp builds Adam with m_policy->parameters()).

Usage: python3 -I legacy_pt_reader.py <model.pt>   (prints a summary)
"""

import collections
import pickle
import sys
import zipfile

import numpy as np

_DTYPES = {"FloatStorage": np.dtype("<f4"), "LongStorage": np.dtype("<i8"),
           "DoubleStorage": np.dtype("<f8"), "IntStorage": np.dtype("<i4")}


class StorageType:
    """Marker for a torch.<X>Storage class named in a persistent id."""

    def __init__(self, name):
        self.name = name


class Module:
    """Stand-in for the __torch__ module classes: the pickled dict is kept as `state`."""

    def __init__(self, *args):
        self.module_type = None
        self.state = None

    def __setstate__(self, state):
        self.state = state


class Tensor:
    """A strided view of a decoded storage, materialized as a numpy array."""

    def __init__(self, storage, offset, size, stride, requires_grad):
        self.size = tuple(size)
        self.stride = tuple(stride)
        self.offset = offset
        self.requires_grad = bool(requires_grad)
        self.storage_key, self.dtype, flat = storage
        if self.size:
            itemsize = flat.dtype.itemsize
            view = np.lib.stride_tricks.as_strided(
                flat[offset:], shape=self.size, strides=[s * itemsize for s in self.stride])
            self.array = np.array(view, copy=True)
        else:
            self.array = np.array(flat[offset], copy=True)

    def __repr__(self):
        return "Tensor%s<%s>" % (self.size, self.array.dtype)


def _rebuild_tensor(storage, offset, size, stride, requires_grad=False, hooks=None, *rest):
    return Tensor(storage, offset, size, stride, requires_grad)


def load(path):
    """Returns the unpickled root (a Module whose .state is a dict)."""
    archive = zipfile.ZipFile(path)
    pickles = [n for n in archive.namelist() if n.endswith("/data.pkl")]
    if len(pickles) != 1:
        raise ValueError("%s: expected one data.pkl, found %d" % (path, len(pickles)))
    prefix = pickles[0][:-len("data.pkl")]
    if prefix + "byteorder" in archive.namelist() and archive.read(prefix + "byteorder").decode() != "little":
        raise ValueError("big-endian archive")

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return _rebuild_tensor
            if module == "torch" and name in _DTYPES:
                return StorageType(name)
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if module.startswith("__torch__"):
                return Module
            raise pickle.UnpicklingError("unexpected class %s.%s" % (module, name))

        def persistent_load(self, pid):
            kind, storage_type, key, location, count = pid
            if kind != "storage":
                raise pickle.UnpicklingError("unexpected persistent id %r" % (pid,))
            dtype = _DTYPES[storage_type.name]
            blob = archive.read(prefix + "data/" + key)
            if len(blob) != dtype.itemsize * count:
                raise pickle.UnpicklingError("storage %s: %d bytes for %d values" % (key, len(blob), count))
            return (key, storage_type.name, np.frombuffer(blob, dtype=dtype))

    with archive.open(pickles[0]) as stream:
        return Unpickler(stream).load()


def parameters(root):
    """[(name, Tensor)] of every Linear weight/bias, in register_module order."""
    out = []
    for module_name, value in root.state.items():
        if isinstance(value, Module) and isinstance(value.state, dict) and "weight" in value.state:
            for kind in ("weight", "bias"):
                out.append(("%s.%s" % (module_name, kind), value.state[kind]))
    return out


def adam(root):
    """Adam options and per-parameter state, in optimizer (params/<i>) order."""
    groups = root.state["param_groups"].state
    group_count = int(groups["param_groups/size"].array)
    result = {"group_count": group_count, "groups": []}
    state = root.state["state"].state
    for g in range(group_count):
        group = groups["param_groups/%d" % g].state
        count = int(group["params/size"].array)
        keys = [group["params/%d" % i] for i in range(count)]
        options = dict(group["options"].state)
        entries = []
        for key in keys:
            entry = state[key].state
            entries.append({"key": key, "step": int(entry["step"]),
                            "exp_avg": entry["exp_avg"], "exp_avg_sq": entry["exp_avg_sq"],
                            "other_fields": sorted(set(entry) - {"step", "exp_avg", "exp_avg_sq"})})
        result["groups"].append({"options": options, "params": entries})
    result["state_keys_in_pickle_order"] = list(state.keys())
    return result


if __name__ == "__main__":
    root = load(sys.argv[1])
    print("pytorch_version:", root.state.get("pytorch_version"))
    for name, tensor in parameters(root):
        print(name, tensor.size, tensor.array.dtype)
    optimizer = adam(root)
    for group in optimizer["groups"]:
        print("options:", group["options"])
        print("steps:", sorted(set(p["step"] for p in group["params"])))
