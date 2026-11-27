#!/usr/bin/env python3
"""Converts a legacy LibTorch red policy (model.pt) into a PPOKit legacy policy text file.

The legacy models are torch::serialize archives: a zip with data.pkl (the module tree,
pickled) and one raw little-endian float32 blob per tensor. This reader needs neither
torch nor numpy: it unpickles the tree with stand-ins for the torch classes and keeps
the network parameters (weight and bias of every Linear). The Adam state is dropped.

Network parameters must be float32. Output, one parameter after the other, values in row-major order with 9 significant
digits (exact round trip of float32):

    ppokit.legacy_policy/1
    architecture <name>
    parameter <module>.<weight|bias> <rows> <columns>
    <values...>
    end

Usage: convert_legacy_model.py <model.pt> <architecture> <output>
       architecture: mission_track_attention | interception_mlp
"""

import pickle
import struct
import sys
import zipfile

# Modules of each architecture, in the order the C++ policies read them.
ARCHITECTURES = {
    "mission_track_attention": [
        "enc1", "enc2", "attn_q", "attn_k", "mu_head", "logstd_head", "fire_head", "critic1", "critic2"],
    "interception_mlp": ["fc1", "fc2", "mu_head", "logstd_head", "critic"],
}


class Storage:
    """Raw float32 storage of one tensor blob."""

    def __init__(self, values):
        self.values = values


class Tensor:
    """A strided float32 view of a storage."""

    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)

    def rows(self):
        """Values as a list of rows (a vector is one row)."""
        if len(self.size) == 1:
            return [[self.storage.values[self.offset + i * self.stride[0]] for i in range(self.size[0])]]
        if len(self.size) != 2:
            raise ValueError("only vectors and matrices are expected, got shape %s" % (self.size,))
        return [[self.storage.values[self.offset + r * self.stride[0] + c * self.stride[1]]
                 for c in range(self.size[1])] for r in range(self.size[0])]


class Module:
    """Stand-in for the __torch__ module classes: just their attributes."""

    def __setstate__(self, state):
        self.__dict__.update(state)


class FloatStorageType:
    """Marker for torch.FloatStorage in persistent ids."""


class LongStorageType:
    """Marker for torch.LongStorage (the optimizer's bookkeeping, not read)."""


# Element format of each storage type.
STORAGE_FORMATS = {FloatStorageType: ("f", 4), LongStorageType: ("q", 8)}


def rebuild_tensor(storage, offset, size, stride, requires_grad=False, hooks=None, *rest):
    return Tensor(storage, offset, size, stride)


class LegacyUnpickler(pickle.Unpickler):
    def __init__(self, archive, prefix, stream):
        super().__init__(stream)
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor
        if module == "torch" and name == "FloatStorage":
            return FloatStorageType
        if module == "torch" and name == "LongStorage":
            return LongStorageType
        if module == "collections" and name == "OrderedDict":
            return dict
        if module.startswith("__torch__"):
            return Module
        raise pickle.UnpicklingError("unexpected class %s.%s in a legacy model" % (module, name))

    def persistent_load(self, pid):
        kind, storage_type, key, location, count = pid
        if kind != "storage" or storage_type not in STORAGE_FORMATS:
            raise pickle.UnpicklingError("only float32 and int64 storages are expected, got %r" % (pid,))
        element, width = STORAGE_FORMATS[storage_type]
        blob = self.archive.read(self.prefix + "data/" + key)
        if len(blob) != width * count:
            raise pickle.UnpicklingError("storage %s holds %d bytes for %d values" % (key, len(blob), count))
        return Storage(struct.unpack("<%d%s" % (count, element), blob))


def load_modules(path):
    with zipfile.ZipFile(path) as archive:
        pickles = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise ValueError("%s: expected one data.pkl, found %d" % (path, len(pickles)))
        prefix = pickles[0][:-len("data.pkl")]
        byteorder = archive.read(prefix + "byteorder").decode() if prefix + "byteorder" in archive.namelist() else "little"
        if byteorder != "little":
            raise ValueError("%s: big-endian tensors are not supported" % path)
        with archive.open(pickles[0]) as stream:
            return LegacyUnpickler(archive, prefix, stream).load()


def write_policy(root, architecture, output):
    lines = ["ppokit.legacy_policy/1", "architecture " + architecture]
    for module in ARCHITECTURES[architecture]:
        layer = getattr(root, module)
        for kind in ("weight", "bias"):
            tensor = getattr(layer, kind)
            rows = tensor.rows()
            columns = len(rows[0])
            lines.append("parameter %s.%s %d %d" % (module, kind, len(rows), columns))
            for row in rows:
                lines.append(" ".join("%.9g" % value for value in row))
    lines.append("end")
    with open(output, "w", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")


def main(arguments):
    if len(arguments) != 3 or arguments[1] not in ARCHITECTURES:
        sys.stderr.write(__doc__)
        return 2
    write_policy(load_modules(arguments[0]), arguments[1], arguments[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
