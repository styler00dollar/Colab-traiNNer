"""
MAT: Mask-Aware Transformer for Large Hole Image Inpainting (CVPR2022)
https://github.com/fenglinglwb/MAT

MAT/networks/mat.py
MAT/networks/basic_module.py
MAT/torch_utils/misc.py
MAT/torch_utils/persistence.py
"""
import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .cpp2 import conv2d_resample
from .cpp2 import upfirdn2d
from .cpp2 import bias_act

import sys
from collections import OrderedDict
import numpy as np

import re
import contextlib
import numpy as np
import torch
import warnings
from .cpp2.util import EasyDict  # import dnnlib

from .cpp2 import misc as misc

import pytorch_lightning as pl

"""Facilities for pickling Python code alongside other data.

The pickled code is automatically imported into a separate Python module
during unpickling. This way, any previously exported pickles will remain
usable even if the original code is no longer available, or if the current
version of the code is not consistent with what was originally pickled."""

import sys
import pickle
import io
import inspect
import copy
import uuid
import types

# ----------------------------------------------------------------------------

_version = 6  # internal version number
_decorators = set()  # {decorator_class, ...}
_import_hooks = []  # [hook_function, ...]
_module_to_src_dict = dict()  # {module: src, ...}
_src_to_module_dict = dict()  # {src: module, ...}

# ----------------------------------------------------------------------------


def persistent_class(orig_class):
    r"""Class decorator that extends a given class to save its source code
    when pickled.

    Example:

        from torch_utils import persistence

        @persistent_class
        class MyNetwork(torch.nn.Module):
            def __init__(self, num_inputs, num_outputs):
                super().__init__()
                self.fc = MyLayer(num_inputs, num_outputs)
                ...

        @persistent_class
        class MyLayer(torch.nn.Module):
            ...

    When pickled, any instance of `MyNetwork` and `MyLayer` will save its
    source code alongside other internal state (e.g., parameters, buffers,
    and submodules). This way, any previously exported pickle will remain
    usable even if the class definitions have been modified or are no
    longer available.

    The decorator saves the source code of the entire Python module
    containing the decorated class. It does *not* save the source code of
    any imported modules. Thus, the imported modules must be available
    during unpickling, also including `torch_utils.persistence` itself.

    It is ok to call functions defined in the same module from the
    decorated class. However, if the decorated class depends on other
    classes defined in the same module, they must be decorated as well.
    This is illustrated in the above example in the case of `MyLayer`.

    It is also possible to employ the decorator just-in-time before
    calling the constructor. For example:

        cls = MyLayer
        if want_to_make_it_persistent:
            cls = persistence.persistent_class(cls)
        layer = cls(num_inputs, num_outputs)

    As an additional feature, the decorator also keeps track of the
    arguments that were used to construct each instance of the decorated
    class. The arguments can be queried via `obj.init_args` and
    `obj.init_kwargs`, and they are automatically pickled alongside other
    object state. A typical use case is to first unpickle a previous
    instance of a persistent class, and then upgrade it to use the latest
    version of the source code:

        with open('old_pickle.pkl', 'rb') as f:
            old_net = pickle.load(f)
        new_net = MyNetwork(*old_obj.init_args, **old_obj.init_kwargs)
        misc.copy_params_and_buffers(old_net, new_net, require_all=True)
    """
    assert isinstance(orig_class, type)
    if is_persistent(orig_class):
        return orig_class

    assert orig_class.__module__ in sys.modules
    orig_module = sys.modules[orig_class.__module__]
    orig_module_src = _module_to_src(orig_module)

    class Decorator(orig_class):
        _orig_module_src = orig_module_src
        _orig_class_name = orig_class.__name__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_args = copy.deepcopy(args)
            self._init_kwargs = copy.deepcopy(kwargs)
            assert orig_class.__name__ in orig_module.__dict__
            _check_pickleable(self.__reduce__())

        @property
        def init_args(self):
            return copy.deepcopy(self._init_args)

        @property
        def init_kwargs(self):
            return EasyDict(copy.deepcopy(self._init_kwargs))

        def __reduce__(self):
            fields = list(super().__reduce__())
            fields += [None] * max(3 - len(fields), 0)
            if fields[0] is not _reconstruct_persistent_obj:
                meta = dict(
                    type="class",
                    version=_version,
                    module_src=self._orig_module_src,
                    class_name=self._orig_class_name,
                    state=fields[2],
                )
                fields[0] = _reconstruct_persistent_obj  # reconstruct func
                fields[1] = (meta,)  # reconstruct args
                fields[2] = None  # state dict
            return tuple(fields)

    Decorator.__name__ = orig_class.__name__
    _decorators.add(Decorator)
    return Decorator


# ----------------------------------------------------------------------------


def is_persistent(obj):
    r"""Test whether the given object or class is persistent, i.e.,
    whether it will save its source code when pickled.
    """
    try:
        if obj in _decorators:
            return True
    except TypeError:
        pass
    return type(obj) in _decorators  # pylint: disable=unidiomatic-typecheck


# ----------------------------------------------------------------------------


def import_hook(hook):
    r"""Register an import hook that is called whenever a persistent object
    is being unpickled. A typical use case is to patch the pickled source
    code to avoid errors and inconsistencies when the API of some imported
    module has changed.

    The hook should have the following signature:

        hook(meta) -> modified meta

    `meta` is an instance of `EasyDict` with the following fields:

        type:       Type of the persistent object, e.g. `'class'`.
        version:    Internal version number of `torch_utils.persistence`.
        module_src  Original source code of the Python module.
        class_name: Class name in the original Python module.
        state:      Internal state of the object.

    Example:

        @persistence.import_hook
        def wreck_my_network(meta):
            if meta.class_name == 'MyNetwork':
                print('MyNetwork is being imported. I will wreck it!')
                meta.module_src = meta.module_src.replace("True", "False")
            return meta
    """
    assert callable(hook)
    _import_hooks.append(hook)


# ----------------------------------------------------------------------------


def _reconstruct_persistent_obj(meta):
    r"""Hook that is called internally by the `pickle` module to unpickle
    a persistent object.
    """
    meta = EasyDict(meta)
    meta.state = EasyDict(meta.state)
    for hook in _import_hooks:
        meta = hook(meta)
        assert meta is not None

    assert meta.version == _version
    module = _src_to_module(meta.module_src)

    assert meta.type == "class"
    orig_class = module.__dict__[meta.class_name]
    decorator_class = persistent_class(orig_class)
    obj = decorator_class.__new__(decorator_class)

    setstate = getattr(obj, "__setstate__", None)
    if callable(setstate):
        setstate(meta.state)  # pylint: disable=not-callable
    else:
        obj.__dict__.update(meta.state)
    return obj


# ----------------------------------------------------------------------------


def _module_to_src(module):
    r"""Query the source code of a given Python module."""
    src = _module_to_src_dict.get(module, None)
    if src is None:
        src = inspect.getsource(module)
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
    return src


def _src_to_module(src):
    r"""Get or create a Python module for the given source code."""
    module = _src_to_module_dict.get(src, None)
    if module is None:
        module_name = "_imported_module_" + uuid.uuid4().hex
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
        exec(src, module.__dict__)  # pylint: disable=exec-used
    return module


# ----------------------------------------------------------------------------


def _check_pickleable(obj):
    r"""Check that the given object is pickleable, raising an exception if
    it is not. This function is expected to be considerably more efficient
    than actually pickling the object.
    """

    def recurse(obj):
        if isinstance(obj, (list, tuple, set)):
            return [recurse(x) for x in obj]
        if isinstance(obj, dict):
            return [[recurse(x), recurse(y)] for x, y in obj.items()]
        if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
            return None  # Python primitive types are pickleable.
        if f"{type(obj).__module__}.{type(obj).__name__}" in [
            "numpy.ndarray",
            "torch.Tensor",
        ]:
            return None  # NumPy arrays and PyTorch tensors are pickleable.
        if is_persistent(obj):
            return None  # Persistent objects are pickleable, by virtue of the constructor check.
        return obj

    with io.BytesIO() as f:
        pickle.dump(recurse(obj), f)


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (
        value.shape,
        value.dtype,
        value.tobytes(),
        shape,
        dtype,
        device,
        memory_format,
    )
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


# ----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num  # 1.8.0a0
except AttributeError:

    def nan_to_num(
        input, nan=0.0, posinf=None, neginf=None, *, out=None
    ):  # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(
            input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out
        )


# ----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0

# ----------------------------------------------------------------------------
# Context manager to suppress known warnings in torch.jit.trace().


class suppress_tracer_warnings(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        return self


# ----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().


def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(
            f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}"
        )
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(torch.as_tensor(size), ref_size),
                    f"Wrong size for dimension {idx}",
                )
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(size, torch.as_tensor(ref_size)),
                    f"Wrong size for dimension {idx}: expected {ref_size}",
                )
        elif size != ref_size:
            raise AssertionError(
                f"Wrong size for dimension {idx}: got {size}, expected {ref_size}"
            )


# ----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().


def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)

    decorator.__name__ = fn.__name__
    return decorator


# ----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(
        self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5
    ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


# ----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {
        name: tensor for name, tensor in named_params_and_buffers(src_module)
    }
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(
                tensor.requires_grad
            )


# ----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.


@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield


# ----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.


def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + "." + name
        flag = False
        if ignore_regex is not None:
            for regex in ignore_regex:
                if re.fullmatch(regex, fullname):
                    flag = True
                    break
        if flag:
            continue
        tensor = tensor.detach()
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (nan_to_num(tensor) == nan_to_num(other)).all(), fullname


# ----------------------------------------------------------------------------
# Print summary table of module hierarchy.


def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {
            id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs
        }

    # Filter out redundant entries.
    if skip_redundant:
        entries = [
            e
            for e in entries
            if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)
        ]

    # Construct table.
    rows = [
        [type(module).__name__, "Parameters", "Buffers", "Output shape", "Datatype"]
    ]
    rows += [["---"] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = "<top-level>" if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split(".")[-1] for t in e.outputs]
        rows += [
            [
                name + (":0" if len(e.outputs) >= 2 else ""),
                str(param_size) if param_size else "-",
                str(buffer_size) if buffer_size else "-",
                (output_shapes + ["-"])[0],
                (output_dtypes + ["-"])[0],
            ]
        ]
        for idx in range(1, len(e.outputs)):
            rows += [
                [name + f":{idx}", "-", "-", output_shapes[idx], output_dtypes[idx]]
            ]
        param_total += param_size
        buffer_total += buffer_size
    rows += [["---"] * len(rows[0])]
    rows += [["Total", str(param_total), str(buffer_total), "-", "-"]]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print(
            "  ".join(
                cell + " " * (width - len(cell)) for cell, width in zip(row, widths)
            )
        )
    print()
    return outputs


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------


@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# ----------------------------------------------------------------------------


@persistent_class
class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=1,  # Learning rate multiplier.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.activation = activation

        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            # out = torch.addmm(b.unsqueeze(0), x, w.t())
            x = x.matmul(w.t())
            out = x + b.reshape([-1 if i == x.ndim - 1 else 1 for i in range(x.ndim)])
        else:
            x = x.matmul(w.t())
            out = bias_act.bias_act(x, b, act=self.activation, dim=x.ndim - 1)
        return out


# ----------------------------------------------------------------------------


@persistent_class
class Conv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
        trainable=True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act.bias_act(
            x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp
        )
        return out


# ----------------------------------------------------------------------------


@persistent_class
class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        style_dim,  # dimension of the style code
        demodulate=True,  # perfrom demodulation
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
    ):
        super().__init__()
        self.demodulate = demodulate

        self.weight = torch.nn.Parameter(
            torch.randn([1, out_channels, in_channels, kernel_size, kernel_size])
        )
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.padding = self.kernel_size // 2
        self.up = up
        self.down = down
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

        self.affine = FullyConnectedLayer(style_dim, in_channels, bias_init=1)

    def forward(self, x, style):
        batch, in_channels, height, width = x.shape
        style = self.affine(style).view(batch, 1, in_channels, 1, 1)
        weight = self.weight * self.weight_gain * style

        if self.demodulate:
            decoefs = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * decoefs.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )
        x = x.view(1, batch * in_channels, height, width)
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=weight,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            groups=batch,
        )
        out = x.view(batch, self.out_channels, *x.shape[2:])

        return out


# ----------------------------------------------------------------------------


@persistent_class
class StyleConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        style_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this layer.
        kernel_size=3,  # Convolution kernel size.
        up=1,  # Integer upsampling factor.
        use_noise=True,  # Enable noise input?
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        demodulate=True,  # perform demodulation
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            up=up,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
        )

        self.use_noise = use_noise
        self.resolution = resolution
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.conv_clamp = conv_clamp

    def forward(self, x, style, noise_mode="random", gain=1):
        x = self.conv(x, style)

        assert noise_mode in ["random", "const", "none"]

        if self.use_noise:
            if noise_mode == "random":
                xh, xw = x.size()[-2:]
                noise = (
                    torch.randn([x.shape[0], 1, xh, xw], device=x.device)
                    * self.noise_strength
                )
            if noise_mode == "const":
                noise = self.noise_const * self.noise_strength
            x = x + noise

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act.bias_act(
            x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp
        )

        return out


# ----------------------------------------------------------------------------


@persistent_class
class ToRGB(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim,
        kernel_size=1,
        resample_filter=[1, 3, 3, 1],
        conv_clamp=None,
        demodulate=False,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

    def forward(self, x, style, skip=None):
        x = self.conv(x, style)
        out = bias_act.bias_act(x, self.bias, clamp=self.conv_clamp)

        if skip is not None:
            if skip.shape != out.shape:
                skip = upfirdn2d.upsample2d(skip, self.resample_filter)
            out = out + skip

        return out


# ----------------------------------------------------------------------------


@misc.profiled_function
def get_style_code(a, b):
    return torch.cat([a, b], dim=1)


# ----------------------------------------------------------------------------


@persistent_class
class DecBlockFirst(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):
        super().__init__()
        self.fc = FullyConnectedLayer(
            in_features=in_channels * 2,
            out_features=in_channels * 4**2,
            activation=activation,
        )
        self.conv = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=4,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, ws, gs, E_features, noise_mode="random"):
        x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = x + E_features[2]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


@persistent_class
class DecBlockFirstV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=4,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, ws, gs, E_features, noise_mode="random"):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[2]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


# ----------------------------------------------------------------------------


@persistent_class
class DecBlock(nn.Module):
    def __init__(
        self,
        res,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):  # res = 2, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, img, ws, gs, E_features, noise_mode="random"):
        style = get_style_code(ws[:, self.res * 2 - 5], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 4], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 3], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img


# ----------------------------------------------------------------------------


@persistent_class
class MappingNet(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        num_ws,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,  # Number of mapping layers.
        embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = (
            [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        )

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False
    ):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function("input"):
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(
                    x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
                )

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(
                        x[:, :truncation_cutoff], truncation_psi
                    )

        return x


# ----------------------------------------------------------------------------


@persistent_class
class DisFromRGB(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation
    ):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )

    def forward(self, x):
        return self.conv(x)


# ----------------------------------------------------------------------------


@persistent_class
class DisBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation
    ):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            down=2,
            activation=activation,
        )
        self.skip = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            down=2,
            bias=False,
        )

    def forward(self, x):
        skip = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        out = skip + x

        return out


# ----------------------------------------------------------------------------


@persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = (
                torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
                if self.group_size is not None
                else N
            )
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


# ----------------------------------------------------------------------------


@persistent_class
class Discriminator(torch.nn.Module):
    def __init__(
        self,
        c_dim,  # Conditioning label (C) dimensionality.
        img_resolution,  # Input resolution.
        img_channels,  # Number of input color channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        channel_decay=1,
        cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
        activation="lrelu",
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4
        self.resolution_log2 = resolution_log2

        def nf(stage):
            return np.clip(
                int(channel_base / 2 ** (stage * channel_decay)), 1, channel_max
            )

        if cmap_dim == None:
            cmap_dim = nf(2)
        if c_dim == 0:
            cmap_dim = 0
        self.cmap_dim = cmap_dim

        if c_dim > 0:
            self.mapping = MappingNet(
                z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None
            )

        Dis = [DisFromRGB(img_channels + 1, nf(resolution_log2), activation)]
        for res in range(resolution_log2, 2, -1):
            Dis.append(DisBlock(nf(res), nf(res - 1), activation))

        if mbstd_num_channels > 0:
            Dis.append(
                MinibatchStdLayer(
                    group_size=mbstd_group_size, num_channels=mbstd_num_channels
                )
            )
        Dis.append(
            Conv2dLayer(
                nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation
            )
        )
        self.Dis = nn.Sequential(*Dis)

        self.fc0 = FullyConnectedLayer(nf(2) * 4**2, nf(2), activation=activation)
        self.fc1 = FullyConnectedLayer(nf(2), 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, images_in, masks_in, c):
        x = torch.cat([masks_in - 0.5, images_in], dim=1)
        x = self.Dis(x)
        x = self.fc1(self.fc0(x.flatten(start_dim=1)))

        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return x


@misc.profiled_function
def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2**stage]


@persistent_class
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(
            in_features=in_features, out_features=hidden_features, activation="lrelu"
        )
        self.fc2 = FullyConnectedLayer(
            in_features=hidden_features, out_features=out_features
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


@misc.profiled_function
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


@misc.profiled_function
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


@persistent_class
class Conv2dLayerPartial(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
        trainable=True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.conv = Conv2dLayer(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            activation,
            up,
            down,
            resample_filter,
            conv_clamp,
            trainable,
        )

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size**2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

    def forward(self, x, mask=None):
        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                )
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)  # 0 or 1
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            return x, None


@persistent_class
class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        down_ratio=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.k = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.v = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.proj = FullyConnectedLayer(in_features=dim, out_features=dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask_windows=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        norm_x = F.normalize(x, p=2.0, dim=-1)
        q = (
            self.q(norm_x)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(norm_x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.v(x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        if mask_windows is not None:
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attn = attn + attn_mask_windows.masked_fill(
                attn_mask_windows == 0, float(-100.0)
            ).masked_fill(attn_mask_windows == 1, float(0.0))
            with torch.no_grad():
                mask_windows = torch.clamp(
                    torch.sum(mask_windows, dim=1, keepdim=True), 0, 1
                ).repeat(1, N, 1)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x, mask_windows


@persistent_class
class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        down_ratio=1,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        if self.shift_size > 0:
            down_ratio = 1
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            down_ratio=down_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.fuse = FullyConnectedLayer(
            in_features=dim * 2, out_features=dim, activation="lrelu"
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(self, x, x_size, mask=None):
        # H, W = self.input_resolution
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)
        if mask is not None:
            mask = mask.view(B, H, W, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            if mask is not None:
                shifted_mask = torch.roll(
                    mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
        else:
            shifted_x = x
            if mask is not None:
                shifted_mask = mask

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        if mask is not None:
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, mask_windows = self.attn(
                x_windows, mask_windows, mask=self.attn_mask
            )  # nW*B, window_size*window_size, C
        else:
            attn_windows, mask_windows = self.attn(
                x_windows, mask_windows, mask=self.calculate_mask(x_size).to(x.device)
            )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
            if mask is not None:
                mask = torch.roll(
                    shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
        else:
            x = shifted_x
            if mask is not None:
                mask = shifted_mask
        x = x.view(B, H * W, C)
        if mask is not None:
            mask = mask.view(B, H * W, 1)

        # FFN
        x = self.fuse(torch.cat([shortcut, x], dim=-1))
        x = self.mlp(x)

        return x, mask


@persistent_class
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation="lrelu",
            down=down,
        )
        self.down = down

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistent_class
class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation="lrelu",
            up=up,
        )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistent_class
class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        down_ratio=1,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample = downsample
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    down_ratio=down_ratio,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.conv = Conv2dLayerPartial(
            in_channels=dim, out_channels=dim, kernel_size=3, activation="lrelu"
        )

    def forward(self, x, x_size, mask=None):
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)
        identity = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x, mask = checkpoint.checkpoint(blk, x, x_size, mask)
            else:
                x, mask = blk(x, x_size, mask)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistent_class
class ToToken(nn.Module):
    def __init__(self, in_channels=3, dim=128, kernel_size=5, stride=1):
        super().__init__()

        self.proj = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=kernel_size,
            activation="lrelu",
        )

    def forward(self, x, mask):
        x, mask = self.proj(x, mask)

        return x, mask


# ----------------------------------------------------------------------------


@persistent_class
class EncFromRGB(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation
    ):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


@persistent_class
class ConvBlockDown(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation
    ):  # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


def token2feature(x, x_size):
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x


def feature2token(x):
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x


@persistent_class
class Encoder(nn.Module):
    def __init__(
        self,
        res_log2,
        img_channels,
        activation,
        patch_size=5,
        channels=16,
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.resolution = []

        for idx, i in enumerate(range(res_log2, 3, -1)):  # from input size to 16x16
            res = 2**i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i), activation)
            else:
                block = ConvBlockDown(nf(i + 1), nf(i), activation)
            setattr(self, "EncConv_Block_%dx%d" % (res, res), block)

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, "EncConv_Block_%dx%d" % (res, res))(x)
            out[res_log2] = x

        return out


@persistent_class
class ToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, activation, drop_rate):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnectedLayer(
            in_features=in_channels, out_features=out_channels, activation=activation
        )
        # self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))
        # x = self.dropout(x)

        return x


@persistent_class
class DecBlockFirstV2(nn.Module):
    def __init__(
        self,
        res,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, ws, gs, E_features, noise_mode="random"):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


# ----------------------------------------------------------------------------


@persistent_class
class DecBlock(nn.Module):
    def __init__(
        self,
        res,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):  # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, img, ws, gs, E_features, noise_mode="random"):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img


@persistent_class
class Decoder(nn.Module):
    def __init__(
        self, res_log2, activation, style_dim, use_noise, demodulate, img_channels
    ):
        super().__init__()
        self.Dec_16x16 = DecBlockFirstV2(
            4, nf(4), nf(4), activation, style_dim, use_noise, demodulate, img_channels
        )
        for res in range(5, res_log2 + 1):
            setattr(
                self,
                "Dec_%dx%d" % (2**res, 2**res),
                DecBlock(
                    res,
                    nf(res - 1),
                    nf(res),
                    activation,
                    style_dim,
                    use_noise,
                    demodulate,
                    img_channels,
                ),
            )
        self.res_log2 = res_log2

    def forward(self, x, ws, gs, E_features, noise_mode="random"):
        x, img = self.Dec_16x16(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(5, self.res_log2 + 1):
            block = getattr(self, "Dec_%dx%d" % (2**res, 2**res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        return img


@persistent_class
class DecStyleBlock(nn.Module):
    def __init__(
        self,
        res,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, img, style, skip, noise_mode="random"):
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + skip
        x = self.conv1(x, style, noise_mode=noise_mode)
        img = self.toRGB(x, style, skip=img)

        return x, img


@persistent_class
class FirstStage(nn.Module):
    def __init__(
        self,
        img_channels,
        img_resolution=256,
        dim=180,
        w_dim=512,
        use_noise=False,
        demodulate=True,
        activation="lrelu",
    ):
        super().__init__()
        res = 64

        self.conv_first = Conv2dLayerPartial(
            in_channels=img_channels + 1,
            out_channels=dim,
            kernel_size=3,
            activation=activation,
        )
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        for i in range(down_time):  # from input size to 64
            self.enc_conv.append(
                Conv2dLayerPartial(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    down=2,
                    activation=activation,
                )
            )

        # from 64 -> 16 -> 64
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1 / 2, 1 / 2, 2, 2]
        num_heads = 6
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        for i, depth in enumerate(depths):
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1 / ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None
            self.tran.append(
                BasicLayer(
                    dim=dim,
                    input_resolution=[res, res],
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_sizes[i],
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=merge,
                )
            )

        # global style
        down_conv = []
        for i in range(int(np.log2(16))):
            down_conv.append(
                Conv2dLayer(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    down=2,
                    activation=activation,
                )
            )
        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down_conv = nn.Sequential(*down_conv)
        self.to_style = FullyConnectedLayer(
            in_features=dim, out_features=dim * 2, activation=activation
        )
        self.ws_style = FullyConnectedLayer(
            in_features=w_dim, out_features=dim, activation=activation
        )
        self.to_square = FullyConnectedLayer(
            in_features=dim, out_features=16 * 16, activation=activation
        )

        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        for i in range(down_time):  # from 64 to input size
            res = res * 2
            self.dec_conv.append(
                DecStyleBlock(
                    res,
                    dim,
                    dim,
                    activation,
                    style_dim,
                    use_noise,
                    demodulate,
                    img_channels,
                )
            )

    def forward(self, images_in, masks_in, ws, noise_mode="random"):
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        skips = []
        x, mask = self.conv_first(x, masks_in)  # input size
        skips.append(x)
        for i, block in enumerate(self.enc_conv):  # input size to 64
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)

        x_size = x.size()[-2:]
        x = feature2token(x)
        mask = feature2token(mask)
        mid = len(self.tran) // 2
        for i, block in enumerate(self.tran):  # 64 to 16
            if i < mid:
                x, x_size, mask = block(x, x_size, mask)
                skips.append(x)
            elif i > mid:
                x, x_size, mask = block(x, x_size, None)
                x = x + skips[mid - i]
            else:
                x, x_size, mask = block(x, x_size, None)

                mul_map = torch.ones_like(x) * 0.5
                mul_map = F.dropout(mul_map, training=True)
                ws = self.ws_style(ws[:, -1])
                add_n = self.to_square(ws).unsqueeze(1)
                add_n = (
                    F.interpolate(
                        add_n, size=x.size(1), mode="linear", align_corners=False
                    )
                    .squeeze(1)
                    .unsqueeze(-1)
                )
                x = x * mul_map + add_n * (1 - mul_map)
                gs = self.to_style(
                    self.down_conv(token2feature(x, x_size)).flatten(start_dim=1)
                )
                style = torch.cat([gs, ws], dim=1)

        x = token2feature(x, x_size).contiguous()
        img = None
        for i, block in enumerate(self.dec_conv):
            x, img = block(
                x, img, style, skips[len(self.dec_conv) - i - 1], noise_mode=noise_mode
            )

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        return img


@persistent_class
class SynthesisNet(nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output image resolution.
        img_channels=3,  # Number of color channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_decay=1.0,
        channel_max=512,  # Maximum number of channels in any layer.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        drop_rate=0.5,
        use_noise=True,
        demodulate=True,
    ):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2

        # first stage
        self.first_stage = FirstStage(
            img_channels,
            img_resolution=img_resolution,
            w_dim=w_dim,
            use_noise=False,
            demodulate=demodulate,
        )

        # second stage
        self.enc = Encoder(
            resolution_log2, img_channels, activation, patch_size=5, channels=16
        )
        self.to_square = FullyConnectedLayer(
            in_features=w_dim, out_features=16 * 16, activation=activation
        )
        self.to_style = ToStyle(
            in_channels=nf(4),
            out_channels=nf(2) * 2,
            activation=activation,
            drop_rate=drop_rate,
        )
        style_dim = w_dim + nf(2) * 2
        self.dec = Decoder(
            resolution_log2, activation, style_dim, use_noise, demodulate, img_channels
        )

    def forward(self, images_in, masks_in, ws, noise_mode="random", return_stg1=False):
        out_stg1 = self.first_stage(images_in, masks_in, ws, noise_mode=noise_mode)

        # encoder
        x = images_in * masks_in + out_stg1 * (1 - masks_in)
        x = torch.cat([masks_in - 0.5, x, images_in * masks_in], dim=1)
        E_features = self.enc(x)

        fea_16 = E_features[4]
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(
            add_n, size=fea_16.size()[-2:], mode="bilinear", align_corners=False
        )
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        E_features[4] = fea_16

        # style
        gs = self.to_style(fea_16)

        # decoder
        img = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode)

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        if not return_stg1:
            return img
        else:
            return img, out_stg1


@persistent_class
class Discriminator(torch.nn.Module):
    def __init__(
        self,
        c_dim,  # Conditioning label (C) dimensionality.
        img_resolution,  # Input resolution.
        img_channels,  # Number of input color channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        channel_decay=1,
        cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
        activation="lrelu",
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4
        self.resolution_log2 = resolution_log2

        if cmap_dim == None:
            cmap_dim = nf(2)
        if c_dim == 0:
            cmap_dim = 0
        self.cmap_dim = cmap_dim

        if c_dim > 0:
            self.mapping = MappingNet(
                z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None
            )

        Dis = [DisFromRGB(img_channels + 1, nf(resolution_log2), activation)]
        for res in range(resolution_log2, 2, -1):
            Dis.append(DisBlock(nf(res), nf(res - 1), activation))

        if mbstd_num_channels > 0:
            Dis.append(
                MinibatchStdLayer(
                    group_size=mbstd_group_size, num_channels=mbstd_num_channels
                )
            )
        Dis.append(
            Conv2dLayer(
                nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation
            )
        )
        self.Dis = nn.Sequential(*Dis)

        self.fc0 = FullyConnectedLayer(nf(2) * 4**2, nf(2), activation=activation)
        self.fc1 = FullyConnectedLayer(nf(2), 1 if cmap_dim == 0 else cmap_dim)

        # for 64x64
        Dis_stg1 = [DisFromRGB(img_channels + 1, nf(resolution_log2) // 2, activation)]
        for res in range(resolution_log2, 2, -1):
            Dis_stg1.append(DisBlock(nf(res) // 2, nf(res - 1) // 2, activation))

        if mbstd_num_channels > 0:
            Dis_stg1.append(
                MinibatchStdLayer(
                    group_size=mbstd_group_size, num_channels=mbstd_num_channels
                )
            )
        Dis_stg1.append(
            Conv2dLayer(
                nf(2) // 2 + mbstd_num_channels,
                nf(2) // 2,
                kernel_size=3,
                activation=activation,
            )
        )
        self.Dis_stg1 = nn.Sequential(*Dis_stg1)

        self.fc0_stg1 = FullyConnectedLayer(
            nf(2) // 2 * 4**2, nf(2) // 2, activation=activation
        )
        self.fc1_stg1 = FullyConnectedLayer(
            nf(2) // 2, 1 if cmap_dim == 0 else cmap_dim
        )

    def forward(self, images_in, masks_in, images_stg1, c):
        x = self.Dis(torch.cat([masks_in - 0.5, images_in], dim=1))
        x = self.fc1(self.fc0(x.flatten(start_dim=1)))

        x_stg1 = self.Dis_stg1(torch.cat([masks_in - 0.5, images_stg1], dim=1))
        x_stg1 = self.fc1_stg1(self.fc0_stg1(x_stg1.flatten(start_dim=1)))

        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
            x_stg1 = (x_stg1 * cmap).sum(dim=1, keepdim=True) * (
                1 / np.sqrt(self.cmap_dim)
            )

        return x, x_stg1


@persistent_class
class Generator(pl.LightningModule):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # resolution of generated image
        img_channels,  # Number of input color channels.
        synthesis_kwargs={},  # Arguments for SynthesisNetwork.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        noise_mode="const",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.noise_mode = noise_mode
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.synthesis = SynthesisNet(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.mapping = MappingNet(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.synthesis.num_layers,
            **mapping_kwargs,
        )

    def forward(
        self,
        images_in,
        masks_in,
        truncation_psi=1,
        truncation_cutoff=None,
        skip_w_avg_update=False,
        return_stg1=False,
    ):
        images_in = (images_in * 2) - 1  # adjusting range to -1/1
        masks_in = 1 - masks_in  # adjust masks to match official pretrain
        z = torch.randn(images_in.size(0), self.img_resolution).to(self.device)
        c = torch.zeros([1, self.c_dim], device="cuda")

        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            skip_w_avg_update=skip_w_avg_update,
        )

        if not return_stg1:
            img = self.synthesis(images_in, masks_in, ws, noise_mode=self.noise_mode)
            return img
        else:
            img, out_stg1 = self.synthesis(
                images_in, masks_in, ws, noise_mode=self.noise_mode, return_stg1=True
            )
            return img, out_stg1


if __name__ == "__main__":
    device = torch.device("cuda:0")
    batch = 1
    res = 512
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3).to(
        device
    )
    D = Discriminator(c_dim=0, img_resolution=res, img_channels=3).to(device)
    img = torch.randn(batch, 3, res, res).to(device)
    mask = torch.randn(batch, 1, res, res).to(device)
    z = torch.randn(batch, 512).to(device)
    G.eval()

    # def count(block):
    #     return sum(p.numel() for p in block.parameters()) / 10 ** 6
    # print('Generator', count(G))
    # print('discriminator', count(D))

    with torch.no_grad():
        img, img_stg1 = G(img, mask, z, None, return_stg1=True)
    print("output of G:", img.shape, img_stg1.shape)
    score, score_stg1 = D(img, mask, img_stg1, None)
    print("output of D:", score.shape, score_stg1.shape)
