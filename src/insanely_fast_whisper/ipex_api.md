API Documentation
General
ipex.optimize(model, dtype=None, optimizer=None, level='O1', inplace=False, conv_bn_folding=None, linear_bn_folding=None, weights_prepack=None, replace_dropout_with_identity=None, optimize_lstm=None, split_master_weight_for_bf16=None, fuse_update_step=None, auto_kernel_selection=None, sample_input=None, graph_mode=None, concat_linear=None)
Apply optimizations at Python frontend to the given model (nn.Module), as well as the given optimizer (optional). If the optimizer is given, optimizations will be applied for training. Otherwise, optimization will be applied for inference. Optimizations include conv+bn folding (for inference only), weight prepacking and so on.

Weight prepacking is a technique to accelerate performance of oneDNN operators. In order to achieve better vectorization and cache reuse, onednn uses a specific memory layout called blocked layout. Although the calculation itself with blocked layout is fast enough, from memory usage perspective it has drawbacks. Running with the blocked layout, oneDNN splits one or several dimensions of data into blocks with fixed size each time the operator is executed. More details information about oneDNN data mermory format is available at oneDNN manual. To reduce this overhead, data will be converted to predefined block shapes prior to the execution of oneDNN operator execution. In runtime, if the data shape matches oneDNN operator execution requirements, oneDNN won’t perform memory layout conversion but directly go to calculation. Through this methodology, called weight prepacking, it is possible to avoid runtime weight data format convertion and thus increase performance.

Parameters
:
model (torch.nn.Module) – User model to apply optimizations on.

dtype (torch.dtype) – Only works for torch.bfloat16 and torch.half a.k.a torch.float16. Model parameters will be casted to torch.bfloat16 or torch.half according to dtype of settings. The default value is None, meaning do nothing. Note: Data type conversion is only applied to nn.Conv2d, nn.Linear and nn.ConvTranspose2d for both training and inference cases. For inference mode, additional data type conversion is applied to the weights of nn.Embedding and nn.LSTM.

optimizer (torch.optim.Optimizer) – User optimizer to apply optimizations on, such as SGD. The default value is None, meaning inference case.

level (string) – "O0" or "O1". No optimizations are applied with "O0". The optimizer function just returns the original model and optimizer. With "O1", the following optimizations are applied: conv+bn folding, weights prepack, dropout removal (inferenc model), master weight split and fused optimizer update step (training model). The optimization options can be further overridden by setting the following options explicitly. The default value is "O1".

inplace (bool) – Whether to perform inplace optimization. Default value is False.

conv_bn_folding (bool) – Whether to perform conv_bn folding. It only works for inference model. The default value is None. Explicitly setting this knob overwrites the configuration set by level knob.

linear_bn_folding (bool) – Whether to perform linear_bn folding. It only works for inference model. The default value is None. Explicitly setting this knob overwrites the configuration set by level knob.

weights_prepack (bool) – Whether to perform weight prepack for convolution and linear to avoid oneDNN weights reorder. The default value is None. Explicitly setting this knob overwrites the configuration set by level knob. Weight prepack works for CPU only.

replace_dropout_with_identity (bool) – Whether to replace nn.Dropout with nn.Identity. If replaced, the aten::dropout won’t be included in the JIT graph. This may provide more fusion opportunites on the graph. This only works for inference model. The default value is None. Explicitly setting this knob overwrites the configuration set by level knob.

optimize_lstm (bool) – Whether to replace nn.LSTM with IPEX LSTM which takes advantage of oneDNN kernels to get better performance. The default value is None. Explicitly setting this knob overwrites the configuration set by level knob.

split_master_weight_for_bf16 (bool) – Whether to split master weights update for BF16 training. This saves memory comparing to master weight update solution. Split master weights update methodology doesn’t support all optimizers. The default value is None. The default value is None. Explicitly setting this knob overwrites the configuration set by level knob.

fuse_update_step (bool) – Whether to use fused params update for training which have better performance. It doesn’t support all optimizers. The default value is None. Explicitly setting this knob overwrites the configuration set by level knob.

sample_input (tuple or torch.Tensor) – Whether to feed sample input data to ipex.optimize. The shape of input data will impact the block format of packed weight. If not feed a sample input, Intel® Extension for PyTorch* will pack the weight per some predefined heuristics. If feed a sample input with real input shape, Intel® Extension for PyTorch* can get best block format. Sample input works for CPU only.

auto_kernel_selection (bool) – Different backends may have different performances with different dtypes/shapes. Default value is False. Intel® Extension for PyTorch* will try to optimize the kernel selection for better performance if this knob is set to True. You might get better performance at the cost of extra memory usage. The default value is None. Explicitly setting this knob overwrites the configuration set by level knob. Auto kernel selection works for CPU only.

graph_mode – (bool) [prototype]: It will automatically apply a combination of methods to generate graph or multiple subgraphs if True. The default value is False.

concat_linear (bool) – Whether to perform concat_linear. It only works for inference model. The default value is None. Explicitly setting this knob overwrites the configuration set by level knob.

Returns
:
Model and optimizer (if given) modified according to the level knob or other user settings. conv+bn folding may take place and dropout may be replaced by identity. In inference scenarios, convolutuon, linear and lstm will be replaced with the optimized counterparts in Intel® Extension for PyTorch* (weight prepack for convolution and linear) for good performance. In bfloat16 or float16 scenarios, parameters of convolution and linear will be casted to bfloat16 or float16 dtype.

Warning

Please invoke optimize function BEFORE invoking DDP in distributed training scenario.

The optimize function deepcopys the original model. If DDP is invoked before optimize function, DDP is applied on the origin model, rather than the one returned from optimize function. In this case, some operators in DDP, like allreduce, will not be invoked and thus may cause unpredictable accuracy loss.

Examples

# bfloat16 inference case.
model = ...
model.load_state_dict(torch.load(PATH))
model.eval()
optimized_model = ipex.optimize(model, dtype=torch.bfloat16)
# running evaluation step.
# bfloat16 training case.
optimizer = ...
model.train()
optimized_model, optimized_optimizer = ipex.optimize(model, dtype=torch.bfloat16, optimizer=optimizer)
# running training step.
torch.xpu.optimize() is an alternative of optimize API in Intel® Extension for PyTorch*, to provide identical usage for XPU device only. The motivation of adding this alias is to unify the coding style in user scripts base on torch.xpu modular.

Examples

# bfloat16 inference case.
model = ...
model.load_state_dict(torch.load(PATH))
model.eval()
optimized_model = torch.xpu.optimize(model, dtype=torch.bfloat16)
# running evaluation step.
# bfloat16 training case.
optimizer = ...
model.train()
optimized_model, optimized_optimizer = torch.xpu.optimize(model, dtype=torch.bfloat16, optimizer=optimizer)
# running training step.
ipex.llm.optimize(model, optimizer=None, dtype=torch.float32, inplace=False, device='cpu', quantization_config=None, qconfig_summary_file=None, low_precision_checkpoint=None, sample_inputs=None, deployment_mode=True, cache_weight_for_large_batch=False)
Apply optimizations at Python frontend to the given transformers model (nn.Module). This API focus on transformers models, especially for generation tasks inference.

Well supported model family with full functionalities: Llama, MLlama, GPT-J, GPT-Neox, OPT, Falcon, Bloom, CodeGen, Baichuan, ChatGLM, GPTBigCode, T5, Mistral, MPT, Mixtral, StableLM, QWen, Git, Llava, Yuan, Phi, Whisper, Maira2, Jamba, DeepSeekV2.

For the model that is not in the scope of supported model family above, will try to apply default ipex.optimize transparently to get benifits (not include quantizations, only works for dtypes of torch.bfloat16 and torch.half and torch.float).

Parameters
:
model (torch.nn.Module) – User model to apply optimizations.

optimizer (torch.optim.Optimizer) – User optimizer to apply optimizations on, such as SGD. The default value is None, meaning inference case.

dtype (torch.dtype) – Now it works for torch.bfloat16 and torch.float. The default value is torch.float. When working with quantization, it means the mixed dtype with quantization.

inplace (bool) – Whether to perform inplace optimization. Default value is False.

device (str) – Specifying the device on which the optimization will be performed-either ‘CPU’ or ‘XPU.

quantization_config (object) – Defining the IPEX quantization recipe (Weight only quant or static quant). Default value is None. Once used, meaning using IPEX quantizatization model for model.generate().(only works on CPU)

qconfig_summary_file (str) – Path to the IPEX static quantization config json file. (only works on CPU) Default value is None. Work with quantization_config under static quantization use case. Need to do IPEX static quantization calibration and generate this file. (only works on CPU)

low_precision_checkpoint (dict or tuple of dict) – For weight only quantization with INT4 weights. If it’s a dict, it should be the state_dict of checkpoint generated by GPTQ by default. If a tuple is provided, it should be (checkpoint, quant_method), where checkpoint is the state_dict and quant_method is dict specifying the quantization method including GPTQ or AWQ, e,g, quant_method = {quant_method: gptq}.

sample_inputs (Tuple tensors) – sample inputs used for model quantization or torchscript. Default value is None, and for well supported model, we provide this sample inputs automaticlly. (only works on CPU)

deployment_mode (bool) – Whether to apply the optimized model for deployment of model generation. It means there is no need to further apply optimization like torchscirpt. Default value is True. (only works on CPU)

cache_weight_for_large_batch (bool) – Whether to cache the dedicated weight for large batch to speed up its inference (e.g., prefill phase) with extra memory usage. It is only valid for non-quantization cases where dtype = bfloat16 and weight-only quantization cases where lowp-mode=BF16/INT8. In other cases, an error will be raised. Default value is False.

Returns
:
optimized model object for model.generate(), also workable with model.forward

Warning

Please invoke ipex.llm.optimize function AFTER invoking DeepSpeed in Tensor Parallel inference scenario.

Examples

# bfloat16 generation inference case.
model = ...
model.load_state_dict(torch.load(PATH))
model.eval()
optimized_model = ipex.llm.optimize(model, dtype=torch.bfloat16)
optimized_model.generate()
ipex.get_fp32_math_mode(device='cpu')
Get the current fpmath_mode setting.

Parameters
:
device (string) – cpu, xpu

Returns
:
Fpmath mode The value will be FP32MathMode.FP32, FP32MathMode.BF32 or FP32MathMode.TF32 (GPU ONLY). oneDNN fpmath mode will be disabled by default if dtype is set to FP32MathMode.FP32. The implicit FP32 to TF32 data type conversion will be enabled if dtype is set to FP32MathMode.TF32. The implicit FP32 to BF16 data type conversion will be enabled if dtype is set to FP32MathMode.BF32.

Examples

import intel_extension_for_pytorch as ipex
# to get the current fpmath mode
ipex.get_fp32_math_mode(device="xpu")
torch.xpu.get_fp32_math_mode() is an alternative function in Intel® Extension for PyTorch*, to provide identical usage for XPU device only. The motivation of adding this alias is to unify the coding style in user scripts base on torch.xpu modular.

Examples

import intel_extension_for_pytorch as ipex
# to get the current fpmath mode
torch.xpu.get_fp32_math_mode(device="xpu")
ipex.set_fp32_math_mode(mode=FP32MathMode.FP32, device='cpu')
Enable or disable implicit data type conversion.

Parameters
:
mode (FP32MathMode) – FP32MathMode.FP32, FP32MathMode.BF32 or FP32MathMode.TF32 (GPU ONLY). oneDNN fpmath mode will be disabled by default if dtype is set to FP32MathMode.FP32. The implicit FP32 to TF32 data type conversion will be enabled if dtype is set to FP32MathMode.TF32. The implicit FP32 to BF16 data type conversion will be enabled if dtype is set to FP32MathMode.BF32.

device (string) – cpu, xpu

Examples

import intel_extension_for_pytorch as ipex
# to enable the implicit data type conversion
ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.BF32)
# to disable the implicit data type conversion
ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.FP32)
torch.xpu.set_fp32_math_mode() is an alternative function in Intel® Extension for PyTorch*, to provide identical usage for XPU device only. The motivation of adding this alias is to unify the coding style in user scripts base on torch.xpu modular.

Examples

import intel_extension_for_pytorch as ipex
# to enable the implicit data type conversion
torch.xpu.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.BF32)
# to disable the implicit data type conversion
torch.xpu.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.FP32)
Memory management
torch.xpu.empty_cache()→ None
Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in sysman toolkit.

Note

empty_cache() doesn’t increase the amount of GPU memory available for PyTorch. However, it may help reduce fragmentation of GPU memory in certain cases. See Memory Management [GPU] for more details about GPU memory management.

torch.xpu.mem_get_info(device: device | str | int | None = None)→ Tuple[int, int]
Return the estimated value of global free and total GPU memory for a given device.

Parameters
:
device (torch.device or int or str, optional) – selected device. Returns statistic for the current device, given by current_device(), if device is None (default) or if the device index is not specified.

Note

See Memory Management [GPU] for more details about GPU memory management.

torch.xpu.memory_stats(device: device | str | int | None = None)→ Dict[str, Any]
Returns a dictionary of XPU memory allocator statistics for a given device.

The return value of this function is a dictionary of statistics, each of which is a non-negative integer.

Core statistics:

"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}": number of allocation requests received by the memory allocator.

"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}": amount of allocated memory.

"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}": number of reserved segments from xpuMalloc().

"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}": amount of reserved memory.

"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}": number of active memory blocks.

"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}": amount of active memory.

"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}": number of inactive, non-releasable memory blocks.

"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}": amount of inactive, non-releasable memory.

For these core statistics, values are broken down as follows.

Pool type:

all: combined statistics across all memory pools.

large_pool: statistics for the large allocation pool (as of October 2019, for size >= 1MB allocations).

small_pool: statistics for the small allocation pool (as of October 2019, for size < 1MB allocations).

Metric type:

current: current value of this metric.

peak: maximum value of this metric.

allocated: historical total increase in this metric.

freed: historical total decrease in this metric.

In addition to the core statistics, we also provide some simple event counters:

"num_alloc_retries": number of failed xpuMalloc calls that result in a cache flush and retry.

"num_ooms": number of out-of-memory errors thrown.

Parameters
:
device (torch.device or int, optional) – selected device. Returns statistics for the current device, given by current_device(), if device is None (default).

Note

See Memory Management [GPU] for more details about GPU memory management.

torch.xpu.memory_summary(device: device | str | int | None = None, abbreviated: bool = False)→ str
Returns a human-readable printout of the current memory allocator statistics for a given device.

This can be useful to display periodically during training, or when handling out-of-memory exceptions.

Parameters
:
device (torch.device or int, optional) – selected device. Returns printout for the current device, given by current_device(), if device is None (default).

abbreviated (bool, optional) – whether to return an abbreviated summary (default: False).

Note

See Memory Management [GPU] for more details about GPU memory management.

torch.xpu.memory_snapshot()
Returns a snapshot of the XPU memory allocator state across all devices.

Interpreting the output of this function requires familiarity with the memory allocator internals.

Note

See Memory Management [GPU] for more details about GPU memory management.

torch.xpu.memory_allocated(device: device | str | int | None = None)→ int
Returns the current GPU memory occupied by tensors in bytes for a given device.

Parameters
:
device (torch.device or int, optional) – selected device. Returns statistic for the current device, given by current_device(), if device is None (default).

Note

This is likely less than the amount shown in sysman toolkit since some unused memory can be held by the caching allocator and some context needs to be created on GPU. See Memory Management [GPU] for more details about GPU memory management.

torch.xpu.max_memory_allocated(device: device | str | int | None = None)→ int
Returns the maximum GPU memory occupied by tensors in bytes for a given device.

By default, this returns the peak allocated memory since the beginning of this program. reset_peak_stats() can be used to reset the starting point in tracking this metric. For example, these two functions can measure the peak allocated memory usage of each iteration in a training loop.

Parameters
:
device (torch.device or int, optional) – selected device. Returns statistic for the current device, given by current_device(), if device is None (default).

Note

See Memory Management [GPU] for more details about GPU memory management.

torch.xpu.memory_reserved(device: device | str | int | None = None)→ int
Returns the current GPU memory managed by the caching allocator in bytes for a given device.

Parameters
:
device (torch.device or int, optional) – selected device. Returns statistic for the current device, given by current_device(), if device is None (default).

Note

See Memory Management [GPU] for more details about GPU memory management.

torch.xpu.max_memory_reserved(device: device | str | int | None = None)→ int
Returns the maximum GPU memory managed by the caching allocator in bytes for a given device.

By default, this returns the peak cached memory since the beginning of this program. reset_peak_stats() can be used to reset the starting point in tracking this metric. For example, these two functions can measure the peak cached memory amount of each iteration in a training loop.

Parameters
:
device (torch.device or int, optional) – selected device. Returns statistic for the current device, given by current_device(), if device is None (default).

Note

See Memory Management [GPU] for more details about GPU memory management.

torch.xpu.reset_peak_memory_stats(device: device | str | int | None = None)→ None
Resets the “peak” stats tracked by the XPU memory allocator.

See memory_stats() for details. Peak stats correspond to the “peak” key in each individual stat dict.

Parameters
:
device (torch.device or int, optional) – selected device. Returns statistic for the current device, given by current_device(), if device is None (default).

Note

See Memory Management [GPU] for more details about GPU memory management.

torch.xpu.memory_stats_as_nested_dict(device: device | str | int | None = None)→ Dict[str, Any]
Returns the result of memory_stats() as a nested dictionary.

torch.xpu.reset_accumulated_memory_stats(device: device | str | int | None = None)→ None
Resets the “accumulated” (historical) stats tracked by the XPU memory allocator.

See memory_stats() for details. Accumulated stats correspond to the “allocated” and “freed” keys in each individual stat dict, as well as “num_alloc_retries” and “num_ooms”.

Parameters
:
device (torch.device or int, optional) – selected device. Returns statistic for the current device, given by current_device(), if device is None (default).

Note

See Memory Management [GPU] for more details about GPU memory management.

Quantization
ipex.quantization.fp8.fp8_autocast(enabled: bool = False, calibrating: bool = False, fp8_recipe: DelayedScaling | None = None, fp8_group: ProcessGroup | None = None, device: str = 'xpu')→ None
Context manager for FP8 usage.

with fp8_autocast(enabled=True):
    out = model(inp)
Parameters
:
enabled (bool, default = True) – whether or not to enable fp8

calibrating (bool, default = False) – calibration mode allows collecting statistics such as amax and scale data of fp8 tensors even when executing without fp8 enabled.

fp8_recipe (recipe.DelayedScaling, default = None) – recipe used for FP8 training.

fp8_group (torch._C._distributed_c10d.ProcessGroup, default = None) – distributed group over which amaxes for the fp8 tensors are reduced at the end of each training step.

C++ API
enum torch_ipex::xpu::FP32_MATH_MODE
specifies the available DPCCP packet types

Values:

enumerator FP32
set floating-point math mode to FP32.

enumerator TF32
set floating-point math mode to TF32.

enumerator BF32
set floating-point math mode to BF32.

enumerator FP32_MATH_MODE_MIN
enumerator FP32_MATH_MODE_MAX
set floating-point math mode.

bool torch_ipex::xpu::set_fp32_math_mode(FP32_MATH_MODE mode)
Enable or disable implicit floating-point type conversion during computation for oneDNN kernels. Set FP32MathMode.FP32 will disable floating-point / type conversion. Set FP32MathMode.TF32 will enable implicit / down-conversion from fp32 to tf32. Set FP32MathMode.BF32 will / enable implicit down-conversion from fp32 to bf16. / / refer to Primitive Attributes: floating / -point math mode for detail description about the definition and / numerical behavior of floating-point math modes. /

Parameters
:
mode – (FP32MathMode): Only works for FP32MathMode.FP32, / FP32MathMode.TF32 and FP32MathMode.BF32. / oneDNN fpmath mode will be disabled by default if dtype is set to / FP32MathMode.FP32. The implicit FP32 to TF32 data type conversion / will be enabled if dtype is set to FP32MathMode.TF32`. The implicit FP32 to BF16 data type conversion will be enabled if dtype is set to FP32MathMode.BF32`.