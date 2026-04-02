# Known Issues

## General Issues

- **distil-whisper updated**: The implementation has been modernized to work with PyTorch 2.7+ and current HuggingFace APIs (Transformers 4.53+, Datasets 4.0+).
- **Cache not utilized properly**: Dataset preprocessing recomputes log-Mels on each run because the cache key includes the entire preprocessing config (including fp16/fp32 toggle). Need to implement caching based on audio SHA-1 instead.
- **Blacklist/evaluate metadata collision**: The blacklist script rewrites Arrow files in-place and drops the "audio/_array" column that evaluate.py expects. Workaround: Run blacklist first, then evaluate, or use `--no-cache`.
- **Documentation command syntax**: Some older documentation incorrectly shows `whisper-tuner --profile <profile>` when it should be `whisper-tuner finetune <profile>`. This has been fixed in most places but may still appear in some examples.

## MPS (Apple Silicon) Specific Issues

### Known Limitations (Updated for PyTorch 2.3 - July 2025)
- **Flash Attention 2**: NOW SUPPORTED! Enable with `export SDPA_ALLOW_FLASH_ATTN=1`. Works for sequences ≤ 4096 and reduces peak memory by ~28%.
- **Mixed Precision**: Use **float16** (native) or float32. Avoid bfloat16 - it's emulated and slower on M-series GPUs. Autocast works for linear/conv/gelu/softmax.
- **Multi-GPU**: Still not supported - no torch.distributed backend for multiple M-series GPUs.
- **Some Operations**: Certain PyTorch ops may fall back to CPU with PYTORCH_ENABLE_MPS_FALLBACK=1

### Common MPS Errors and Solutions

1. **"MPS backend out of memory"**
   - Solution: Reduce batch size or set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`
   - With Flash Attention 2 enabled, memory usage drops ~28%, often eliminating OOMs
   - The unified memory architecture can cause swapping instead of OOM errors

2. **"Operation X not implemented for MPS"**
   - Solution: Set `PYTORCH_ENABLE_MPS_FALLBACK=1` (performance impact)
   - Most common ops now have MPS kernels in PyTorch 2.3
   - Report specific operations for future fixes

3. **Slower than expected performance**
   - Enable Flash Attention 2: `export SDPA_ALLOW_FLASH_ATTN=1`
   - Check if operations are falling back to CPU
   - Disable `PYTORCH_ENABLE_MPS_FALLBACK` after testing
   - Use appropriate batch sizes for your chip

4. **Numerical differences from CUDA**
   - MPS uses different floating-point optimizations
   - Use fp16 (not bf16) for best performance on Apple Silicon
   - Increase tolerance in accuracy checks if needed
   - Results should be functionally equivalent

### Performance Considerations
- **Memory**: Apple Silicon uses unified memory - monitor total system RAM usage
- **Batch Sizes**: 
  - M1/M2 Pro: Start with micro-batch 2-4
  - M1/M2 Max: Start with micro-batch 4-6
  - M1/M2 Ultra: Can handle micro-batch 4-6 comfortably (20s segments, fp16, FA2)
- **Data Loading**: `num_workers > 0` is now stable (was buggy pre-2.3). Sweet spot is 4-6 workers per CPU cluster.
