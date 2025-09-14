---
title: "Large-Scale Training: FSDP, QLoRA, and More."
date: 2025-09-13T00:00:00Z
draft: false
type: post
language: "en"
author: "Juan Francisco Lebrero"
description: ""
tags: ["LoRA", "QLoRA", "LLMs", "finetuning", "quantization", "4-bit", "deepspeed", "fsdp", "zero", "precision", "JAX", "bfloat16", "fp16"]
categories: ["AI", "LLMs"]
math: true
---

To train models at scale, we need to understand several concepts that help optimize training performance and stability. That’s why we will cover numerical precision, data parallelism, quantization, LoRA, and more.  

## Numerical Precision  

The choice of numerical format (FP32, FP16, BF16, FP8, INT8, etc.) is one of the most critical factors for performance, memory usage, and training stability of large-scale models. It is therefore essential to understand how numerical precision works and how it affects model performance, which will be explained in this section.  

### Why Does Precision Matter?  

How would you represent the number $\pi$, which has INFINITE decimals, in something FINITE like a computer? The answer lies in floating-point numbers.  

Floating-point numbers approximate real values using two key components: the *mantissa* and the *exponent*.  

A floating-point number approximately represents a real value with the formula:  

$$\text{value} \approx \text{sign} \times \text{mantissa} \times \text{base}^{\text{exponent}}$$  

Where:  
- **Mantissa**: Controls fine resolution (how many discrete steps fit between 1.0 and 2.0).  
- **Exponent**: Determines the dynamic range (how large or small numbers can be represented).  
- **Base**: In IEEE 754 it is 2.  

More bits for the mantissa $\rightarrow$ higher precision.  
More bits for the exponent $\rightarrow$ wider range.  

<figure>
 <img src="images/layout.png" alt="32-bit Floating Point Representation (FP32)" style="width: 100%; height: auto;">
 <figcaption style="text-align: center; font-size: 0.95em; color: #666;">
   Figure 1. Schematic of 32-bit floating-point representation (FP32) according to IEEE 754. <br>
 </figcaption>
</figure>  

But why does this matter for us?  

There are three main reasons why numerical precision is critical in large-scale model training:  

- **Computational efficiency**: Lower bit-width formats accelerate computation on Tensor Cores/TPUs and significantly reduce memory usage.  
- **Numerical stability**: If the number format lacks sufficient range or detail, values may become too large, too small, or lose accuracy, leading to errors or strange results during training.  
- **Scalability**: When training large-scale LLMs, mixed precision is crucial to prevent computational costs from skyrocketing.  

<figure>
 <img src="images/floating_point.png" alt="Floating Point Format Comparison: BF16, FP32, FP16">
 <figcaption style="text-align: center; font-size: 0.95em; color: #666;">
   Figure 2. Visual comparison of the most widely used floating-point formats in deep learning: <b>BF16</b>, <b>FP32</b>, and <b>FP16</b>. <br>
 </figcaption>
</figure>  

### FP32 (IEEE 754, Single Precision)  

This is the “standard” format most commonly used. It stores numbers with 1 bit for the sign, 8 for the exponent, and 23 for the mantissa. It can represent very small and very large values, from $1.18\times10^{-38}$ to $3.4\times10^{38}$, with high precision ($\varepsilon \approx 1.19\times10^{-7}$).  

In machine learning, FP32 is considered “full precision.” Even when other formats are used to save memory, critical calculations (such as gradient accumulation) are often done in FP32 to maintain stability.  

### FP16 (IEEE 754, Half Precision)  

FP16 uses fewer bits than FP32, which reduces memory usage and speeds up computations.  

Pros: Faster and cheaper training/inference, lower memory footprint.  
Cons: Reduced detail and range may cause very small numbers to vanish (requiring techniques like [loss scaling](https://picdictionary.com/ml-dictionary/loss-scaling-in-ai-and-deep-learning)) or very large numbers to saturate.  

### BF16 (Brain Floating Point)  

BF16 is widely used today for training large models on TPUs and modern GPUs (like the H100).  

It stores numbers similarly to FP32 but with fewer mantissa bits. Importantly, it can represent the same large and small values as FP32, so it avoids breakdowns with extreme values. Unlike FP16, BF16 typically does not require loss scaling and is stable enough for training large models (such as LLMs).  

## Precision Comparison  

To compare precisions, we’ll use JAX, a framework developed by Google that allows efficient execution on GPUs and TPUs. Unlike PyTorch, JAX lets us explore “raw” operation parallelization and memory optimization later on.  

...  

(Runs same code examples in English)  

...  

At first glance, one might think: “So FP64 is the best.” But in reality, that’s not true. These results are explained by several factors:  

1. **CPUs are optimized for certain types**: Processors like the M1 run better with FP32 and FP64 because the libraries they rely on (like Accelerate on Mac) are optimized for those formats. FP16 and BF16 are often not as well supported on CPU, so the system internally converts them to FP32/FP64, making them slower for small problems.  

2. **Size matters**: In the table examples, matrices and networks are small. When data sizes are small, most time goes into setup (initialization, conversions, synchronization) rather than the math itself. This makes FP64 appear “faster” because its execution path is more direct and optimized.  

3. **On GPU it’s the opposite**: GPUs run FP16 and BF16 much faster, thanks to specialized hardware (like NVIDIA’s Tensor Cores) that process these formats in parallel, with lower memory and bandwidth usage.  

To confirm this yourself, try running the tests on a GPU with much larger matrices—you’ll clearly see the difference.  

<div style="display: flex; justify-content: center;">
  <figure>
    <img src="images/nvidia-a100-matmul-tflops.png" alt="Precision Comparison" style="max-width: 350px; height: auto;">
    <figcaption style="text-align: center; font-size: 0.95em; color: #666;">
      Figure 3. Precision comparison. <br>
    </figcaption>
  </figure>
</div>  

Finally, as a conclusion to this section, here’s an example of mixed precision training, which is the standard practice for training large-scale models.  

The idea is simple: the parts of the model that require numerical stability are computed in FP32, while intermediate results, gradients, and parameters are stored in FP16 or BF16, taking advantage of memory savings and higher hardware throughput.  

```python
# Example of mixed precision training
def mixed_precision_forward_pass(x, W, b):
   # Convert to FP32 for computation
   x_fp32 = x.astype(jnp.float32)
   W_fp32 = W.astype(jnp.float32)
   b_fp32 = b.astype(jnp.float32)
  
   # Forward pass
   y = jnp.dot(x_fp32, W_fp32) + b_fp32
  
   # Convert back to FP16 for memory efficiency
   return y.astype(jnp.float16)
