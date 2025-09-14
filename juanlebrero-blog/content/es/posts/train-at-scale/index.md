---
title: "Entrenamiento a Gran Escala: FSDP, QLoRA, y más."
date: 2025-09-13T00:00:00Z
draft: false
type: post
language: "es"
author: "Juan Francisco Lebrero"
description: ""
tags: ["LoRA", "QLoRA", "LLMs", "finetuning", "cuantización", "4-bit", "deepspeed", "fsdp", "zero", "precisión", "JAX", "bfloat16", "fp16"]
categories: ["IA", "LLMs"]
math: true
---



Para poder entrenar modelos a gran escala, necesitamos entender diversos conceptos que nos van a ayudar a optimizar el rendimiento y la estabilidad del entrenamiento. Por eso, vamos a ver conceptos como precisión numérica, paralelización de datos, cuantización, LoRA, y más.


## Precisión numérica


La elección del formato numérico (FP32, FP16, BF16, FP8, INT8, etc.) constituye uno de los factores más determinantes para el rendimiento, el uso de memoria y la estabilidad del entrenamiento de modelos de gran escala. Por eso, es importante entender como funciona la precisión numérica y como afecta al rendimiento de los modelos, lo que se explicará en esta sección.



### ¿Por qué Importa la Precisión?


¿Como representarías el número $\pi$, un número con decimales INFINITOS, en algo FINITO como lo es una computadora? De está pregunta, surge como respuesta el punto flotante.

Los números representados con punto flotante representan, de manera aproximada, a los números reales, y lo hacen con dos componentes clave: la _mantisa_ y el _exponente_.


Un número en punto flotante representa aproximadamente un valor real mediante la fórmula:


$$\text{valor} \approx \text{signo} \times \text{mantisa} \times \text{base}^{\text{exponente}}$$


donde:
- **Mantisa**: Controla la resolución fina (cuántos pasos discretos entra en el intervalo 1.0 - 2.0)
- **Exponente**: Determina el rango dinámico (qué tan grandes o pequeños pueden ser los números representables)
- **Base**: En IEEE 754 es 2.


Más bits para la mantisa $\rightarrow$ mayor precisión

Más bits para el exponente $\rightarrow$ mayor rango


<figure>
 <img src="images/layout.png" alt="Representación de Punto Flotante de 32 bits (FP32)" style="width: 100%; height: auto;">
 <figcaption style="text-align: center; font-size: 0.95em; color: #666;">
   Figura 1. Esquema de la representación de un número en punto flotante de 32 bits (FP32) según el estándar IEEE 754. <br>
  
 </figcaption>
</figure>


Pero, ¿por qué nos interesa a nosotros? 


Bueno, hay tres razones principales por las que la precisión numérica es crucial en el entrenamiento de modelos de gran escala:


- **Eficiencia computacional**: Los formatos de menor ancho de bits aceleran el cómputo en Tensor Cores/TPUs y reducen bastante el uso de memoria.


- **Estabilidad numérica**: Básicamente, si el formato de número no tiene suficiente rango o detalle, los números pueden volverse demasiado grandes, demasiado chiquitos o perder precisión, lo que puede causar errores o resultados raros durante el entrenamiento.


- **Escabilidad**: Cuando entrenamos LLMs a gran escala, aprovechar la precisión mixta es crucial para que el costo computacional no se nos vaya a la luna.

<figure>
 <img src="images/floating_point.png" alt="Comparativa de formatos de punto flotante: BF16, FP32 y FP16">
 <figcaption style="text-align: center; font-size: 0.95em; color: #666;">
   Figura 2. Comparación visual de los formatos de punto flotante más utilizados en deep learning: <b>BF16</b>, <b>FP32</b> y <b>FP16</b>. <br>
  
 </figcaption>
</figure>




### FP32 (IEEE 754, precisión simple)

Este es el formato “normal” que se usa casi siempre. Guarda los números usando 1 bit para el signo, 8 para el exponente y 23 para la parte decimal (mantisa). Puede representar números muy chicos y muy grandes, desde $1.18\times10^{-38}$ hasta $3.4\times10^{38}$, y su precisión es muy alta ($\varepsilon \approx 1.19\times10^{-7}$).

En machine learning, FP32 es lo que se considera “precisión completa”. Incluso cuando usamos otros formatos para ahorrar memoria, los cálculos importantes (como acumular los gradientes) se hacen en FP32 para que el entrenamiento no se vuelva inestable.


### FP16 (IEEE 754, half)

FP16 es un formato de número que usa menos memoria y permite que todo vaya más rápido. Básicamente, guarda los números usando menos bits que el formato normal (FP32), así que ocupa menos espacio y acelera los cálculos.

Lo bueno: hace que entrenar y usar modelos sea más rápido y barato. Lo malo: como tiene menos detalle y menos rango, a veces los números muy chicos pueden desaparecer (por eso se suele usar <a href="https://picdictionary.com/ml-dictionary/loss-scaling-in-ai-and-deep-learning" target="_blank" rel="noopener">loss scaling</a> para evitarlo), y si los números son muy grandes, se pueden "saturar" y perder información.



### BF16 (Brain Floating Point)

BF16 es el formato que más se usa hoy para entrenar modelos grandes en TPUs y GPUs modernas (como la H100). 

Guarda los números de una forma parecida a FP32 (el formato “normal”), pero con menos detalle en los decimales. Lo importante es que puede representar números igual de grandes o chicos que FP32, así que no se “rompe” con números extremos. Además, casi siempre funciona bien sin tener que hacer trucos raros como el _loss scaling_. Aunque no tiene tanta precisión en los decimales como FP16, para entrenar modelos grandes (como los LLMs) suele ser suficiente y no da problemas.


## Comparación de precisiones


Para comparar las precisiones, voy a usar JAX, un framework de ML hecho por Google, que permite realizar operaciones de manera eficiente en GPUs y TPUs. La razón de utilizar JAX y no PyTorch, por ejemplo, es que JAX nos permitirá más adelante ver en "crudo" la paralelización de las operaciones y la optimización de la memoria.


Primero, importamos JAX y vemos la versión y el backend, así como los dispositivos disponibles:


```python
import jax
# Configuración de JAX
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
```


Necesitamos funciones para obtener el uso de memoria y medir el tiempo de ejecución. Para esto, vamos a usar `psutil`, `tracemalloc` y `time`.


```python
import jax.numpy as jnp
import psutil
import tracemalloc
import time


def get_memory_usage():
   """Obtiene el uso actual de memoria en MB"""
   process = psutil.Process()
   return process.memory_info().rss / 1024 / 1024


def measure_memory_and_time(func):
   def wrapper(*args, **kwargs):
       tracemalloc.start()
       start_memory = get_memory_usage()
      
       start_time = time.time()
       result = func(*args, **kwargs)
       jax.block_until_ready(result)
       end_time = time.time()
      
       end_memory = get_memory_usage()
       current, peak = tracemalloc.get_traced_memory()
       tracemalloc.stop()
      
       return {
           'result': result,
           'execution_time': end_time - start_time,
           'memory_delta': end_memory - start_memory,
           'peak_memory': peak / 1024 / 1024,
           'current_memory': current / 1024 / 1024
       }
   return wrapper
```


Ahora, vamos a medir el rendimiento y la memoria de la multiplicación de matrices y la "red neuronal". Para esto, vamos a usar el decorador `measure_memory_and_time` que definimos anteriormente.



```python
@measure_memory_and_time
def matrix_multiplication_test(dtype, shape):
   """Prueba de multiplicación de matrices con precisión dada"""
   key = jax.random.PRNGKey(42)
   key1, key2 = jax.random.split(key)
  
   def matmul_operation():
       a = jax.random.normal(key1, shape, dtype=dtype)
       b = jax.random.normal(key2, shape, dtype=dtype)
       return jnp.dot(a, b)
  
   return matmul_operation()


@measure_memory_and_time
def neural_network_forward_pass_test(dtype, input_size, hidden_size, output_size):
   """Prueba de pase forward de red neuronal simple"""
   key = jax.random.PRNGKey(123)
   keys = jax.random.split(key, 3)
  
   def nn_forward():
       # Inicialización de pesos
       W1 = jax.random.normal(keys[0], (input_size, hidden_size), dtype=dtype)
       b1 = jax.random.normal(keys[1], (hidden_size,), dtype=dtype)
       W2 = jax.random.normal(keys[2], (hidden_size, output_size), dtype=dtype)
       b2 = jax.random.normal(keys[2], (output_size,), dtype=dtype)
      
       # Datos de entrada
       x = jax.random.normal(keys[0], (input_size,), dtype=dtype)
      
       # Pase forward
       h = jnp.tanh(jnp.dot(x, W1) + b1)
       y = jnp.dot(h, W2) + b2
      
       return y
  
   return nn_forward()
```
Para correr las pruebas, simplemente llamamos a las funciones `matrix_multiplication_test` y `neural_network_forward_pass_test` con los tipos de precisión y las dimensiones de las matrices y la red neuronal, por ejemplo:


```python
matrix_multiplication_test(jnp.float16, (5000, 5000))
neural_network_forward_pass_test(jnp.float16, 784, 256, 10)
```

Para correr las pruebas, voy a incluir también FP64, para mostrar algo que puede ser contraintuitivo.

Dependiendo en qué hardware las estemos corriendo, los resultados pueden variar bastante. 
Corriendo las pruebas en un M1, obtenemos los siguientes resultados para la multiplicación de matrices:

| Precisión | Tiempo (s) | Memoria Pico (MB) |
| --------- | ---------- | ----------------- |
| FP16      | 1.024      | 0.013             |
| BF16      | 0.978      | 0.008             |
| FP32      | 0.943      | 0.008             |
| FP64      | 0.928      | 0.009             |

Y para la red neuronal:

| Precisión | Tiempo (s) | Memoria Pico (MB) |
| --------- | ---------- | ----------------- |
| FP16      | 0.007      | 0.020             |
| BF16      | 0.004      | 0.015             |
| FP32      | 0.003      | 0.015             |
| FP64      | 0.002      | 0.016             |

A simple vista, uno pensaría: “entonces FP64 es lo mejor”. Pero en realidad no es así. Estos resultados se explican por varios factores:

1. **CPUs están optimizadas para ciertos tipos**: Los procesadores como el M1 funcionan mejor con FP32 y FP64 porque las librerías que usan (como Accelerate en Mac) están hechas para esos formatos. En cambio, FP16 y BF16 no están tan bien soportados en CPU, así que muchas veces el sistema tiene que convertirlos a FP32 o FP64 antes de hacer las cuentas, y eso las hace más lentas cuando el problema es chico.

2. **El tamaño importa**: En los ejemplos de las tablas, las matrices y redes son chicas. Cuando los datos son pequeños, la mayor parte del tiempo se va en preparar todo (inicializar, convertir tipos, sincronizar), no en hacer las cuentas en sí. Por eso, a veces FP64 parece “más rápido”, pero es porque el camino para ese tipo es más directo y está mejor optimizado.

3. **En GPU es al revés**: En las GPUs, FP16 y BF16 son mucho más rápidos porque el hardware tiene partes especiales (como Tensor Cores en NVIDIA) que están hechas para trabajar con estos formatos de baja precisión y pueden hacer muchas operaciones a la vez, usando menos memoria y ancho de banda.


Para que lo comprueben ustedes mismos, les propongo que ejecuten las pruebas en una GPU usando matrices de dimensiones mucho mayores; así podrán observar la diferencia por su cuenta. A continuación les muestro una gráfica que ilustra claramente esa gran diferencia.


<div style="display: flex; justify-content: center;">
  <figure>
    <img src="images/nvidia-a100-matmul-tflops.png" alt="Comparación de precisiones" style="max-width: 350px; height: auto;">
    <figcaption style="text-align: center; font-size: 0.95em; color: #666;">
      Figura 3. Comparación de precisiones. <br>
    </figcaption>
  </figure>
</div>



Por útimo, a modo de conclusión de esta sección, les dejo un ejemplo de implementación de precisión mixta, que es lo que suele hacerse en la práctica para entrenar modelos a gran escala.

La idea central es simple: las partes del modelo que requieren estabilidad numérica se calculan en FP32, mientras que los resultados intermedios, gradientes y parámetros se almacenan en FP16 o BF16, aprovechando así el ahorro de memoria y el mayor throughput del hardware.


```python
# Ejemplo de entrenamiento con precisión mixta
def mixed_precision_forward_pass(x, W, b):
   # Convertir a FP32 para cómputo
   x_fp32 = x.astype(jnp.float32)
   W_fp32 = W.astype(jnp.float32)
   b_fp32 = b.astype(jnp.float32)
  
   # Pase forward
   y = jnp.dot(x_fp32, W_fp32) + b_fp32
  
   # Convertir de vuelta a FP16 para eficiencia de memoria
   return y.astype(jnp.float16)
```


# Quantización


