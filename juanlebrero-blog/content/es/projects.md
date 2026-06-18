---
title: "Proyectos"
description: "Productos, una consultora, herramientas open-source y sistemas de investigación construidos por Juan Francisco Lebrero."
type: "page"
language: "es"
---


Una selección de lo que construí: productos que estoy llevando al mercado, trabajo de consultoría, herramientas open-source que la gente usa de verdad, y sistemas de investigación. La mayoría está en [GitHub](https://github.com/frizynn).

## Productos

**[AWAM](https://awam.lat)** es el sistema operativo de ejecución para marcas de indumentaria argentinas. Una plataforma de comercio AI-first que corre todo el loop comercial: desde el mensaje de WhatsApp de un cliente hasta la recomendación, el checkout, el pago, la factura electrónica, el descuento de stock y la recuperación de demanda perdida. Seis runtimes de IA especializados sobre un core durable y multi-tenant. Next.js · React · Prisma · Postgres · Mercado Pago · facturación ARCA.

**[Quent](https://quent.live)** es un copiloto de reuniones en tiempo real. Transcripción en vivo y un overlay cognitivo flotante que responde preguntas sobre tus propios documentos (RAG) y genera memory cards mientras hablás. Front end en Electron sobre un pipeline multi-agente con Deepgram, Groq y Cerebras.

**[Reacher](https://reacher.sh)** es un agente autónomo de prospección B2B. Investiga empresas, redacta outreach personalizado, envía, monitorea respuestas y hace follow-up, sin supervisión, a **$0.06 por empresa**. Harness de agentes propio en TypeScript con observation masking, compactación de contexto y tracking de costos.

## Consultoría

**[sesgo.ai](https://sesgo.ai)** es mi consultora de IA: machine learning, data science, agentes de IA, data engineering y MLOps para equipos de LatAm y Estados Unidos. Entre el trabajo reciente está el **Arison WMS Assistant**, una capa de IA conversacional sobre un sistema de gestión de depósitos (WMS).

## Open source y herramientas

**[linkedin-cli](https://github.com/frizynn/linkedin-cli)** ★56. Un cliente de terminal para LinkedIn que funciona con sesiones reales del navegador en vez de OAuth. Leé feed, perfiles y búsquedas; escribí posts, reacciones y comentarios; con manejo de rate-limit y proxy.

**[gralph](https://github.com/frizynn/gralph)** ★16. Una implementación de alto rendimiento del loop autónomo Ralph. Convierte un PRD en un DAG de tareas y corre muchos agentes de código en paralelo, cada uno aislado en su propio git worktree, un PR por tarea.

**[trail](https://github.com/frizynn/trail).** Memoria compartida y append-only para tu equipo y sus agentes de código. Markdown plano en tu repo, se abre como vault de Obsidian, linkea a tu tracker. Sin base de datos, sin daemon.

**[code-optimizer](https://github.com/frizynn/code-optimizer).** Una auditoría de optimización de código que corre 14+ agentes especialistas en paralelo, cada uno cazando una clase de anti-patrón de performance, con PRs de auto-fix.

**[autoagent](https://github.com/frizynn/autoagent).** Un sistema que descubre iterativamente mejores arquitecturas agénticas, usando verificación formal con TLA+ para chequear pipelines concurrentes antes de gastar tokens.

**[wa2vault](https://github.com/frizynn/wa2vault).** Un archivador read-only de WhatsApp que baja chats a un vault de Obsidian, transcribiendo audios localmente con Whisper. Privacy-first y 100% offline.

## Investigación y sistemas de ML

**Vision foundation model de embriones (LiNAR).** Aprendizaje de representaciones self-supervised estilo JEPA sobre imágenes time-lapse de FIV, prediciendo viabilidad sin biopsia.

**Hierarchical Reasoning Model.** Trabajo sobre una arquitectura de razonamiento inspirada en el cerebro de 27M de parámetros, con kernels CUDA propios, que supera a modelos mucho más grandes en ARC-AGI, Sudoku y laberintos.

**[Entrenamiento a escala](https://github.com/frizynn/training-at-scale).** Entrenar LLMs de 70B en GPUs de consumo con FSDP + QLoRA.

**[text2sql](https://github.com/frizynn/text2sql).** Lenguaje natural a SQL, offline y solo en CPU, con refinamiento por MCTS y decodificación restringida por gramática.

---

Esto es una muestra. Alrededor de 70 repositorios más, desde simuladores de ARMv8 y filesystems Unix-v6 hasta trabajos de visión y NLP, están en [github.com/frizynn](https://github.com/frizynn).
