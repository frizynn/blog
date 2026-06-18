---
title: "No tengo un equipo. Tengo agentes con memoria."
date: 2026-06-18
draft: false
type: post
language: "es"
author: "Juan Francisco Lebrero"
description: "Cómo corro unos veinte repositorios siendo una sola persona. Automatización, memoria de agentes, y la parte que nadie automatiza, más lo que en serio cuesta."
tags: ["agentes", "automatización", "ia", "workflow", "memoria"]
categories: ["ensayos"]
---

Trabajo en el equipo de Mercado Libre que entrena nuestros propios modelos de lenguaje. Es más o menos todo lo que voy a decir de eso. También llevo adelante una pequeña consultora de IA, construyo un SaaS de comercio, hago research en un laboratorio de la facultad, y todavía estoy terminando la carrera de ingeniería. En cualquier momento dado estoy shippeando en alrededor de veinte repositorios.

La gente asume que tengo un equipo. No tengo. Tengo un sistema, y el corazón no es la parte de la que todos hablan. No son los agentes escribiendo código. Es la memoria que tienen abajo. Acá está todo, con honestidad, incluido lo que me cuesta.

Una pregunta justa es cómo se sostiene todo esto con un trabajo full time, así que voy a ser directo. Mercado Libre es uno de verdad, con un manager, un calendario y deadlines, y se lleva mis horas de foco. Todo lo demás vive en los márgenes alrededor. Las mañanas temprano, las noches, los fines de semana, y un server que sigue trabajando cuando ya cerré la laptop.

## Escribir código dejó de ser lo difícil

Hace unos años el cuello de botella era obvio. Tenías una idea, y después te pasabas dos semanas tipeándola hasta que existía. Eso se terminó. Si le pido a un agente que diseñe y construya un backend, es prácticamente un problema resuelto. Me da una arquitectura limpia, las migraciones, los tests, la config de deploy, a la primera.

Así que el cuello de botella se movió. Cuando tipear es gratis, lo que queda es el contexto (mantener una docena de proyectos coherentes en la cabeza), la coordinación (correr muchos agentes sin que se vuelva un caos), y decidir qué construir en primer lugar. Para esas tres cosas es mi sistema. La mayor parte corre mientras duermo.

## Capa 1, las cosas que no vuelvo a tocar a mano

Mi regla es boluda y funciona. Si algo lo hice a mano dos veces, la tercera lo hace un agente.

Me armé mi propio server para esto. Siempre hay algo corriendo. No lo digo para hacer efecto. No me voy a dormir sin tener agentes trabajando, y me da cosa dejar la máquina al pedo cuando hay backlog.

Esto es algo de lo que corre, las veinticuatro horas.

- Una rutina diaria se despierta, audita el SEO técnico y la preparación para citas de IA de mi SaaS, y abre pull requests en draft con contenido. Los reviso con el café.
- [Reacher](https://reacher.sh), un agente autónomo de prospección que escribí, sale a conseguir clientes para la consultora. Investiga empresas, redacta el outreach, lo manda y hace follow-up, a unos seis centavos por empresa. Trabaja la punta de mi embudo mientras yo trabajo en todo lo demás.
- [wa2vault](https://github.com/frizynn/wa2vault) archiva conversaciones de WhatsApp en un vault que mis agentes pueden leer, transcribiendo los audios localmente. Contexto que de otra forma perdería.
- CLIs que construí para las plataformas que no quieren ser automatizadas, para distribución y lectura.
- Cuando estoy en una reunión, tomo notas con [Quent](https://quent.live), mi copiloto de reuniones en tiempo real, y le paso la transcripción directo al agente que va a actuar sobre eso.

Nada de esto es glamoroso. Ese es el punto. Es la capa aburrida que te devuelve horas.

## Capa 2, el problema de la amnesia

Acá está lo que nadie te cuenta cuando empezás a correr agentes a escala. Más agentes sin memoria es simplemente más caos, más rápido.

Un agente que se olvida todo entre sesiones te convierte a vos en la memoria. Pasás a ser el portapapeles humano, reexplicando las mismas decisiones, repegando el mismo contexto, veinte veces por día. Eso no escala. Es agotador de una forma difícil de nombrar hasta que se termina.

Lo que de verdad cambió cómo trabajo fue darles memoria persistente a mis agentes. Cada vez que arranco un proyecto nuevo, se crea un vault con él, y mi CLAUDE.md y mi agents.md están configurados para que los agentes escriban ahí sin que se los pida. Las decisiones, las idas y vueltas, las notas de reuniones, el razonamiento detrás de cada elección, todo cae en Markdown plano. Un archivo por tarea. El vault es el tracker. Append, nunca rewrite. (Me gustó tanto que lo convertí en una herramienta, [trail](https://github.com/frizynn/trail), pero la idea importa más que la herramienta.)

El efecto es difícil de exagerar. Un agente que arranco hoy retoma una decisión que tomé hace tres semanas, en otro repositorio, y simplemente sigue. Dejé de ser el portapapeles.

Y como es Markdown y no código, el contexto viaja. Puedo comprimir un vault y pasárselo a alguien que no programa, su agente lo lee, y queda al día con meses de decisiones en minutos. Esta es la capa de la que nadie postea capturas, y es todo el juego.

## Capa 3, cómo los manejo en serio

Casi no escribo código directo ya. Corro goals en Codex y Claude Code, los dos en el plan más alto, los de 20x, y casi todos los días choco contra los límites de uso. Arriba de eso me apoyo en un stack de skills que fui juntando. Una de ellas, un "thermo-nuclear code quality review" del equipo de Cursor, la corro después de cada implementación que hago, sin excepción. Confío en ella más de lo que podría explicarte del todo, y es un lugar raro donde estar. Voy a volver a eso.

También voy a ser honesto con lo que no uso. Construí un orquestador multi-agente para correr muchos agentes de código en paralelo. Lo terminé abandonando. Algunas herramientas se ganan el lugar y otras no, y pretender lo contrario es como los blogs te mienten.

El único hábito que importa más que cualquier herramienta no es una herramienta. Antes de implementar algo, discuto con el agente diez o quince minutos sobre qué estamos haciendo en realidad. No el código. La forma de la cosa. Los trade-offs. Lo que no vamos a construir. Esa conversación es el trabajo ahora. El código viene después.

## El plan es el nuevo código

Esta es la parte a la que no paro de darle vueltas.

Venimos subiendo una escalera de abstracción hace setenta años. Assembly, después C, después lenguajes con garbage collection, después frameworks. Cada capa nos dejó dejar de pensar en la de abajo. Estamos en un escalón nuevo. El plan, el spec, la discusión de quince minutos, eso es el código fuente. El código de verdad es la salida compilada. Yo edito el plan, y el agente lo compila en un repositorio.

Y va a haber una capa arriba de esta también. Siempre la hay. La pregunta interesante es de qué está hecha.

Porque acá está lo que lo delata. El código está resuelto, y las ideas no.

## Dónde los agentes todavía se caen de cara

Probá esto. Pedile a tu agente que diseñe una arquitectura de backend. Vas a tener algo limpio y correcto, al toque. Ahora pedile que diseñe la arquitectura agéntica óptima para un problema que todavía no tiene una respuesta estándar. Se traba. Te da algo plausible y un poco mal, van y vienen, y a veces no llegás nunca del todo.

El mismo modelo. Resultados salvajemente distintos. ¿Por qué?

Mi explicación es que estos modelos son interpoladores extraordinarios y extrapoladores flojos. Una arquitectura de backend es una región densa, masivamente representada, resuelta del espacio. Hay un millón de buenas en los datos de entrenamiento. El modelo no está inventando tu backend, está recuperando el consenso y ajustándolo a tus detalles. Eso es interpolación, y es genuinamente sobrehumano en eso.

La arquitectura agéntica óptima es la frontera. Los datos son escasos, no hay consenso, y para un montón de problemas reales la respuesta correcta todavía no existe en la distribución. Hay que inventarla. Eso es extrapolación, y extrapolación es una palabra más educada para tener una idea nueva. Los modelos son malos en eso por la misma razón por la que son buenos en lo otro. Están moldeados por lo que ya existe.

Que es exactamente por qué la discusión de quince minutos importa. En esa conversación, yo hago la extrapolación y el agente hace la interpolación. Yo traigo la idea nueva a medio formar, y él trae el peso entero de todo lo que ya se probó, al instante, para que yo pueda testear mi idea contra eso. La colaboración funciona porque cada uno hace la parte que el otro no puede. El problema sin resolver nunca fue escribir el código. Es tener la idea que vale la pena compilar.

## Lo que cuesta

Estaría mintiendo si terminara ahí, en la nota limpia sobre ideas y abstracción.

Esta vida tiene un precio y lo pago todos los días. Trabajo hasta dieciocho horas por día. Me levanto a las ocho y me acuesto a las doce o la una, y aun así algo queda corriendo en el server. De cierta forma lo amo, y quiero ser claro en que esto no es disciplina hablando. Son ganas. Amo de verdad construir cosas y resolver problemas, y dame una noche libre y la voy a gastar en un repositorio.

Pero el costo es real y no lo quiero maquillar. Perdí contacto con amigos. Resigné la libertad de entrenar, de salir a tomar algo cuando tengo ganas, de tener un día que no esté contabilizado. Los agentes corren mientras duermo, pero yo sigo siendo el que en realidad no para.

Hay un costo más silencioso debajo de ese. Todo esto es una torre de Jenga. Cada pieza sostiene a las otras. Los laburos pagan el tiempo, las herramientas y el aire para seguir construyendo, y lo que construyo es lo que algún día podría volver opcionales a los laburos. Sacá un solo bloque y la torre se tambalea, así que no lo saco.

Y acá está lo que todo este sistema terminó enseñándome. Automaticé el código, el outreach, la memoria, el laburo de hormiga, y me devolví horas. Lo único que logró eso fue correr cada excusa hasta que el único problema difícil que quedó fue el humano. No qué construir. A qué comprometerme. El modelo puede interpolar cualquier backend que le describa, pero no puede decidir qué vale una vida. Esa decisión no compila.

Así que avanzo rápido en diez frentes y le digo leverage, y capaz lo es. Pero igual que el modelo es brillante en lo resuelto y se pierde en la frontera, yo soy brillante ejecutando y me da miedo elegir. Correr todo en paralelo se siente como progreso, y lo es, pero también puede ser la forma más sofisticada que encontré de evitar elegir. Los agentes me dieron memoria y tiempo. Lo que todavía no tengo es el coraje de apuntar todo a una sola cosa y averiguar si tenía razón.

Y eso nunca fue un problema de código, lo que me hace pensar que no es solo mío. Las herramientas están volviendo gratis la ejecución para todos, así que lo escaso deja de ser si podés construir algo y pasa a ser si sabés qué vale la pena construir, y si te vas a comprometer. La capacidad va a estar en todos lados. La dirección no. Y cuando construir no cuesta nada, estar ocupado es el lugar más fácil del mundo para esconderse. Así que si te movés en diez direcciones y le decís ambición, preguntate una vez si sos rápido porque elegiste, o porque nunca lo hiciste. Yo todavía me debo la respuesta.
