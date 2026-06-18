---
title: "I don't have a team. I have agents with memory."
date: 2026-06-18
draft: false
type: post
language: "en"
author: "Juan Francisco Lebrero"
description: "How I run about twenty repositories as one person. Automation, agent memory, and the part nobody automates, plus what it actually costs."
tags: ["agents", "automation", "ai", "workflow", "memory"]
categories: ["essays"]
---

I work on the team at Mercado Libre that trains our own large language models. That's about as much as I'll say about it. I also run a small AI consultancy, build a commerce SaaS, do research at a university lab, and I'm still finishing my engineering degree. At any given moment I'm shipping across roughly twenty repositories.

People assume I have a team. I don't. I have a system, and the core of it is not the part everyone talks about. It isn't the agents writing code. It's the memory underneath them. Here is the whole thing, honestly, including what it costs me.

A fair question is how any of this coexists with a full-time job, so let me be straight about it. Mercado Libre is a real one, with a manager, a calendar, and deadlines, and it gets my focused hours. Everything else lives in the margins around it. Early mornings, late nights, weekends, and a server that keeps working when I've closed the laptop.

## Writing code stopped being the hard part

A few years ago the bottleneck was obvious. You had an idea, and then you spent two weeks typing it into existence. That's over. If I ask an agent to design and build a backend, it's basically a solved problem. It gives me a clean architecture, the migrations, the tests, the deploy config, on the first try.

So the bottleneck moved. When the typing is free, what's left is context (keeping a dozen projects coherent in your head), coordination (running many agents without it turning into chaos), and deciding what to build in the first place. Those are the three things my system is actually for. Most of it runs while I'm asleep.

## Layer 1, the things I never touch by hand again

My rule is stupid and it works. If I've done something by hand twice, the third time an agent does it.

I built my own server for this. Something is always running on it. I'm not saying that for effect. I don't go to sleep without agents working, and it feels wrong to leave the machine idle when there's a backlog.

Here is some of what runs on it, around the clock.

- A daily routine wakes up, audits the technical SEO and AI-citation readiness of my SaaS, and opens draft pull requests with content. I review them with coffee.
- [Reacher](https://reacher.sh), an autonomous prospection agent I wrote, goes out and finds clients for the consultancy. It researches companies, drafts the outreach, sends it, and follows up, at about six cents per company. It works the top of my funnel while I work on everything else.
- [wa2vault](https://github.com/frizynn/wa2vault) archives WhatsApp conversations into a vault my agents can read, transcribing voice notes locally. Context I would otherwise lose.
- CLIs I built for the platforms that don't want to be automated, for distribution and reading.
- When I'm in a meeting, I take notes with [Quent](https://quent.live), my real-time meeting copilot, and hand the transcript straight to the agent that's going to act on it.

None of this is glamorous. That's the point. It's the boring layer that buys back hours.

## Layer 2, the amnesia problem

Here's what nobody tells you when you start running agents at scale. More agents without memory is just more chaos, faster.

An agent that forgets everything between sessions turns you into the memory. You become the human clipboard, re-explaining the same decisions, re-pasting the same context, twenty times a day. That doesn't scale. It's exhausting in a way that's hard to name until it stops.

What actually changed how I work was giving my agents persistent memory. Every time I start a new project, a vault is created with it, and my CLAUDE.md and agents.md are wired so the agents write to it without being asked. Decisions, the back and forth, meeting notes, the reasoning behind a choice, all of it lands in plain Markdown. One file per task. The vault is the tracker. Append, never rewrite. (I liked this enough that I turned it into a tool, [trail](https://github.com/frizynn/trail), but the idea matters more than the tool.)

The effect is hard to overstate. An agent I start today picks up a decision I made three weeks ago, in a different repository, and just continues. I stopped being the clipboard.

And because it's Markdown and not code, the context travels. I can zip a vault and hand it to someone who doesn't program, their agent reads it, and they're caught up on months of decisions in minutes. This is the layer nobody posts screenshots of, and it's the whole game.

## Layer 3, how I actually drive them

I barely write code directly anymore. I run goals in Codex and Claude Code, both on their top-tier plans, the 20x ones, and most days I run straight into the usage limits. On top of that I lean on a stack of skills I've collected. One of them, a "thermo-nuclear code quality review" from the Cursor team, runs after every single implementation I do, without exception. I trust it more than I could fully explain to you, and that's a strange place to be. I'll come back to it.

I'll also be honest about what I don't use. I built a multi-agent orchestrator for running many coding agents in parallel. I basically abandoned it. Some tools earn their keep and some don't, and pretending otherwise is how blogs lie to you.

The one habit that matters more than any tool isn't a tool at all. Before I implement anything, I argue with the agent for ten or fifteen minutes about what we're actually doing. Not the code. The shape of the thing. The trade-offs. What we're not going to build. That conversation is the work now. The code is downstream of it.

## The plan is the new code

This is the part I keep turning over.

We've been climbing a ladder of abstraction for seventy years. Assembly, then C, then garbage-collected languages, then frameworks. Each layer let us stop thinking about the one below it. We're on a new rung now. The plan, the spec, the fifteen-minute argument, that is the source code. The actual code is the compiled output. I edit the plan, and the agent compiles it into a repository.

And there will be a layer above this one too. There always is. The interesting question is what it's made of.

Because here's the thing that gives it away. Code is solved, and ideas are not.

## Where the agents still fall on their face

Try this. Ask your agent to design a backend architecture. You'll get something clean and correct, immediately. Now ask it to design the optimal agentic architecture for a problem that doesn't have a standard answer yet. It stumbles. It gives you something plausible and slightly wrong, you go back and forth, and sometimes you never quite get there.

Same model. Wildly different results. Why?

My explanation is that these models are extraordinary interpolators and weak extrapolators. A backend architecture is a dense, massively represented, solved region of the space. There are a million good ones in the training data. The model isn't inventing your backend, it's retrieving the consensus and fitting it to your details. That's interpolation, and it's genuinely superhuman at it.

The optimal agentic architecture is the frontier. The data is sparse, there's no consensus, and for a lot of real problems the right answer doesn't exist in the distribution yet. It has to be invented. That's extrapolation, and extrapolation is a politer word for having a new idea. Models are bad at it for the same reason they're good at the other thing. They're shaped by what already exists.

Which is exactly why the fifteen-minute argument matters. In that conversation, I'm doing the extrapolation and the agent is doing the interpolation. I bring the half-formed new idea, and it brings the entire weight of everything that's already been tried, instantly, so I can test my idea against it. The collaboration works because each of us is doing the part the other one can't. The unsolved problem was never writing the code. It's having the idea worth compiling.

## What it costs

I'd be lying if I ended there, on the clean note about ideas and abstraction.

This life has a price and I pay it daily. I work up to eighteen hours a day. I wake up at eight and I go to sleep at twelve or one, and even then something is running on the server. In a way I love it, and I want to be clear that this isn't discipline talking. It's appetite. I genuinely love building things and solving problems, and given a free evening I'll spend it on a repository.

But the trade is real and I don't want to dress it up. I've lost touch with friends. I've given up the freedom to work out, to go out for a drink when I feel like it, to have a day that isn't accounted for. The agents run while I sleep, but I'm still the one who doesn't really stop.

There's a quieter cost underneath that one. The whole thing is a Jenga tower. Each piece holds up the others. The jobs buy the time and the tools and the runway to keep building, and the building is what might one day make the jobs optional. Pull any single block out and the tower wobbles, so I don't.

And here's what the whole setup finally taught me. I automated the code, the outreach, the memory, the busywork, and I bought back hours. All that did was clear away every excuse until the only hard problem left was the human one. Not what to build. What to commit to. The model can interpolate any backend I describe, but it can't decide what's worth a life. That decision doesn't compile.

So I move fast on ten fronts and call it leverage, and maybe it is. But the same way the model is brilliant at the solved thing and lost at the frontier, I'm brilliant at executing and scared of choosing. Running everything in parallel feels like progress, and it is, but it might also be the most sophisticated way I've found to avoid choosing. The agents gave me memory and time. What I still don't have is the nerve to point all of it at one thing and find out if I was right.

And that was never a code problem, which makes me think it isn't only mine. The tools are making execution free for everyone, so the scarce thing stops being whether you can build something and becomes whether you know what's worth building, and whether you'll commit to it. Capability is about to be everywhere. Direction won't be. And when building costs nothing, staying busy is the easiest place in the world to hide. So if you're running in ten directions and calling it ambition, ask yourself once whether you're fast because you chose, or because you never did. I still owe myself the answer.
