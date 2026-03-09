# Conflict Analysis Framework

When considering whether to split a task into multiple subagents with separate worktrees,
the primary risk is **merge conflicts** when branches are merged back to main.

## Conflict Risk Levels

### LOW RISK — Safe to split into multiple agents

Conditions (ALL must be true):
- Each agent's file scope is **completely disjoint** (no shared files)
- Agents do not modify **shared configuration files** (package.json, pyproject.toml, etc.)
- Agents do not add **imports or dependencies** that the other agent also adds
- No agent modifies **shared types/interfaces** that another agent consumes

Examples of low-risk splits:
- Backend API (src/api/) vs Frontend UI (src/components/) vs Tests (tests/)
- Module A (src/auth/) vs Module B (src/billing/) when they have no shared imports
- Code writing (src/) vs Documentation (docs/) vs CI config (.github/)

### MEDIUM RISK — Split with caution, add guidelines

Conditions:
- File scopes are mostly disjoint but share 1-2 common files
- Shared files are **additive only** (e.g., adding new entries to a registry, new routes to a router)

Mitigation strategies:
- Instruct agents to APPEND to shared files, never reorganize them
- Have one agent handle the shared file, others only touch their own scope
- Plan merge order: merge the agent that modifies shared files LAST

Examples:
- Two feature modules that both need to register routes in a central router
- Multiple agents that each add new test files but share a test config

### HIGH RISK — Do NOT split, use single agent

Conditions (ANY one is enough):
- Agents would modify the **same files** (especially the same functions/classes)
- Task involves **refactoring** that changes function signatures, class hierarchies, or module structure
- Agents share **state or data models** that might evolve during the task
- Task has **sequential dependencies** where agent B needs agent A's output
- Changes involve **cross-cutting concerns** (logging, error handling, auth middleware)

Examples:
- Refactoring a monolith into services (touches everything)
- Adding a new field to a data model used across frontend and backend
- Changing an API contract that both client and server depend on
- Reorganizing import structure or module boundaries

## Decision Checklist

For each candidate agent pair, answer these questions:

1. List every file Agent A might modify → File Set A
2. List every file Agent B might modify → File Set B
3. Compute intersection: File Set A ∩ File Set B
4. If intersection is empty → LOW RISK
5. If intersection contains only additive-append files → MEDIUM RISK
6. If intersection contains files that need structural changes → HIGH RISK → DO NOT SPLIT

## When in Doubt

**Default to a single agent.** The time cost of resolving merge conflicts,
plus the cognitive overhead of coordinating multiple agents, almost always
exceeds the time saved by parallelization for tasks under 2 hours of agent work.

Splitting is only worth it when:
- Each track is genuinely **1+ hours** of independent work
- File scopes are **provably disjoint**
- The user is comfortable with basic git merge workflow
