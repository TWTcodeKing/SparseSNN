---
name: subagent-architect
description: >
  Design and generate Claude Code subagent .md configuration files from natural language task descriptions.
  Use this skill whenever the user wants to create a subagent, design an agent workflow, plan parallel
  development tasks, split a project into subagents, or asks "help me create an agent for X".
  Also triggers when the user describes a development task and wants to know whether it should be handled
  by one agent or multiple agents, or when they mention concerns about merge conflicts between worktrees.
  This skill handles the full pipeline: task analysis → decomposition decision → conflict risk assessment →
  subagent .md file generation.
---

# Subagent Architect

A skill for turning natural language task descriptions into well-structured Claude Code subagent `.md` files,
with built-in intelligence for task decomposition and merge conflict risk assessment.

## Workflow Overview

When the user describes a task, follow this pipeline in order:

```
1. Clarify Intent     → Understand what the user wants to accomplish
2. Analyze Scope      → Evaluate task granularity and complexity
3. Decomposition Gate → Decide: single agent or multiple agents?
4. Conflict Analysis  → If multiple agents, assess worktree merge risk
5. Final Decision     → Single agent / multiple agents / forced single due to conflict risk
6. Generate Files     → Output properly formatted subagent .md file(s)
```

Do NOT skip the analysis steps. Even if the user says "just make me an agent", run through
steps 2-4 internally and share your reasoning briefly before generating the file.

---

## Step 1: Clarify Intent

Ask the user (if not already clear) these essentials:

- **What is the task?** What should this agent accomplish?
- **What is the project context?** Language, framework, repo structure if relevant.
- **Any constraints?** Read-only? No bash? Specific model preference? Budget concerns?

If the user has already provided enough context in their message, skip directly to analysis.

---

## Step 2: Analyze Scope

Evaluate the task along these dimensions:

| Dimension | Single Agent Signal | Multi-Agent Signal |
|-----------|--------------------|--------------------|
| File scope | Touches < 5 files or 1 module | Touches 3+ distinct modules/layers |
| Task independence | Steps are sequential/dependent | Steps can run in parallel |
| Expertise needed | One domain (e.g., just backend) | Multiple domains (backend + frontend + tests) |
| Estimated duration | Short (< 30 min of agent work) | Long (1+ hours of agent work) |
| Context demand | Fits in one context window | Would overflow one context window |

---

## Step 3: Decomposition Gate

Based on the scope analysis, make one of three decisions:

**→ SINGLE AGENT**: The task is cohesive, sequential, or touches overlapping files.
Generate one subagent .md file.

**→ MULTIPLE AGENTS (candidate)**: The task has clear parallel tracks with independent file scopes.
Proceed to Step 4 (Conflict Analysis) before confirming.

**→ ASK USER**: The task is ambiguous. Present your analysis and let the user decide.

---

## Step 4: Conflict Risk Analysis

This is the critical step. Before recommending multiple agents, evaluate the merge conflict risk.

Read the reference file at `references/conflict-analysis.md` for the detailed conflict analysis framework.

**The core principle: if merge conflicts are likely, fall back to a single agent.**
Wasted time resolving conflicts almost always exceeds the time saved by parallelization.

After analysis, present a brief summary to the user:

```
## Task Analysis

**Scope:** [brief description]
**Recommendation:** [Single Agent / Multiple Agents]
**Reason:** [1-2 sentences]

[If multiple agents:]
**Conflict Risk:** Low — agents operate on completely separate file sets
**Agent breakdown:**
  1. agent-name-1: [responsibility] → files: src/api/...
  2. agent-name-2: [responsibility] → files: src/ui/...
  3. agent-name-3: [responsibility] → files: tests/...

[If forced single due to conflict risk:]
**Why not split:** Both tracks would modify [shared files], causing merge conflicts.
```

---

## Step 5: Generate Subagent Files

Generate one or more `.md` files following this exact format specification.

### File Format

```markdown
---
name: <kebab-case-name>
description: <when this agent should be invoked — be specific and actionable>
tools: <comma-separated list, or omit to inherit all>
model: <sonnet | opus | haiku | inherit>
isolation: worktree  # include ONLY if this agent is part of a multi-agent setup
skills: <comma-separated skill names, if needed>
---

<system prompt: role definition, rules, constraints, output expectations>
```

### Field Guidelines

**name**: Use kebab-case. Keep it descriptive but short (2-4 words).
Good: `api-developer`, `test-writer`, `security-reviewer`
Bad: `agent1`, `my-agent`, `do-stuff`

**description**: This is how Claude decides when to delegate to this agent.
Write it as a clear trigger condition, not a vague summary.
Good: "Implement and modify REST API endpoints in src/api/ following project conventions"
Bad: "Helps with backend stuff"

**tools**: Choose the minimum set needed. This is a security boundary.

| Role Pattern | Recommended Tools |
|-------------|------------------|
| Read-only reviewer/auditor | `Read, Grep, Glob` |
| Researcher with web access | `Read, Grep, Glob, WebFetch, WebSearch` |
| Code writer/developer | `Read, Write, Edit, Bash, Glob, Grep` |
| Documentation writer | `Read, Write, Edit, Glob, Grep` |
| Full access (inherits all) | _(omit the tools field entirely)_ |

**model**: Match complexity to cost.

| Task Type | Recommended Model |
|-----------|------------------|
| Simple/repetitive tasks, linting, formatting | `haiku` |
| Standard development, testing, review | `sonnet` |
| Complex reasoning, architecture, writing | `opus` |
| Match parent conversation | `inherit` |

**isolation: worktree**: Add this ONLY when the agent is part of a multi-agent parallel setup
where file isolation is needed. Do NOT add for single agents or read-only agents.

### System Prompt Guidelines

The body below the frontmatter is the agent's system prompt. Structure it like this:

```markdown
You are a [specific role]. Your job is to [primary responsibility].

## Rules
- [Hard constraint 1]
- [Hard constraint 2]
- [File scope restriction if applicable]

## Approach
[How the agent should tackle tasks — methodology, priorities, patterns to follow]

## Output
[What the agent should produce — format, location, naming conventions]
```

Key principles for the system prompt:
- Be specific about the role — "senior backend engineer specializing in Python FastAPI" beats "developer"
- State file scope restrictions explicitly — "Only modify files under src/api/" prevents wandering
- Include project-specific conventions if the user mentioned them
- Keep it under 50 lines — longer prompts dilute focus

---

## Examples

Read `references/examples.md` for complete examples of single-agent and multi-agent outputs
across different scenarios.

---

## Common Patterns

**Thesis/Research project:**
- experiment-runner (sonnet, full tools, worktree) — runs experiments
- paper-writer (opus, no Bash, worktree) — writes .tex/.md files only
- data-analyst (sonnet, Read + Bash + Grep) — analyzes results, read-only on source code

**Web application:**
- backend-dev (sonnet, full tools, worktree) — src/api/, src/models/
- frontend-dev (sonnet, full tools, worktree) — src/components/, src/pages/
- test-writer (sonnet, full tools, worktree) — tests/

**Code review/audit (always single context, no worktree needed):**
- security-reviewer (opus, Read + Grep + Glob) — finds vulnerabilities
- perf-reviewer (sonnet, Read + Bash + Grep) — profiles and benchmarks

**Refactoring (usually single agent due to high conflict risk):**
- refactorer (opus, full tools) — single agent because refactoring touches many shared files

---

## Output Delivery

After generating the file(s):

1. Show the complete file content to the user
2. Tell the user exactly where to save it:
   - Project-level: `.claude/agents/<name>.md`
   - User-level (all projects): `~/.claude/agents/<name>.md`
3. If multiple agents, provide the launch commands:
   ```bash
   claude --worktree <task-1> --tmux
   claude --worktree <task-2> --tmux
   ```
4. Remind: "You can also just ask Claude to use a specific agent by name, 
   or Claude will auto-delegate based on the description."
