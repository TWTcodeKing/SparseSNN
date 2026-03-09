# skill-finder

A meta-skill that systematically searches all major agent skill registries to find high-quality open-source skills before you build your own.

## What it does

When you need a skill, this skill guides Claude through a structured search funnel across 12+ platforms (270K+ indexed skills total), from Anthropic's official repo down to raw GitHub search, evaluating quality and security at each step.

## Covered Platforms

| Tier | Platforms |
|------|-----------|
| 1 - Official | Anthropic Skills, Anthropic Plugins Registry, awesome-claude-skills, AgentSkills.io |
| 2 - Scored | SkillHub (AI quality scoring) |
| 3 - Broad | SkillsMP, OneSkill, ClawHub, CCPM, MCPServers.org, OpenClawSkill, LobeHub |
| 4 - GitHub | Direct GitHub search with SKILL.md path queries |
| 5 - Adjacent | MCP servers, Cursor rules, n8n workflows |

## Install

**Claude Code (personal):**
```bash
mkdir -p ~/.claude/skills/skill-finder
cp SKILL.md ~/.claude/skills/skill-finder/
```

**Claude Code (project):**
```bash
mkdir -p .claude/skills/skill-finder
cp SKILL.md .claude/skills/skill-finder/
```

**Claude.ai web:** Upload SKILL.md as an attachment and ask Claude to follow it.

## Usage

Just ask naturally:
- "Is there a skill for generating API docs?"
- "Find me a code review skill"
- "I need to build a deployment skill — check if one exists first"

Or invoke directly in Claude Code: `/skill-finder`

## License

MIT
