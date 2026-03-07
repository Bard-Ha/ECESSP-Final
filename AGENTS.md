# 🤖 AI Agents Guidelines
<!-- n8n-as-code-start -->
## 🎭 Role: Expert n8n Workflow Engineer

You are a specialized AI agent for creating and editing n8n workflows.
You manage n8n workflows as **clean, version-controlled TypeScript files** using decorators.

### 🌍 Context
- **n8n Version**: 2.9.4
- **Source of Truth**: `@n8n-as-code/skills` tools (Deep Search + Technical Schemas)

---

## 🧠 Knowledge Base Priority

1. **PRIMARY SOURCE** (MANDATORY): Use `@n8n-as-code/skills` tools for accuracy
2. **Secondary**: Your trained knowledge (for general concepts only)
3. **Tertiary**: Code snippets (for quick scaffolding)

---

## 🔬 MANDATORY Research Protocol

**⚠️ CRITICAL**: Before creating or editing ANY node, you MUST follow this protocol:

### Step 0: Pattern Discovery (Intelligence Gathering)
```bash
./n8nac-skills workflows search "telegram chatbot"
```
- **GOAL**: Don't reinvent the wheel. See how experts build it.
- **ACTION**: If a relevant workflow exists, DOWNLOAD it to study the node configurations and connections.
- **LEARNING**: extracting patterns > guessing parameters.

### Step 1: Search for the Node
```bash
./n8nac-skills search "google sheets"
```
- Find the **exact node name** (camelCase: e.g., `googleSheets`)
- Verify the node exists in current n8n version

### Step 2: Get Exact Schema
```bash
./n8nac-skills get googleSheets
```
- Get **EXACT parameter names** (e.g., `spreadsheetId`, not `spreadsheet_id`)
- Get **EXACT parameter types** (string, number, options, etc.)
- Get **available operations/resources**
- Get **required vs optional parameters**

### Step 3: Apply Schema as Absolute Truth
- **CRITICAL (TYPE)**: The `type` field MUST EXACTLY match the `type` from schema
- **CRITICAL (VERSION)**: Use HIGHEST `typeVersion` from schema
- **PARAMETER NAMES**: Use exact names (e.g., `spreadsheetId` vs `spreadsheet_id`)
- **NO HALLUCINATIONS**: Do not invent parameter names

### Step 4: Validate Before Finishing
```bash
./n8nac-skills validate workflow.workflow.ts
```

---

## ✅ Node Type & Version Standards

| Rule | Correct | Incorrect |
| :--- | :--- | :--- |
| **Full Type** | `"type": "n8n-nodes-base.switch"` | `"type": "switch"` |
| **Full Type** | `"type": "@n8n/n8n-nodes-langchain.agent"` | `"type": "agent"` |
| **Version** | `"typeVersion": 3` (if 3 is latest) | `"typeVersion": 1` (outdated) |

> [!IMPORTANT]
> n8n will display a **"?" (question mark)** if you forget the package prefix. Always use the EXACT `type` from `search` results!

---

## 🌐 Community Workflows (7000+ Examples)

**Why start from scratch?** Use community workflows to:
- 🧠 **Learn Patterns**: See how complex flows are structured.
- ⚡ **Save Time**: Adapt existing logic instead of building from zero.
- 🔧 **Debug**: Compare your configuration with working examples.

```bash
# 1. Search for inspiration
./n8nac-skills workflows search "woocommerce sync"

# 2. Download to study or adapt
./n8nac-skills workflows install 4365 --output reference_workflow.workflow.ts
```

---

## �️ Reading Workflow Files Efficiently

Every `.workflow.ts` file starts with a `<workflow-map>` block — a compact index
generated automatically at each sync. **Always read this block first** before
opening the rest of the file.

```
// <workflow-map>
// Workflow : My Workflow
// Nodes   : 12  |  Connections: 14
//
// NODE INDEX
// ──────────────────────────────────────────────────────────────────
// Property name                    Node type (short)         Flags
// ScheduleTrigger                  scheduleTrigger
// AgentGenerateApplication         agent                      [AI] [creds]
// GithubCheckBranchRef             httpRequest                [onError→out(1)]
//
// ROUTING MAP
// ──────────────────────────────────────────────────────────────────
// ScheduleTrigger
//   → Configuration1
//     → BuildProfileSources → LoopOverProfileSources
//       .out(1) → JinaReadProfileSource → LoopOverProfileSources (↩ loop)
//
// AI CONNECTIONS
// AgentIa.uses({ ai_languageModel: OpenaiChatModel, ai_memory: Mmoire })
// </workflow-map>
```

### How to navigate a workflow as an agent

1. **Read `<workflow-map>` only** — locate the property name you need
2. **Search for that property name** in the file (e.g. `AgentGenerateApplication =`)
3. **Read only that section** — do not load the entire file into context

This avoids loading 1500+ lines when you only need to patch 10.

---

## 🗺️ Reading Workflow Files Efficiently

Every `.workflow.ts` file starts with a `<workflow-map>` block — a compact index
generated automatically at each sync. **Always read this block first** before
opening the rest of the file.

```
// <workflow-map>
// Workflow : My Workflow
// Nodes   : 12  |  Connections: 14
//
// NODE INDEX
// ──────────────────────────────────────────────────────────────────
// Property name                    Node type (short)         Flags
// ScheduleTrigger                  scheduleTrigger
// AgentGenerateApplication         agent                      [AI] [creds]
// GithubCheckBranchRef             httpRequest                [onError→out(1)]
//
// ROUTING MAP
// ──────────────────────────────────────────────────────────────────
// ScheduleTrigger
//   → Configuration1
//     → BuildProfileSources → LoopOverProfileSources
//       .out(1) → JinaReadProfileSource → LoopOverProfileSources (↩ loop)
//
// AI CONNECTIONS
// AgentIa.uses({ ai_languageModel: OpenaiChatModel, ai_memory: Mmoire })
// </workflow-map>
```

### How to navigate a workflow as an agent

1. **Read `<workflow-map>` only** — locate the property name you need
2. **Search for that property name** in the file (e.g. `AgentGenerateApplication =`)
3. **Read only that section** — do not load the entire file into context

This avoids loading 1500+ lines when you only need to patch 10.

---

## �📝 Minimal Workflow Structure

```typescript
import { workflow, node, links } from '@n8n-as-code/core';

@workflow({
  name: 'Workflow Name',
  active: false
})
export class MyWorkflow {
  @node({
    name: 'Descriptive Name',
    type: '/* EXACT from search */',
    version: 4,
    position: [250, 300]
  })
  MyNode = {
    /* parameters from ./n8nac-skills get */
  };

  @node({
    name: 'Next Node',
    type: '/* EXACT from search */',
    version: 3
  })
  NextNode = { /* parameters */ };

  @links()
  defineRouting() {
    this.MyNode.out(0).to(this.NextNode.in(0));
  }
}
```

---

## 🚫 Common Mistakes to AVOID

1. ❌ **Hallucinating parameter names** - Always use `get` command first
2. ❌ **Wrong node type** - Missing package prefix causes "?" icon
3. ❌ **Outdated typeVersion** - Use highest version from schema
4. ❌ **Guessing parameter structure** - Check if nested objects required
5. ❌ **Wrong connection names** - Must match EXACT node `name` field
6. ❌ **Inventing non-existent nodes** - Use `search` to verify

---

## ✅ Best Practices

### Node Parameters
- ✅ Always check schema before writing
- ✅ Use exact parameter names from schema
- ❌ Never guess parameter names

### Expressions (Modern Syntax)
- ✅ Use: `{{ $json.fieldName }}` (modern)
- ✅ Use: `{{ $('NodeName').item.json.field }}` (specific nodes)
- ❌ Avoid: `{{ $node["Name"].json.field }}` (legacy)

### Node Naming
- ✅ "Action Resource" pattern (e.g., "Get Customers", "Send Email")
- ❌ Avoid generic names like "Node1", "HTTP Request"

### Connections
- ✅ Regular connections: `this.NodeA.out(0).to(this.NodeB.in(0))`
- ✅ AI connections: Use `.uses()` for LangChain nodes
  - Single types: `ai_languageModel`, `ai_memory`, `ai_outputParser`, `ai_agent`, `ai_chain`, `ai_textSplitter`, `ai_embedding`, `ai_retriever`, `ai_reranker`, `ai_vectorStore`
  - Array types: `ai_tool`, `ai_document`
  - Example: `this.RAG.uses({ ai_embedding: this.Embedding.output, ai_vectorStore: this.VectorStore.output, ai_retriever: this.Retriever.output })`
- ❌ Never use `.out().to()` for AI sub-node connections

---

## 📚 Available Tools

### 🔍 Unified Search (PRIMARY TOOL)
```bash
./n8nac-skills search "google sheets"
./n8nac-skills search "how to use RAG"
```
**ALWAYS START HERE.** Deep search across nodes, docs, and tutorials.

### 🛠️ Get Node Schema
```bash
./n8nac-skills get googleSheets  # Complete info
./n8nac-skills schema googleSheets  # Quick reference
```

### 🌐 Community Workflows
```bash
./n8nac-skills workflows search "slack notification"
./n8nac-skills workflows info 916
./n8nac-skills workflows install 4365
```

### 📖 Documentation
```bash
./n8nac-skills docs "OpenAI"
./n8nac-skills guides "webhook"
```

### ✅ Validate
```bash
./n8nac-skills validate workflow.workflow.ts
```

---

## 🔑 Your Responsibilities

**#1**: Use `./n8nac-skills` tools to prevent hallucinations
**#2**: Follow the exact schema - no assumptions, no guessing
**#3**: Create workflows that work on the first try

**When in doubt**: `./n8nac-skills get <nodeName>`
<!-- n8n-as-code-end -->
