# Graph Report - Conext Engineering RAG  (2026-05-16)

## Corpus Check
- 5 files · ~12,318 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 16 nodes · 17 edges · 3 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]

## God Nodes (most connected - your core abstractions)
1. `RouterOutput` - 2 edges
2. `MemoryEntry` - 2 edges
3. `RetrievalChunk` - 2 edges
4. `BudgetZoneReport` - 2 edges
5. `BudgetLog` - 2 edges
6. `TestQuestion` - 2 edges
7. `ScorecardEntry` - 2 edges
8. `AgentState` - 2 edges
9. `Constants, budget caps and model names stay here` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities

### Community 0 - "Community 0"
Cohesion: 0.39
Nodes (8): BaseModel, BudgetLog, BudgetZoneReport, MemoryEntry, RetrievalChunk, RouterOutput, ScorecardEntry, TestQuestion

### Community 1 - "Community 1"
Cohesion: 0.67
Nodes (2): AgentState, TypedDict

### Community 2 - "Community 2"
Cohesion: 1.0
Nodes (1): Constants, budget caps and model names stay here

## Knowledge Gaps
- **1 isolated node(s):** `Constants, budget caps and model names stay here`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 1`** (3 nodes): `AgentState`, `state.py`, `TypedDict`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 2`** (2 nodes): `config.py`, `Constants, budget caps and model names stay here`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `Constants, budget caps and model names stay here` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._