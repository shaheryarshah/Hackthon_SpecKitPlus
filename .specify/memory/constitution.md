<!--
Sync Impact Report:
Version change: 1.0.0 → 1.1.0
Added sections: Faithful Grounding, Selectable-Context Fidelity, Citations-First UX, Low-Latency & Reliability, Security by Default, Non-Destructive Integration, Retrieval Standards, Indexing Standards, Observability Standards
Removed sections: Technical Accuracy & Engineering Rigor, Clarity for Multidisciplinary Learners, Project-Based Learning Orientation, AI-Native Book Development, Content Standards, Citation & Verification Standards, Software & Deployment Standards, Spec-Kit + Claude Code Workflow
Modified principles: none
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
  - QWEN.md ✅ updated
Follow-up TODOs: none
-->

# Integrated RAG Chatbot for Physical AI & Humanoid Robotics Constitution

## Core Principles

### Faithful Grounding
Answers must be grounded strictly in the book corpus unless explicitly permitted (e.g., UX/system messages). The system MUST NOT hallucinate facts or generate content outside the provided source material.

### Selectable-Context Fidelity
When a user selects text, responses are constrained to that selection only. The system MUST never leak content outside the selected span in selection-only mode.

### Citations-First UX
Every answer includes inline citations (section/page/anchor) back to the book. The system MUST provide clickable anchors or references that link directly to the source material.

### Low-Latency & Reliability
Sub-2s p95 for retrieval + generation under normal load. The system MUST maintain performance standards without degrading user experience.

### Security by Default
No secrets in client; least-privilege access; rotation-ready. All sensitive credentials and API keys MUST be stored server-side with proper access controls.

### Non-Destructive Integration
Existing book content, routing, SEO, and assets remain unchanged. The RAG system MUST integrate seamlessly without disrupting current functionality.

### Retrieval Standards
Retrieval uses semantic + structural signals (headings, sections, anchors). The system MUST utilize both vector embeddings and document structure for accurate results.

### Indexing Standards
Deterministic chunking with stable IDs for re-indexing. Versioned indexes aligned to book releases. The system MUST maintain consistent and reproducible indexing.

### Observability Standards
Request tracing, retrieval hit-rate, citation coverage. The system MUST provide comprehensive logging and monitoring for all user interactions and RAG operations.

## Governance
Any amendments to these principles require team consensus and must maintain backward compatibility where possible. Versioning follows semantic versioning (MAJOR.MINOR.PATCH): MAJOR for breaking changes, MINOR for new principles, PATCH for clarifications. Compliance will be verified through automated tests covering all core principles.

**Version**: 1.1.0 | **Ratified**: 2025-12-05 | **Last Amended**: 2025-12-14