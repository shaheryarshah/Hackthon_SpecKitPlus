---
id: 1
title: Update RAG Chatbot Constitution
stage: constitution
date: 2025-12-14
surface: agent
model: Qwen
feature: none
branch: main
user: Shaheryar
command: /sp.constitution
labels: [constitution, rag, chatbot, physical-ai]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
 - .specify/templates/plan-template.md
 - .specify/templates/spec-template.md
 - .specify/templates/tasks-template.md
 - QWEN.md
tests:
 - none
---

## Prompt

```
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

You are updating the project constitution at `.specify/memory/constitution.md`. This file is a TEMPLATE containing placeholder tokens in square brackets (e.g. `[PROJECT_NAME]`, `[PRINCIPLE_1_NAME]`). Your job is to (a) collect/derive concrete values, (b) fill the template precisely, and (c) propagate any amendments across dependent artifacts.

Follow this execution flow:

1. Load the existing constitution template at `.specify/memory/constitution.md`.
   - Identify every placeholder token of the form `[ALL_CAPS_IDENTIFIER]`.
   **IMPORTANT**: The user might require less or more principles than the ones used in the template. If a number is specified, respect that - follow the general template. You will update the doc accordingly.

2. Collect/derive values for placeholders:
   - If user input (conversation) supplies a value, use it.
   - Otherwise infer from existing repo context (README, docs, prior constitution versions if embedded).
   - For governance dates: `RATIFICATION_DATE` is the original adoption date (if unknown ask or mark TODO), `LAST_AMENDED_DATE` is today if changes are made, otherwise keep previous.
   - `CONSTITUTION_VERSION` must increment according to semantic versioning rules:
     - MAJOR: Backward incompatible governance/principle removals or redefinitions.
     - MINOR: New principle/section added or materially expanded guidance.
     - PATCH: Clarifications, wording, typo fixes, non-semantic refinements.
   - If version bump type ambiguous, propose reasoning before finalizing.

3. Draft the updated constitution content:
   - Replace every placeholder with concrete text (no bracketed tokens left except intentionally retained template slots that the project has chosen not to define yet—explicitly justify any left).
   - Preserve heading hierarchy and comments can be removed once replaced unless they still add clarifying guidance.
   - Ensure each Principle section: succinct name line, paragraph (or bullet list) capturing non‑negotiable rules, explicit rationale if not obvious.
   - Ensure Governance section lists amendment procedure, versioning policy, and compliance review expectations.

4. Consistency propagation checklist (convert prior checklist into active validations):
   - Read `.specify/templates/plan-template.md` and ensure any "Constitution Check" or rules align with updated principles.
   - Read `.specify/templates/spec-template.md` for scope/requirements alignment—update if constitution adds/removes mandatory sections or constraints.
   - Read `.specify/templates/tasks-template.md` and ensure task categorization reflects new or removed principle-driven task types (e.g., observability, versioning, testing discipline).
   - Read each command file in `.specify/templates/commands/*.md` (including this one) to verify no outdated references (agent-specific names like CLAUDE only) remain when generic guidance is required.
   - Read any runtime guidance docs (e.g., `README.md`, `docs/quickstart.md`, or agent-specific guidance files if present). Update references to principles changed.

5. Produce a Sync Impact Report (prepend as an HTML comment at top of the constitution file after update):
   - Version change: old → new
   - List of modified principles (old title → new title if renamed)
   - Added sections
   - Removed sections
   - Templates requiring updates (✅ updated / ⚠ pending) with file paths
   - Follow-up TODOs if any placeholders intentionally deferred.

6. Validation before final output:
   - No remaining unexplained bracket tokens.
   - Version line matches report.
   - Dates ISO format YYYY-MM-DD.
   - Principles are declarative, testable, and free of vague language ("should" → replace with MUST/SHOULD rationale where appropriate).

7. Write the completed constitution back to `.specify/memory/constitution.md` (overwrite).

8. Output a final summary to the user with:
   - New version and bump rationale.
   - Any files flagged for manual follow-up.
   - Suggested commit message (e.g., `docs: amend constitution to vX.Y.Z (principle additions + governance update)`).

Formatting & Style Requirements:

- Use Markdown headings exactly as in the template (do not demote/promote levels).
- Wrap long rationale lines to keep readability (<100 chars ideally) but do not hard enforce with awkward breaks.
- Keep a single blank line between sections.
- Avoid trailing whitespace.

If the user supplies partial updates (e.g., only one principle revision), still perform validation and version decision steps.

If critical info missing (e.g., ratification date truly unknown), insert `TODO(<FIELD_NAME>): explanation` and include in the Sync Impact Report under deferred items.

Do not create a new template; always operate on the existing `.specify/memory/constitution.md` file.

---

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage:
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.


/sp.constitution # /sp.constitution

**Project:** Integrated RAG Chatbot for *Physical AI & Humanoid Robotics* (Vercel-deployed book)

## Core principles

* **Faithful grounding:** Answers must be grounded strictly in the book corpus unless explicitly permitted (e.g., UX/system messages).
* **Selectable-context fidelity:** When a user selects text, responses are constrained to that selection only.
* **Citations-first UX:** Every answer includes inline citations (section/page/anchor) back to the book.
* **Low-latency & reliability:** Sub-2s p95 for retrieval + generation under normal load.
* **Security by default:** No secrets in client; least-privilege access; rotation-ready.
* **Non-destructive integration:** Existing book content, routing, SEO, and assets remain unchanged.

## Key standards

* Retrieval uses semantic + structural signals (headings, sections, anchors).
* Deterministic chunking with stable IDs for re-indexing.
* Versioned indexes aligned to book releases.
* Observability: request tracing, retrieval hit-rate, citation coverage.

## Constraints

* Frontend embedded within the existing Vercel site (no iframe that blocks SEO).
* Backend via FastAPI (serverless-compatible).
* Vector store: Qdrant Cloud Free Tier.
* Metadata store: Neon Serverless Postgres.
* Agents/UI: OpenAI Agents / ChatKit SDKs.

## Success criteria

* ≥95% answers include valid citations.
* Selection-only mode never leaks content outside the selection.
* Zero regression to existing book pages.
* Passes security review (no client-side secrets).

---

# /sp.specify

## Target users

* Readers of *Physical AI & Humanoid Robotics* seeking precise, cited answers.
* Educators/researchers using section-level queries.

## Functional requirements

1. **Ask-the-book:** Free-form questions answered from the full book.
2. **Ask-the-selection:** User-highlighted text constrains retrieval strictly to that span.
3. **Citations:** Clickable anchors to sections/headers.
4. **Streaming responses:** Token streaming with interim citations.
5. **Feedback loop:** Thumbs up/down stored for evaluation.

## Non-functional requirements

* p95 latency ≤ 2s (warm).
* Availability ≥ 99.5%.
* Accessibility: keyboard navigation, ARIA labels.

## Data & indexing

* Source: Docusaurus markdown/MDX build artifacts.
* Chunking: 300–600 tokens with header-aware boundaries.
* Metadata: book_version, chapter, section, anchor, char_start/end.

## Architecture

* **Frontend:** Book UI + ChatKit widget.
* **API:** FastAPI (RAG orchestration, auth, rate-limit).
* **Retrieval:** Qdrant (vectors + metadata filters).
* **Relational:** Neon (sessions, feedback, versions).
* **LLM/Agents:** OpenAI Agents (tooling for retrieve/cite).

## Security & privacy

* Secrets only on server; env-based config.
* Row-level constraints for feedback data.
* Rate limiting per IP/session.

## Out of scope

* Web-wide search beyond the book.
* Training/fine-tuning models.
* Account systems beyond anonymous sessions.

---

# /sp.plan

## Phase 0 — Prep

* Rotate exposed credentials and move to environment variables.
* Snapshot current Vercel build for rollback.

## Phase 1 — Corpus & Index

* Export MD/MDX → normalized text.
* Header-aware chunking + stable IDs.
* Embed and upsert to Qdrant with metadata.
* Persist index manifest in Neon.

## Phase 2 — Backend (FastAPI)

* Endpoints: /query, /query-selection, /feedback, /health.
* Retrieval with metadata filters (selection mode uses char bounds).
* Agent prompt enforcing citations-only answers.

## Phase 3 — Frontend Embed

* ChatKit widget with two modes (Book / Selection).
* Text selection hook passes anchors + char ranges.
* Citation rendering with deep links.

## Phase 4 — Quality & Testing

* Unit: chunking, filters, citation mapping.
* Integration: selection leakage tests.
* Load: p95 latency checks.

## Phase 5 — Deploy & Observe

* Vercel API routes or serverless FastAPI.
* Metrics: hit-rate, citation coverage, latency.

## Decisions & tradeoffs

* Chunk size vs. recall.
* Serverless cold starts vs. always-on.
* Free-tier limits vs. batching.

## Acceptance checks

* Selection-only constraint verified.
* Citation links valid.
* No content regressions on book pages.

## Response snapshot

I'll help you create or update the project constitution based on the provided principles. Let me start by checking if a constitution template already exists.

...

[The rest of the response to the user's request]

## Outcome

- ✅ Impact: Updated constitution for RAG chatbot project with 9 core principles; updated dependent templates to align with new principles
- 🧪 Tests: none
- 📁 Files: .specify/memory/constitution.md, .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md, QWEN.md
- 🔁 Next prompts: /sp.specify, /sp.plan, /sp.tasks
- 🧠 Reflection: The constitution update was comprehensive, changing from textbook principles to RAG chatbot-specific principles with proper governance and versioning.

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): none
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Create feature specification based on updated constitution
