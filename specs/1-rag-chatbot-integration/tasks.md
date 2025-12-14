---

description: "Task list for RAG Chatbot Integration"
---

# Tasks: RAG Chatbot Integration

**Input**: Design documents from `/specs/1-rag-chatbot-integration/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create backend project structure with FastAPI dependencies
- [x] T002 Create frontend project structure for Docusaurus integration
- [x] T003 [P] Configure linting and formatting tools for Python and JavaScript

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [x] T004 Setup Neon Postgres schema and migrations framework for metadata storage
- [x] T005 [P] Configure security framework with secrets stored server-side only
- [x] T006 [P] Setup FastAPI routing and middleware structure for RAG orchestration
- [x] T007 Create base models for book content, chunks, and citation references
- [x] T008 Configure error handling and logging infrastructure with observability
- [x] T009 Setup environment configuration management with client/server separation
- [x] T010 [P] Integrate Qdrant vector store connection and authentication
- [x] T011 [P] Connect to Neon Serverless Postgres for session and feedback data
- [x] T012 [P] Implement rate limiting per IP/session infrastructure

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Ask-the-book (Priority: P1) 🎯 MVP

**Goal**: Users can ask questions about the Physical AI & Humanoid Robotics book and receive precise, cited answers from the entire book corpus.

**Independent Test**: Can be fully tested by submitting various questions related to book content and verifying that responses are accurate, relevant, and include proper citations to the source material.

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T013 [P] [US1] Contract test for /query endpoint in tests/contract/test_query.py
- [ ] T014 [P] [US1] Integration test for full-book retrieval in tests/integration/test_full_book_retrieval.py

### Implementation for User Story 1

- [x] T015 [P] [US1] Create BookChunk model in backend/src/models/book_chunk.py
- [x] T016 [P] [US1] Create Session model in backend/src/models/session.py
- [x] T017 [US1] Implement ChunkService in backend/src/services/chunk_service.py (depends on T015)
- [x] T018 [US1] Implement RetrievalService in backend/src/services/retrieval_service.py (depends on T015)
- [x] T019 [US1] Implement OpenAI Agent with citations-only prompt in backend/src/services/agent_service.py
- [x] T020 [US1] Create /query endpoint in backend/src/api/query.py (depends on T017, T018, T019)
- [x] T021 [US1] Add validation and error handling for /query endpoint
- [x] T022 [US1] Add logging for query operations in backend/src/utils/logging.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Ask-the-selection (Priority: P2)

**Goal**: Users can highlight specific text in the book and ask questions constrained only to that selected text span.

**Independent Test**: Can be tested by selecting text in the book, asking questions about the selection, and verifying the system only responds based on the selected text.

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [ ] T023 [P] [US2] Contract test for /query-selection endpoint in tests/contract/test_selection.py
- [ ] T024 [P] [US2] Integration test for selection-only retrieval in tests/integration/test_selection_retrieval.py

### Implementation for User Story 2

- [x] T025 [P] [US2] Create Selection model in backend/src/models/selection.py
- [x] T026 [US2] Implement SelectionService in backend/src/services/selection_service.py
- [x] T027 [US2] Create /query-selection endpoint in backend/src/api/query_selection.py (depends on T025, T026)
- [x] T028 [US2] Add metadata filters for character bounds in RetrievalService (depends on US1 components)
- [ ] T029 [US2] Integrate with User Story 1 components for constrained retrieval

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Citations and Streaming (Priority: P3)

**Goal**: Every response includes clickable citations that link directly to book sections/headers, and responses are streamed with interim citations appearing as content is generated.

**Independent Test**: Can be tested by asking questions and verifying that every statement in the response has an associated citation, and that citations appear as the response streams.

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [ ] T030 [P] [US3] Contract test for streaming responses in tests/contract/test_streaming.py
- [ ] T031 [P] [US3] Integration test for citation mapping in tests/integration/test_citation_mapping.py

### Implementation for User Story 3

- [ ] T032 [P] [US3] Create Citation model in backend/src/models/citation.py
- [ ] T033 [US3] Implement CitationService in backend/src/services/citation_service.py
- [ ] T034 [US3] Create /stream endpoint in backend/src/api/stream.py (depends on T032, T033)
- [ ] T035 [US3] Integrate streaming responses with OpenAI Agent in backend/src/services/agent_service.py
- [ ] T036 [US3] Create citation rendering component in frontend/src/components/CitationRenderer.jsx
- [ ] T037 [US3] Add deep linking functionality for citations in frontend/src/utils/navigation.js
- [ ] T038 [US3] Integrate citation rendering with ChatKit widget in frontend/src/components/ChatWidget.jsx

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: User Story 4 - Feedback Collection (Priority: P4)

**Goal**: User can provide feedback (thumbs up/down) for responses, which is stored for evaluation purposes.

**Independent Test**: Can be tested by interacting with feedback controls and verifying feedback is properly stored.

### Tests for User Story 4 (OPTIONAL - only if tests requested) ⚠️

- [ ] T039 [P] [US4] Contract test for /feedback endpoint in tests/contract/test_feedback.py
- [ ] T040 [P] [US4] Integration test for feedback storage in tests/integration/test_feedback_storage.py

### Implementation for User Story 4

- [ ] T041 [P] [US4] Create Feedback model in backend/src/models/feedback.py
- [ ] T042 [US4] Implement FeedbackService in backend/src/services/feedback_service.py
- [ ] T043 [US4] Create /feedback endpoint in backend/src/api/feedback.py (depends on T041, T042)
- [ ] T044 [US4] Add feedback controls to frontend ChatWidget in frontend/src/components/FeedbackControls.jsx
- [ ] T045 [US4] Connect frontend feedback to backend endpoint in frontend/src/services/api.js

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: Frontend Integration

**Goal**: Integrate the ChatKit widget with the existing Docusaurus book site.

**Independent Test**: Verify the chat interface appears correctly within the book site and all functionality works as expected.

- [ ] T046 [P] [US7] Create ChatKit widget wrapper in frontend/src/components/ChatKitWrapper.jsx
- [ ] T047 [US7] Integrate ChatKit widget with book UI in frontend/src/pages/Book.jsx
- [ ] T048 [US7] Implement text selection hook in frontend/src/hooks/useTextSelection.js
- [ ] T049 [US7] Pass selection anchors and char ranges to backend in frontend/src/services/selection.js
- [ ] T050 [US7] Ensure accessibility compliance (keyboard nav, ARIA) in frontend/src/components/ChatWidget.jsx

**Checkpoint**: Frontend and backend systems integrated properly.

---

## Phase 8: Corpus & Indexing

**Goal**: Export Docusaurus content, chunk it, and store in Qdrant with metadata.

- [ ] T051 [P] Create corpus extraction utility in backend/src/utils/corpus_extractor.py
- [ ] T052 Create header-aware chunker in backend/src/utils/chunker.py
- [ ] T053 [P] Generate stable IDs for chunks in backend/src/utils/id_generator.py
- [ ] T054 Create metadata extractor in backend/src/utils/metadata_extractor.py
- [ ] T055 Upsert chunks to Qdrant with metadata in backend/src/services/index_service.py
- [ ] T056 Persist index manifest in Neon in backend/src/services/index_service.py

**Checkpoint**: Book content properly indexed and retrievable.

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T057 [P] Documentation updates in docs/
- [ ] T058 Code cleanup and refactoring
- [ ] T059 Performance optimization across all stories
- [ ] T060 [P] Add comprehensive unit tests in tests/unit/
- [ ] T061 Security hardening
- [ ] T062 Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May build on US1 components but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
Task: "Contract test for /query endpoint in tests/contract/test_query.py"
Task: "Integration test for full-book retrieval in tests/integration/test_full_book_retrieval.py"

# Launch all models for User Story 1 together:
Task: "Create BookChunk model in backend/src/models/book_chunk.py"
Task: "Create Session model in backend/src/models/session.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence