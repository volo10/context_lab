# Product Requirements Document (PRD)
## Context Window Impact Analysis Lab

**Version:** 1.0.0
**Author:** Boris Volovelsky
**Date:** December 2025
**Course:** M.Sc. Computer Science - LLM Context Windows Lab

---

## 1. Executive Summary

### 1.1 Problem Statement
Large Language Models (LLMs) have finite context windows that limit the amount of information they can process simultaneously. This leads to several challenges:
- **Lost in the Middle**: Information placed in the middle of long contexts is often missed
- **Context Accumulation**: As context grows, accuracy decreases and latency increases
- **Inefficient Token Usage**: Sending full context wastes tokens and increases costs

### 1.2 Solution Overview
A Python-based experimental framework that demonstrates and measures these phenomena, and evaluates mitigation strategies including RAG (Retrieval-Augmented Generation) and context engineering techniques.

### 1.3 Success Metrics (KPIs)
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Experiment Coverage | 4/4 experiments | All experiments run successfully |
| Test Coverage | ≥70% | pytest-cov report |
| RAG vs Full Context | RAG faster & more accurate | Experiment 3 metrics |
| Documentation | Complete | All required docs present |

---

## 2. Stakeholders

| Role | Name/Group | Interest |
|------|------------|----------|
| Primary User | M.Sc. Students | Learn context window phenomena |
| Instructor | Dr. Yoram Segal | Evaluate lab completion |
| Developers | Context Lab Team | Maintain and extend the framework |

---

## 3. Functional Requirements

### 3.1 Experiment 1: Needle in Haystack (Lost in the Middle)
**Priority:** High
**Description:** Demonstrate that facts placed in the middle of context are harder to retrieve than those at the start or end.

**Acceptance Criteria:**
- [x] Generate synthetic documents with embedded facts
- [x] Test fact retrieval at start, middle, and end positions
- [x] Measure accuracy by position
- [x] Generate accuracy-by-position visualization

### 3.2 Experiment 2: Context Window Size Impact
**Priority:** High
**Description:** Analyze how increasing context size affects model accuracy and latency.

**Acceptance Criteria:**
- [x] Test with document counts: [2, 5, 10, 20, 50]
- [x] Measure accuracy, latency, and token usage
- [x] Generate latency vs. size visualization

### 3.3 Experiment 3: RAG vs Full Context
**Priority:** High
**Description:** Compare RAG retrieval approach against providing full context.

**Acceptance Criteria:**
- [x] Use 20 Hebrew documents across medical, tech, legal domains
- [x] Implement chunking and embedding pipeline
- [x] Use ChromaDB for vector storage
- [x] Compare accuracy, latency, and token usage
- [x] Generate comparison visualization

### 3.4 Experiment 4: Context Engineering Strategies
**Priority:** High
**Description:** Benchmark strategies for managing context in multi-step conversations.

**Acceptance Criteria:**
- [x] Implement SELECT strategy (RAG on history)
- [x] Implement COMPRESS strategy (summarization)
- [x] Implement WRITE strategy (external memory)
- [x] Implement ISOLATE strategy (compartmentalization)
- [x] Measure latency, accuracy, and token usage per strategy

---

## 4. Non-Functional Requirements

### 4.1 Performance
- Simulation mode: Complete all experiments in < 60 seconds
- Real LLM mode: Complete all experiments in < 15 minutes

### 4.2 Compatibility
- Python 3.8+
- Windows, macOS, Linux
- Optional: Ollama for real LLM testing

### 4.3 Usability
- CLI interface with --experiment flag
- Sensible defaults for all parameters
- Clear error messages and troubleshooting guidance

### 4.4 Maintainability
- Modular code structure
- Comprehensive docstrings
- 70%+ test coverage

---

## 5. User Stories

### US-1: Run All Experiments
**As a** student
**I want to** run all experiments with a single command
**So that** I can generate complete results for submission

**Acceptance:** `python context_lab.py` runs all 4 experiments

### US-2: Run Specific Experiment
**As a** student
**I want to** run a specific experiment
**So that** I can focus on one phenomenon at a time

**Acceptance:** `python context_lab.py --experiment 3` runs only experiment 3

### US-3: Generate Visualizations
**As a** student
**I want to** generate publication-quality plots
**So that** I can include them in my lab report

**Acceptance:** `python visualize.py` generates all required plots

### US-4: Extend with Custom Strategy
**As a** developer
**I want to** add a new context strategy
**So that** I can test novel approaches

**Acceptance:** Subclass `ContextStrategy` and add to experiment 4

---

## 6. Constraints and Dependencies

### 6.1 Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20.0 | Numerical operations |
| pandas | ≥1.3.0 | Data analysis |
| matplotlib | ≥3.4.0 | Visualization |
| seaborn | ≥0.11.0 | Statistical plots |
| langchain | ≥0.1.0 | LLM framework (optional) |
| chromadb | ≥0.4.0 | Vector database (optional) |

### 6.2 Constraints
- Ollama must be running for real LLM mode
- Internet connection required for downloading embedding models

---

## 7. Out of Scope

The following are explicitly NOT included in this lab:
- Production-ready API deployment
- Multi-user support
- Real-time streaming responses
- Custom model fine-tuning
- Cloud deployment automation

---

## 8. Milestones and Deliverables

| Milestone | Deliverables | Status |
|-----------|--------------|--------|
| M1: Core Implementation | Experiments 1-4 | ✅ Complete |
| M2: Testing | 60+ unit tests, 70%+ coverage | ✅ Complete |
| M3: Documentation | README, PRD, Architecture | ✅ Complete |
| M4: Visualization | All experiment plots | ✅ Complete |

---

## 9. Appendix

### 9.1 References
- [Lost in the Middle Paper](https://arxiv.org/abs/2307.03172)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### 9.2 Glossary
| Term | Definition |
|------|------------|
| LLM | Large Language Model |
| RAG | Retrieval-Augmented Generation |
| Context Window | Maximum tokens an LLM can process |
| Embedding | Vector representation of text |
