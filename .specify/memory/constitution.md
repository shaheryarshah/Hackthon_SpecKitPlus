<!--
Sync Impact Report:
Version change: n/a → 1.0.0
Added sections: Technical Accuracy & Engineering Rigor, Clarity for Multidisciplinary Learners, Project-Based Learning Orientation, AI-Native Book Development, Content Standards, Citation & Verification Standards, Software & Deployment Standards, Spec-Kit + Claude Code Workflow
Removed sections: none
Modified principles: none
Templates requiring updates:
  - .specify/templates/plan-template.md ⚠ pending (need to ensure Constitution Check aligns with updated principles)
  - .specify/templates/spec-template.md ⚠ pending (need to ensure requirements align with new principles)
  - .specify/templates/tasks-template.md ⚠ pending (need to ensure task categorization reflects new principles)
  - QWEN.md ⚠ pending (may need to reference the new project constitution in the Code Standards section)
Follow-up TODOs: none
-->

# Textbook for Physical AI & Humanoid Robotics Constitution

## Core Principles

### Technical Accuracy & Engineering Rigor
Every robotics, AI, control theory, or mechanical systems concept must follow authoritative standards (IEEE, ROS, ISO robotics standards, academic sources). Mathematical models must be verified (dynamics, kinematics, control, simulation).

### Clarity for Multidisciplinary Learners
Written for students with backgrounds in Computer Science, Mechatronics, AI/ML, Electrical/Mechanical Engineering. Explanations must include diagrams, equations, code samples, and step-by-step learning pathways.

### Project-Based Learning Orientation
Every chapter must include hands-on exercises, labs, ROS simulation tasks, or humanoid robotics projects. Physical AI concepts should be demonstrated through experiments (simulated or hardware-based).

### AI-Native Book Development
All text, code, and structure must be created collaboratively using Claude Code + Spec-Kit Plus. Maintain AI-generated traceability for every major section.

### Content Standards
Minimum 10 Chapters covering Foundations of Physical AI, Fundamentals of Humanoid Robotics, Kinematics & Dynamics, Actuators/Sensors/Embodiment, Control Systems, Reinforcement Learning, Vision & Perception, Simulation, Human-Robot Interaction, and Future Directions. Each chapter must contain diagrams, code examples, projects, and summaries.

### Citation & Verification Standards
All factual claims must be traceable to sources with IEEE or APA citation style. Use peer-reviewed or authoritative robotics literature when possible. Minimum 20 citations across the book.

## Software & Deployment Standards
Book must be created with Docusaurus v3, deployed on GitHub Pages. Include table of contents, search, sidebar navigation, auto-generated API/code blocks. Build scripts must run with standard Node.js LTS.

## Spec-Kit + Claude Code Workflow
Each chapter must be generated using clear Spec-Kit prompts. Versioned content via Git. Every major rewrite documented via commit message referencing the spec section.

## Governance
Content must adhere to all listed constraints (word count, file structure, media requirements, zero-tolerance plagiarism). Success criteria include technical accuracy, pedagogical quality, Spec & AI compliance, and deployment success.

**Version**: 1.0.0 | **Ratified**: 2025-12-05 | **Last Amended**: 2025-12-05