---
id: 1
title: Update Project Constitution
stage: constitution
date: 2025-12-05
surface: agent
model: Claude Code
feature: Project Constitution
branch: main
user: Shaheryar
command: .sp.constitution
labels: [constitution, governance, specification]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - 
---

## Prompt

/sp.constitution Project: Textbook for Physical AI & Humanoid Robotics, created using Spec-Kit Plus, Claude Code, and deployed using Docusaurus on GitHub Pages

Core Principles

Technical Accuracy & Engineering Rigor

Every robotics, AI, control theory, or mechanical systems concept must follow authoritative standards (IEEE, ROS, ISO robotics standards, academic sources).

Mathematical models must be verified (dynamics, kinematics, control, simulation).

Clarity for Multidisciplinary Learners

Written for students with backgrounds in:

Computer Science

Mechatronics

AI/ML

Electrical/Mechanical Engineering

Explanations must include diagrams, equations, code samples, and step-by-step learning pathways.

Project-Based Learning Orientation

Every chapter must include hands-on exercises, labs, ROS simulation tasks, or humanoid robotics projects.

Physical AI concepts should be demonstrated through experiments (simulated or hardware-based).

AI-Native Book Development

All text, code, and structure must be created collaboratively using Claude Code + Spec-Kit Plus.

Maintain AI-generated traceability for every major section.

Key Standards
Content Standards

Minimum 10 Chapters, covering:

Foundations of Physical AI

Fundamentals of Humanoid Robotics

Kinematics & Dynamics

Actuators, Sensors, and Embodiment

Control Systems for Humanoids

Reinforcement Learning & Embodied Intelligence

Vision & Multimodal Perception

Simulation (Isaac Gym, Mujoco, PyBullet, ROS Gazebo)

Human-Robot Interaction & Safety

Future Directions in Physical AI

Each chapter must contain:

At least 1 diagram

At least 1 code example (Python, ROS, or simulation code)

A project or lab section

Summary + key vocabulary list

Citation & Verification Standards

All factual claims must be traceable to sources.

Citation style: IEEE or APA (author choice but consistent)

Use peer-reviewed or authoritative robotics literature when possible.

Minimum 20 citations across the book.

Software & Deployment Standards

Book must be created with Docusaurus v3.

Deployment target: GitHub Pages (gh-pages branch).

Include:

Table of Contents

Search

Sidebar Navigation

Auto-generated API/code blocks

Build scripts must run with standard Node.js LTS.

Spec-Kit + Claude Code Workflow

Each chapter must be generated using clear Spec-Kit prompts.

Versioned content via Git.

Every major rewrite documented via commit message referencing the spec section.

Constraints

Word count:

Full textbook: 15,000 – 25,000 words

Each chapter: 1,200 – 2,500 words

File Structure:

/docs/ for all chapters

/static/ for images

/spec/ for Spec-Kit prompts and constitutions

/src/ for custom components

Media Requirements:

Diagrams: SVG or PNG

Code examples: embedded fenced blocks

All images must be open-licensed or originally generated

Plagiarism:

0% tolerance

All text must be original or properly cited.

Success Criteria

Technical Quality

Book accurately explains foundational and advanced robotics and AI concepts.

All formulas checked for correctness.

All code executes or is simulation-valid.

Pedagogical Quality

Clear learning path from beginner → advanced topics.

Projects encourage hands-on experimentation.

Spec & AI Compliance

All chapters produced according to Spec-Kit Plus workflows.

All content traceable to AI-assisted generation logs.

Deployment Success

Fully functional Docusaurus site publicly available on GitHub Pages.

Navigation, styling, and search work correctly.

Review & Validation

At least one peer (student or mentor) reviews each chapter.

Book passes fact-checking & technical validation.

## Response snapshot

I've successfully created an updated constitution for the "Textbook for Physical AI & Humanoid Robotics" project based on your specifications. Here's an overview of what was done:

1. **Constitution Template Update**:
   - Filled in the project name: "Textbook for Physical AI & Humanoid Robotics"
   - Added the 4 core principles you specified:
     - Technical Accuracy & Engineering Rigor
     - Clarity for Multidisciplinary Learners
     - Project-Based Learning Orientation
     - AI-Native Book Development
   - Added 2 additional principles with your specific requirements:
     - Content Standards (with 10 chapter topics)
     - Citation & Verification Standards (IEEE/APA style, minimum 20 citations)
   - Included supporting sections:
     - Software & Deployment Standards (Docusaurus v3, GitHub Pages)
     - Spec-Kit + Claude Code Workflow (versioned content via Git)
   - Added governance section referencing all constraints and success criteria
   - Set version to 1.0.0 with ratification and amendment dates as today (2025-12-05)

2. **Sync Impact Report**:
   - Included an HTML comment at the top of the constitution file detailing:
     - Version change: n/a → 1.0.0
     - Added sections: All the principles and standards you specified
     - Templates requiring updates: .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md, and QWEN.md

3. **Template Checks**:
   - Verified that the plan, spec, and tasks templates refer to the constitution for guidance
   - Confirmed that these templates would need updates to align with the new project-specific principles

The updated constitution file has been saved to `.specify/memory/constitution.md` and now fully reflects the requirements for your Physical AI & Humanoid Robotics textbook project.

## Outcome

- ✅ Impact: Created new project constitution with 6 principles and governance structure
- 🧪 Tests: No automated tests needed for constitution file
- 📁 Files: Updated .specify/memory/constitution.md with full content
- 🔁 Next prompts: Ready to create feature specs based on this constitution
- 🧠 Reflection: The constitution establishes clear standards for the textbook project

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): All validation checks passed
- Prompt variant (if applicable): Standard constitution update prompt
- Next experiment (smallest change to try): Begin creating chapter specifications based on constitution