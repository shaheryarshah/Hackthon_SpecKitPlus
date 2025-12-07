# Data Model: Physical AI & Humanoid Robotics Textbook

## Overview

This document describes the key entities and data structures for the Physical AI & Humanoid Robotics textbook project. These entities represent the core concepts that will be covered in the textbook and form the foundation for content organization.

## Entity Definitions

### 1. Textbook

**Description**: The complete educational resource containing chapters, diagrams, code examples, labs, and assessments.

**Attributes**:
- id: Unique identifier for the textbook
- title: "Physical AI & Humanoid Robotics"
- version: Version number following semantic versioning
- totalWords: Word count (between 15,000 and 25,000)
- chapterCount: Number of chapters (minimum 10)
- citationCount: Number of academic citations (minimum 20)
- targetAudience: Array of audience types (CS students, Mechatronics, AI/ML, Engineering)
- prerequisites: Array of required knowledge (Python, linear algebra, calculus, ML basics)
- createdDate: Date of creation
- lastUpdated: Date of last update
- status: Current status (draft, in review, published)

### 2. Chapter

**Description**: Individual sections covering specific topics in Physical AI and humanoid robotics.

**Attributes**:
- id: Unique identifier for the chapter
- textbookId: Reference to the parent textbook
- title: Title of the chapter
- number: Sequential number of the chapter
- wordCount: Number of words in the chapter
- module: The module this chapter belongs to (Physical AI, ROS 2, Simulation, etc.)
- learningOutcomes: Array of learning outcomes for the chapter
- keyTerms: Array of key terms and definitions
- content: Markdown content of the chapter
- diagrams: Array of diagram references
- codeExamples: Array of code example references
- labExercises: Array of lab exercise references
- assessmentQuestions: Array of assessment question references
- citations: Array of citation references
- createdDate: Date of creation
- lastUpdated: Date of last update
- status: Current status (draft, in review, published)

**Relationships**:
- One textbook to many chapters (1:M)
- One chapter to many diagrams (1:M)
- One chapter to many code examples (1:M)
- One chapter to many lab exercises (1:M)
- One chapter to many assessment questions (1:M)

### 3. CodeExample

**Description**: Practical implementations in Python/ROS demonstrating concepts.

**Attributes**:
- id: Unique identifier for the code example
- chapterId: Reference to the parent chapter
- title: Title of the code example
- description: Brief description of what the code example demonstrates
- language: Programming language (Python, ROS 2 specific)
- code: The actual code content in text format
- dependencies: Array of required dependencies or packages
- expectedOutput: Description of expected output or behavior
- tags: Array of tags for categorization (ROS, Python, Navigation, etc.)
- createdDate: Date of creation
- lastUpdated: Date of last update
- verified: Boolean indicating if the code has been verified to work

**Relationships**:
- One chapter to many code examples (1:M)

### 4. Diagram

**Description**: Visual representations that illustrate key robotics concepts.

**Attributes**:
- id: Unique identifier for the diagram
- chapterId: Reference to the parent chapter
- title: Title of the diagram
- description: Brief description of what the diagram illustrates
- fileName: Name of the diagram file
- filePath: Path to the diagram file in the static assets
- format: File format (SVG, PNG, etc.)
- altText: Alternative text for accessibility
- tags: Array of tags for categorization (ROS, Kinematics, Perception, etc.)
- createdDate: Date of creation
- lastUpdated: Date of last update
- creator: Tool or person who created the diagram

**Relationships**:
- One chapter to many diagrams (1:M)

### 5. LabExercise

**Description**: Hands-on activities for students to practice concepts.

**Attributes**:
- id: Unique identifier for the lab exercise
- chapterId: Reference to the parent chapter
- title: Title of the lab exercise
- description: Brief description of the lab exercise
- objectives: Array of learning objectives for the lab
- requiredEquipment: Array of required equipment or software
- estimatedDuration: Estimated time to complete the lab
- instructions: Detailed step-by-step instructions
- expectedResults: Description of expected results
- assessmentCriteria: Array of assessment criteria
- difficultyLevel: Difficulty level (beginner, intermediate, advanced)
- createdDate: Date of creation
- lastUpdated: Date of last update
- status: Current status (draft, in review, published)

**Relationships**:
- One chapter to many lab exercises (1:M)

### 6. AssessmentQuestion

**Description**: Evaluations to test student comprehension of material.

**Attributes**:
- id: Unique identifier for the assessment question
- chapterId: Reference to the parent chapter
- type: Type of question (multiple choice, short answer, practical, etc.)
- questionText: The actual question text
- answerOptions: Array of possible answer options (for multiple choice)
- correctAnswer: The correct answer
- explanation: Explanation of the correct answer
- difficultyLevel: Difficulty level (beginner, intermediate, advanced)
- tags: Array of tags for categorization (ROS, Python, Navigation, etc.)
- createdDate: Date of creation
- lastUpdated: Date of last update
- usedIn: Array of assessments where this question appears

**Relationships**:
- One chapter to many assessment questions (1:M)

### 7. Citation

**Description**: References to academic or authoritative sources.

**Attributes**:
- id: Unique identifier for the citation
- chapterId: Reference to the parent chapter (optional, can be for entire textbook)
- citationType: Type of citation (IEEE, APA)
- sourceType: Type of source (book, journal, conference, website, etc.)
- title: Title of the source
- authors: Array of author names
- publicationDate: Date of publication
- publisher: Publisher information
- url: URL to the source (if applicable)
- doi: DOI of the source (if applicable)
- accessedDate: Date the source was accessed
- citationText: The formatted citation text
- referenceNumber: Sequential number for in-text citation
- createdDate: Date of creation
- lastUpdated: Date of last update

**Relationships**:
- One chapter to many citations (1:M)

### 8. NavigationComponent

**Description**: Table of contents, sidebar, search functionality for content access.

**Attributes**:
- id: Unique identifier for the navigation component
- type: Type of navigation component (tableOfContents, sidebar, search, etc.)
- configuration: Configuration settings for the component
- items: Array of navigation items (for table of contents or sidebar)
- createdDate: Date of creation
- lastUpdated: Date of last update

## State Transitions

### Chapter States
- DRAFT → IN_REVIEW → PUBLISHED
- PUBLISHED → IN_REVIEW → PUBLISHED (for updates)

### Textbook States
- DRAFT → IN_REVIEW → PUBLISHED
- PUBLISHED → IN_REVIEW → PUBLISHED (for updates)

## Validation Rules

1. **Textbook Validation**:
   - Word count must be between 15,000 and 25,000
   - Chapter count must be at least 10
   - Citation count must be at least 20

2. **Chapter Validation**:
   - Must include at least 1 diagram
   - Must include at least 1 code example
   - Must include at least 1 lab exercise
   - Must include at least 1 assessment question

3. **CodeExample Validation**:
   - Must be in Python or ROS 2 specific syntax
   - Must have been verified to work

4. **AssessmentQuestion Validation**:
   - Must have a correct answer defined
   - Must have an explanation for the answer