# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Overview

This quickstart guide will help you get up and running with the Physical AI & Humanoid Robotics textbook project. This project is designed to teach students how to build AI-driven humanoid robots using ROS 2, Gazebo, Unity, and NVIDIA Isaac, culminating in a full Vision-Language-Action humanoid capstone project.

## Prerequisites

Before starting with the textbook content, ensure you have the following prerequisites:

1. **Software Requirements**:
   - Ubuntu 22.04 LTS (for simulation environments)
   - ROS 2 Humble Hawksbill installed
   - Node.js LTS (for Docusaurus development)
   - Git for version control

2. **Knowledge Requirements**:
   - Python programming experience
   - Basic linear algebra and calculus
   - Introductory machine learning experience

## Getting Started

### 1. Setting up the Development Environment

First, clone the textbook repository:

```bash
git clone [repository-url]
cd [repository-directory]
```

### 2. Installing Dependencies

Install Docusaurus dependencies:

```bash
npm install
```

### 3. Running the Textbook Locally

To run the textbook site locally for development or review:

```bash
npm start
```

This will start a local development server with hot reloading. The textbook will be accessible at `http://localhost:3000`.

### 4. Building for Production

To build the static files for deployment:

```bash
npm run build
```

The built files will be available in the `build/` directory.

## Textbook Structure

The textbook is organized into 7 layers, each building upon the previous:

1. **Foundations of Physical AI** - Embodied intelligence concepts
2. **ROS 2 Basics** - Robotic Nervous System fundamentals
3. **Simulation Environments** - Gazebo and Unity simulation
4. **NVIDIA Isaac Platform** - Isaac Sim and Isaac ROS
5. **Humanoid Robotics** - Kinematics, locomotion, HRI
6. **Vision-Language-Action Robotics** - Whisper, LLMs, and action execution
7. **Capstone Project** - End-to-end autonomous humanoid implementation

Each chapter follows a standardized structure:
- Overview / Learning Outcomes
- Core Concepts
- Equations, Diagrams, and Models
- Code Examples (Python/ROS 2)
- Simulation Demonstration (Gazebo/Isaac)
- Hands-on Lab / Mini Project
- Common Pitfalls & Debugging Notes
- Summary & Key Terms
- Further Reading + Citations
- Assessment Questions

## Working with Code Examples

All code examples are provided in Python/ROS 2 format and can be found alongside the relevant text. To run an example:

1. Navigate to the example directory (typically `static/code/` in the repository)
2. Ensure your ROS 2 environment is sourced:
   ```bash
   source /opt/ros/humble/setup.bash
   ```
3. Run the example according to its specific instructions

## Working with Simulations

Simulation examples require specific ROS 2 packages to be installed. For Gazebo simulations:

1. Install the Gazebo Harmonic packages:
   ```bash
   sudo apt install ros-humble-gazebo-*
   ```

For NVIDIA Isaac Sim:
1. Ensure you have an RTX GPU with the latest drivers
2. Install Isaac Sim following NVIDIA's documentation
3. Verify that CUDA and cuDNN are properly configured

## Contributing to the Textbook

If you're contributing to the textbook content:

1. Create a new branch from the main branch
2. Add or modify Markdown files in the `docs/` directory
3. Update diagrams in the `static/img/` directory if needed
4. Add or modify code examples in the `static/code/` directory
5. Run local build to verify formatting and links:
   ```bash
   npm run build
   ```
6. Submit a pull request with your changes

## Running Tests and Validation

To validate that your changes meet the textbook requirements:

1. **Content validation**: Verify that each chapter includes diagrams, code examples, labs, and assessments
2. **Build validation**: Run `npm run build` to ensure the site builds without errors
3. **Link validation**: Check that all internal links work correctly
4. **Code example validation**: Test that code examples execute correctly in your environment

## Deployment

The textbook is automatically deployed to GitHub Pages on each merge to the main branch. To manually deploy:

```bash
GIT_USER=<Your GitHub username> npm run deploy
```

## Support and Issues

If you encounter issues with the textbook content or setup:

1. Check the troubleshooting section for common solutions
2. Submit an issue in the GitHub repository with detailed information about the problem
3. For content-related queries, refer to the appropriate chapter or contact the maintainers