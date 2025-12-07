# Prerequisites

Before diving into the Physical AI & Humanoid Robotics textbook, you need to set up your development environment with the required software and tools.

## Software Requirements

1. **Ubuntu 22.04 LTS** (for simulation environments)
   - Other Linux distributions may work but are not officially supported
   - Windows users: Consider using WSL2 or a virtual machine

2. **ROS 2 Humble Hawksbill** (Long Term Support version)
   - Installation guide: https://docs.ros.org/en/humble/Installation.html
   - Install Desktop version for full functionality

3. **Node.js LTS** (for Docusaurus development)
   - Download from: https://nodejs.org/
   - Version 18 or higher required

4. **Git** for version control
   - Usually pre-installed on Ubuntu
   - Install with: `sudo apt install git`

## Installation Steps

### ROS 2 Humble Installation

1. Set locale:
   ```bash
   locale  # check for UTF-8
   sudo apt update && sudo apt install locales
   sudo locale-gen en_US en_US.UTF-8
   sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
   export LANG=en_US.UTF-8
   ```

2. Add ROS 2 apt repository:
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros- humble.repos | sudo tee /etc/ros/humble.repos
   sudo apt install curl gnupg lsb-release
   curl -sSL https://packages.ros.org/ros.asc | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/humble/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. Install ROS 2 packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   sudo apt install ros-humble-cv-bridge ros-humble-tf2-tools ros-humble-tf2-geometry-msgs
   sudo apt install python3-colcon-common-extensions
   ```

4. Source the setup:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

### Node.js Installation

1. Download and install Node.js LTS from https://nodejs.org/
   - Or use a Node version manager like nvm:
   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   source ~/.bashrc
   nvm install --lts
   nvm use --lts
   ```

### Getting the Textbook Code

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd [repository-directory]
   ```

2. Install Docusaurus dependencies:
   ```bash
   npm install
   ```

## Knowledge Requirements

Ensure you have experience with:

- **Python Programming**: Functions, classes, modules, exception handling
- **Basic Linear Algebra**: Vectors, matrices, transformations
- **Basic Calculus**: Derivatives, integrals, optimization
- **Introductory Machine Learning**: Concepts of training, inference, neural networks

## Optional but Recommended

- NVIDIA RTX GPU for Isaac Sim (4090 recommended, but any RTX card will work)
- Experience with version control systems (Git)
- Basic understanding of robotics concepts (forward/inverse kinematics, control theory)