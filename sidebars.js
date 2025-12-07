// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['intro', 'prerequisites'],
    },
    {
      type: 'category',
      label: 'Physical AI Fundamentals',
      items: ['physical-ai/intro', 'physical-ai/sensors-perception'],
    },
    {
      type: 'category',
      label: 'ROS 2 Basics',
      items: ['ros2-basics/intro', 'ros2-basics/nodes-topics', 'ros2-basics/urdf', 'ros2-basics/rclpy'],
    },
    {
      type: 'category',
      label: 'Simulation',
      items: ['simulation/intro', 'simulation/gazebo', 'simulation/digital-twins'],
    },
    {
      type: 'category',
      label: 'NVIDIA Isaac Platform',
      items: ['isaac-platform/intro', 'isaac-platform/isaac-sim', 'isaac-platform/isaac-ros', 'isaac-platform/vslam-nav2'],
    },
    {
      type: 'category',
      label: 'Humanoid Robotics',
      items: ['humanoid-robotics/intro', 'humanoid-robotics/kinematics', 'humanoid-robotics/locomotion', 'humanoid-robotics/manipulation', 'humanoid-robotics/hri'],
    },
    {
      type: 'category',
      label: 'Vision-Language-Action Robotics',
      items: ['vla-robotics/intro', 'vla-robotics/whisper', 'vla-robotics/llm-planning', 'vla-robotics/action-execution'],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: ['capstone/intro', 'capstone/system-design'],
    },
    {
      type: 'category',
      label: 'Appendices',
      items: ['appendices/hardware', 'appendices/reinforcement-learning', 'appendices/sim-to-real'],
    }
  ],
};

export default sidebars;