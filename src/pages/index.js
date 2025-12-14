import React from 'react';
import Layout from '@theme/Layout';
import HeroSection from '@site/src/components/HeroSection';

export default function Home() {
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="The textbook for next-generation robotics and AI">
      <main>
        <HeroSection />
      </main>
    </Layout>
  );
}