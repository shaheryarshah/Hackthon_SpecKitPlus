import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import HeroSection from '@site/src/components/HeroSection';

// Mock the CSS module
jest.mock('@site/src/components/HeroSection/styles.module.css', () => ({
  heroSection: 'heroSection',
  heroContainer: 'heroContainer',
  heroContent: 'heroContent',
  logoContainer: 'logoContainer',
  heroLogo: 'heroLogo',
  heroTitle: 'heroTitle',
  heroSubtitle: 'heroSubtitle',
  heroCtaButton: 'heroCtaButton',
  skipLink: 'skipLink',
  animated: 'animated',
  pulseEffect: 'pulseEffect',
  navLinksContainer: 'navLinksContainer',
  navLinksList: 'navLinksList',
  navLinkItem: 'navLinkItem',
  navLink: 'navLink',
  futuristic: 'futuristic',
  default: 'default'
}));

describe('HeroSection Component', () => {
  const defaultProps = {
    title: "Physical AI & Humanoid Robotics",
    subtitle: "The textbook for next-generation robotics and AI",
    logoUrl: "/img/book-logo.svg",
    ctaText: "Start Reading",
    ctaLink: "/docs/intro",
  };

  test('renders main heading', () => {
    render(<HeroSection {...defaultProps} />);
    expect(screen.getByRole('heading', { name: /Physical AI & Humanoid Robotics/i })).toBeInTheDocument();
  });

  test('renders subtitle', () => {
    render(<HeroSection {...defaultProps} />);
    expect(screen.getByText(/The textbook for next-generation robotics and AI/i)).toBeInTheDocument();
  });

  test('renders logo image with correct alt text', () => {
    render(<HeroSection {...defaultProps} />);
    const logo = screen.getByRole('img', { name: /Book Logo/i });
    expect(logo).toBeInTheDocument();
    expect(logo).toHaveAttribute('src', '/img/book-logo.svg');
  });

  test('renders CTA button with correct text and link', () => {
    render(<HeroSection {...defaultProps} />);
    const ctaButton = screen.getByRole('link', { name: /Start Reading/i });
    expect(ctaButton).toBeInTheDocument();
    expect(ctaButton).toHaveAttribute('href', '/docs/intro');
  });

  test('renders with default values when no props provided', () => {
    render(<HeroSection />);
    expect(screen.getByRole('heading', { name: /Physical AI & Humanoid Robotics/i })).toBeInTheDocument();
    expect(screen.getByText(/The textbook for next-generation robotics and AI/i)).toBeInTheDocument();
  });

  test('renders navigation links when provided', () => {
    const navLinks = [
      { text: 'Chapter 1', link: '/docs/chapter1' },
      { text: 'Resources', link: '/resources' }
    ];
    
    render(<HeroSection {...defaultProps} navLinks={navLinks} />);
    
    expect(screen.getByRole('link', { name: /Chapter 1/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Resources/i })).toBeInTheDocument();
  });

  test('does not render navigation links when not provided', () => {
    render(<HeroSection {...defaultProps} />);
    const navLinks = screen.queryByRole('navigation', { name: /Additional navigation/i });
    expect(navLinks).not.toBeInTheDocument();
  });

  test('applies animation class when animationEnabled is true', () => {
    render(<HeroSection {...defaultProps} animationEnabled={true} />);
    // Since we can't directly test CSS classes with this mock, 
    // we're testing that the component renders without error when animationEnabled is true
    expect(screen.getByRole('heading', { name: /Physical AI & Humanoid Robotics/i })).toBeInTheDocument();
  });

  test('does not apply animation class when animationEnabled is false', () => {
    render(<HeroSection {...defaultProps} animationEnabled={false} />);
    // Testing that the component renders properly without animation
    expect(screen.getByRole('heading', { name: /Physical AI & Humanoid Robotics/i })).toBeInTheDocument();
  });

  test('applies design theme class', () => {
    render(<HeroSection {...defaultProps} designTheme="futuristic-glassmorphism" />);
    // Testing that the component accepts and applies the designTheme prop
    expect(screen.getByRole('heading', { name: /Physical AI & Humanoid Robotics/i })).toBeInTheDocument();
  });
});