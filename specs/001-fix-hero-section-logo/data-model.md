# Data Model: Fix Missing Hero Section and Book Logo

## Overview
This feature is primarily a UI/UX enhancement with minimal data requirements. The hero section will not store persistent data but will include configuration elements for display purposes.

## Components & Properties

### HeroSection Component
- `title`: string - Main heading text for the hero section
- `subtitle`: string - Subheading text that describes the purpose
- `logoUrl`: string - Path to the book logo image
- `ctaText`: string - Call-to-action button text (e.g., "Get Started", "Learn More")
- `ctaLink`: string - URL the call-to-action button points to
- `designTheme`: string - CSS class or theme identifier for styling (e.g., "futuristic-glassmorphism")
- `animationEnabled`: boolean - Whether to enable subtle animations

### Accessibility Properties
- `ariaLabel`: string - Accessibility label for screen readers
- `skipToContentLink`: string - Anchor link to main content for accessibility

## Configuration Schema
```javascript
{
  "heroSection": {
    "title": "Physical AI & Humanoid Robotics",
    "subtitle": "The textbook for next-generation robotics and AI",
    "logoUrl": "/img/book-logo.svg",
    "ctaText": "Start Reading",
    "ctaLink": "/docs/intro",
    "designTheme": "futuristic-glassmorphism",
    "animationEnabled": true,
    "ariaLabel": "Main introduction section"
  }
}
```