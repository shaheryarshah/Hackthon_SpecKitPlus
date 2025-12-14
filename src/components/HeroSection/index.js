import React from 'react';
import styles from './styles.module.css';

/**
 * HeroSection Component
 * Displays a hero section with a book logo and customizable content
 * Implements futuristic design aesthetics as specified in the project
 *
 * @param {Object} props - Component properties
 * @param {string} props.title - Main heading text for the hero section
 * @param {string} [props.subtitle] - Subheading text that describes the purpose
 * @param {string} props.logoUrl - Path to the book logo image
 * @param {string} [props.ctaText] - Call-to-action button text
 * @param {string} [props.ctaLink] - URL the call-to-action button points to
 * @param {Array<{text: string, link: string, external?: boolean}>} [props.navLinks] - Additional navigation links
 * @param {string} [props.designTheme='default'] - CSS class for styling theme
 * @param {boolean} [props.animationEnabled=true] - Whether to enable subtle animations
 * @param {string} [props.ariaLabel='Main introduction section'] - Accessibility label for screen readers
 * @param {string} [props.skipToContentLink='#main'] - Anchor link to main content for accessibility
 * @param {Function} [props.onCtaClick] - Called when the call-to-action button is clicked
 */
const HeroSection = ({
  title = "Physical AI & Humanoid Robotics",
  subtitle = "The textbook for next-generation robotics and AI",
  logoUrl = "/img/book-logo.svg",
  ctaText = "Start Reading",
  ctaLink = "/docs/intro",
  navLinks = [],
  designTheme = "default",
  animationEnabled = true,
  ariaLabel = "Main introduction section",
  skipToContentLink = "#main",
  onCtaClick
}) => {
  // Accessibility enhancement: skip to content link
  const skipLink = (
    <a href={skipToContentLink} className={styles.skipLink}>
      Skip to main content
    </a>
  );

  return (
    <section
      className={`${styles.heroSection} ${styles[designTheme]} ${animationEnabled ? styles.animated : ''}`}
      aria-label={ariaLabel}
    >
      {skipLink}
      <div className={styles.heroContainer}>
        <div className={styles.heroContent}>
          <div className={styles.logoContainer}>
            <img
              src={logoUrl}
              alt="Book Logo"
              className={`${styles.heroLogo} ${animationEnabled ? styles.pulseEffect : ''}`}
              loading="eager"  /* Hero section is above the fold, so eager loading is appropriate */
              fetchPriority="high"  /* Prioritize loading of the hero image */
            />
          </div>
          <h1 className={styles.heroTitle}>{title}</h1>
          {subtitle && <p className={styles.heroSubtitle}>{subtitle}</p>}
          {ctaText && (
            <a
              href={ctaLink}
              className={styles.heroCtaButton}
              aria-label={ctaText}
              onClick={(e) => {
                // Optional: Add analytics or other tracking here
                if (typeof onCtaClick === 'function') {
                  onCtaClick(e);
                }
              }}
            >
              {ctaText}
            </a>
          )}
        </div>

        {/* Additional navigation links */}
        {navLinks && navLinks.length > 0 && (
          <nav className={styles.navLinksContainer} aria-label="Additional navigation">
            <ul className={styles.navLinksList}>
              {navLinks.map((link, index) => (
                <li key={index} className={styles.navLinkItem}>
                  <a
                    href={link.link}
                    className={styles.navLink}
                    aria-label={link.text}
                    onClick={(e) => {
                      if (link.external) {
                        e.preventDefault();
                        window.open(link.link, '_blank', 'noopener,noreferrer');
                      }
                    }}
                  >
                    {link.text}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        )}
      </div>
    </section>
  );
};

export default HeroSection;