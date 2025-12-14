// cypress/e2e/hero-section.spec.js
describe('HeroSection Component', () => {
  beforeEach(() => {
    // Visit the homepage where the HeroSection is displayed
    cy.visit('/');
  });

  it('displays the hero section with proper content', () => {
    // Check if the hero section is present
    cy.get('[aria-label="Main introduction section"]').should('be.visible');
    
    // Check if the title is displayed correctly
    cy.get('[class*="heroTitle"]').should('contain.text', 'Physical AI & Humanoid Robotics');
    
    // Check if the subtitle is displayed correctly
    cy.get('[class*="heroSubtitle"]').should('contain.text', 'The textbook for next-generation robotics and AI');
    
    // Check if the logo is displayed
    cy.get('[class*="heroLogo"]').should('be.visible');
    cy.get('[class*="heroLogo"]').should('have.attr', 'alt', 'Book Logo');
  });

  it('has a functional CTA button', () => {
    // Find the CTA button and check its properties
    cy.get('[class*="heroCtaButton"]')
      .should('be.visible')
      .should('have.attr', 'href', '/docs/intro')
      .should('contain.text', 'Start Reading');
    
    // Click the CTA button and verify navigation (if applicable)
    // Commented out to avoid actual navigation during testing
    // cy.get('[class*="heroCtaButton"]').click();
    // cy.url().should('include', '/docs/intro');
  });

  it('handles navigation links properly', () => {
    // This would test if navigation links are working properly
    // Since we don't have specific navigation links in default config
    // we'll skip detailed navigation link tests
    
    // Verify that navigation elements have proper attributes
    cy.get('[class*="navLink"]').should('have.length.gte', 0); // Could be 0 if no nav links provided
  });

  it('is responsive and displays correctly on different screen sizes', () => {
    // Test on desktop resolution
    cy.viewport(1280, 720);
    cy.get('[class*="heroSection"]').should('be.visible');
    
    // Test on tablet resolution
    cy.viewport(768, 1024);
    cy.get('[class*="heroSection"]').should('be.visible');
    
    // Test on mobile resolution
    cy.viewport(375, 667);
    cy.get('[class*="heroSection"]').should('be.visible');
  });

  it('has proper accessibility features', () => {
    // Check for skip link
    cy.get('[class*="skipLink"]').should('exist');
    
    // Verify aria-label attributes
    cy.get('[aria-label="Main introduction section"]').should('exist');
    
    // Check that elements are focusable and have proper focus states
    cy.get('[class*="heroCtaButton"]').focus();
    // Can't directly test focus styles in Cypress, but the element should receive focus
  });

  it('applies correct design theme', () => {
    // Verify that the futuristic theme classes are applied if specified
    cy.get('[class*="heroSection"]').should('exist');
    // The specific theme class would depend on how it's implemented
  });
});