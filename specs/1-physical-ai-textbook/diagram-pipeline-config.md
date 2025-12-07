# Diagram Generation Configuration

This configuration file sets up the diagram generation pipeline for the textbook.

## Mermaid Configuration

```js
{
  "startOnLoad": true,
  "theme": "default",
  "flowchart": {
    "useMaxWidth": true,
    "htmlLabels": true,
    "curve": "basis"
  },
  "sequence": {
    "diagramMarginX": 50,
    "diagramMarginY": 10,
    "boxTextMargin": 5,
    "noteMargin": 10,
    "messageMargin": 35,
    "mirrorActors": true
  },
  "themeVariables": {
    "primaryColor": "#cde4ff",
    "primaryBorderColor": "#8cb3ff",
    "primaryTextColor": "#000000",
    "lineColor": "#333333"
  }
}
```

## PlantUML Configuration

Default settings for PlantUML diagram generation:
- Format: SVG
- Skinparam: Robotic theme with consistent colors
- Scale: 1.0 (for consistency across diagrams)

## Directory Structure

The diagram generation pipeline will look for source files in the following locations:
- Mermaid: Any .md file containing mermaid code blocks
- PlantUML: Files with .puml extension in static/diagrams/
- SVG: Files with .svg extension in static/diagrams/
- PNG: Files with .png extension in static/diagrams/

## Build Process

During the Docusaurus build process:
1. Mermaid diagrams are rendered in the browser using the Mermaid JS library
2. Static diagram files (SVG, PNG) are copied from static/diagrams/ to the build output
3. All diagrams are optimized for web performance

## Diagram Standards

All diagrams in the textbook must adhere to the following standards:
1. Consistent color palette across all diagrams in each chapter
2. Clear, readable fonts (minimum 14pt for important text)
3. Proper accessibility with alt text and descriptions
4. Appropriate file size (under 100KB for raster images)
5. Meaningful file names that relate to their content