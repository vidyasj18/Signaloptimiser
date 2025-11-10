# Traffic Signal Timing Optimization Report

## LaTeX Report Structure

This directory contains the LaTeX source files for the project report.

### Directory Structure

```
report/
├── main.tex                    # Main LaTeX file
├── sections/                   # Individual chapter files
│   ├── titlepage.tex          # Title page
│   ├── declaration.tex        # Declaration page
│   ├── certificate.tex        # Certificate page
│   ├── acknowledgement.tex    # Acknowledgement
│   ├── abstract.tex           # Chapter 1: Abstract
│   ├── introduction.tex       # Chapter 2: Introduction
│   ├── methodology.tex        # Chapter 3: Methodology
│   ├── irc_standards.tex      # Chapter 4: IRC Standards & Assumptions
│   ├── ml_optimization.tex    # Chapter 5: ML Optimization
│   ├── yolo_optimization.tex  # Chapter 6: YOLO Optimization
│   ├── sumo_validation.tex    # Chapter 7: SUMO Validation
│   ├── conclusion.tex         # Chapter 8: Conclusion
│   └── references.tex         # Chapter 9: References & Bibliography
├── images/                     # Directory for figures (add your images here)
└── README.md                   # This file
```

### Compilation Instructions

#### Using pdflatex (Recommended)

```bash
cd report
pdflatex main.tex
pdflatex main.tex  # Run twice for correct references and TOC
```

#### Using latexmk (Automatic)

```bash
cd report
latexmk -pdf main.tex
```

#### For bibliography support (if added later)

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Adding Images

1. Place your image files in the `images/` directory
2. In your LaTeX file, reference them using:

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{your_image_name}
    \caption{Your caption here}
    \label{fig:your_label}
\end{figure}
```

### Features

- **Hyperlinked Table of Contents**: Click on any entry to jump to that section
- **List of Figures**: Automatically generated
- **List of Tables**: Automatically generated
- **Consistent Formatting**: Professional report layout
- **Easy Navigation**: Cross-references and hyperlinks throughout

### Required LaTeX Packages

The following packages are used (all standard in most LaTeX distributions):
- hyperref (for hyperlinks)
- graphicx (for images)
- amsmath, amssymb (for mathematical notation)
- geometry (for page layout)
- fancyhdr (for headers/footers)
- listings (for code blocks)
- booktabs (for professional tables)
- caption, subcaption (for figure captions)
- longtable (for long tables)

### Output

The compilation will generate:
- `main.pdf` - The final report document
- Auxiliary files (.aux, .log, .toc, etc.) - Can be safely deleted

### Cleaning Up

To remove auxiliary files:

```bash
# Linux/Mac
rm *.aux *.log *.toc *.lof *.lot *.out

# Windows
del *.aux *.log *.toc *.lof *.lot *.out
```

### Notes

- The report is based on `report.txt` content
- All sections are modular and can be edited independently
- Images should be added to the `images/` folder as needed
- Bibliography support can be added by uncommenting the bibliography lines in `main.tex`

### Authors

- Bhuvan Dharwad (231CV214)
- Bidisha Koley (231CV215)
- Abhijith Sogal (231CV203)
- Vidya S. J (231CV155)

### Supervisor

Dr. Suresha S. N.  
Department of Civil Engineering  
NITK Surathkal

