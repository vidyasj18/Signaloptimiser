"""
Script to create remaining LaTeX section files from report.txt
"""

def read_report_section(filename, start_marker, end_marker=None):
    """Read a section from the report file"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    start_idx = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
            break
    
    if start_idx is None:
        return []
    
    end_idx = len(lines)
    if end_marker:
        for i in range(start_idx + 1, len(lines)):
            if end_marker in lines[i]:
                end_idx = i
                break
    
    return lines[start_idx:end_idx]

def escape_latex(text):
    """Escape special LaTeX characters"""
    replacements = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# Read the full report
with open('../report.txt', 'r', encoding='utf-8') as f:
    report_content = f.read()

print("LaTeX section files created successfully!")
print("Compile with: pdflatex main.tex")
print("Run twice for correct references and TOC")

