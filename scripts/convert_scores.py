"""Precisely convert metric scores to percentage scale (×100) across all manuscript sections.
Rules:
- Convert: MS, OE, stage scores (detect/disrupt/breach/control/sustain), command_resilience
- Convert: ΔMS, ΔOE (become percentage points)
- Convert: CI bounds like [0.6840, 0.6855]
- Do NOT convert: LER (>1 ratio), time (hours), θ/ε params, threat/EW/time coeffs,
  SD values, scenario weights, severity/boost values in BottleneckDetector table
"""
import re, os

SECTIONS_DIR = 'manuscript/sections'

# ── helper: is this number in a "metric" context? ──
def is_metric_context(line, match_start, match_end, val):
    """Heuristic: check surrounding text to decide if this is a metric score."""
    prefix = line[max(0,match_start-50):match_start]
    suffix = line[match_end:match_end+50]
    full_context = (prefix + suffix).lower()
    
    # Definitely NOT metrics
    if re.search(r'\\theta|\\epsilon|temperature|threat|pressure|time\.pressure|ew\.threat', full_context):
        return False
    # LER column header check
    if re.search(r'LER|ler', line[:match_start]) and val > 0.99:
        return False
    # Parameter sweep tables (θ row, ε row)
    if re.match(r'^\s*0\.\d+\s*&', line.strip()):
        if val < 0.99:
            # Could be θ value row in hyperparam table - these have 0.1, 0.3, 0.5 etc
            if val in [0.1, 0.3, 0.5, 0.7, 0.9, 0.005, 0.01, 0.02, 0.05]:
                return False
    # Very small values used as parameters
    if val < 0.01:
        return False
    # Severity scores and boost values in BottleneckDetector table
    if 'severity' in full_context or 'boost' in full_context:
        return False
    # Deficit values in BottleneckDetector
    if 'deficit' in prefix.lower():
        return False
    
    return True


def should_convert_line_for_metrics(line):
    """Check if line contains metric scores that should be converted."""
    # Lines with LER values mixed in - only convert MS/OE columns
    # Lines with only LER, time, params - skip entirely
    
    # Check for table rows with MS/OE/LER pattern (5 or 6 columns)
    # These have: label & MS & OE & LER & time & cmd_resilience
    if re.search(r'\\midrule|\\toprule|\\bottomrule|Setting|Stage|Method|Seed|Scenario|Summary|Mean|\\shortstack', line):
        pass  # Process these
    
    return True


def convert_line(line):
    """Convert metric scores in a line from [0,1] to [0,100] percentage scale."""
    
    def replacer(m):
        full = m.group(0)
        val = float(full)
        
        # Context check
        if not is_metric_context(line, m.start(), m.end(), val):
            return full
        
        new_val = val * 100
        # Preserve decimal precision
        decimals = len(full.split('.')[1]) if '.' in full else 0
        if decimals <= 2:
            return f'{new_val:.1f}'
        else:
            return f'{new_val:.2f}'
    
    # Match 0.xxxx numbers (2-4 decimal places)
    return re.sub(r'(?<![0-9\.\-])(0\.[0-9]{2,4})(?![0-9])', replacer, line)


# ── Main ──
for fname in sorted(os.listdir(SECTIONS_DIR)):
    if not fname.endswith('.tex'):
        continue
    
    path = os.path.join(SECTIONS_DIR, fname)
    with open(path, encoding='utf-8') as f:
        original = f.read()
    
    lines = original.split('\n')
    new_lines = []
    changed = 0
    
    for line in lines:
        nl = convert_line(line)
        if nl != line:
            changed += 1
        new_lines.append(nl)
    
    if changed > 0:
        new_content = '\n'.join(new_lines)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'{fname}: {changed} lines converted')
    else:
        print(f'{fname}: no changes')
