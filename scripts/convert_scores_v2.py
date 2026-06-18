"""v2: Only convert MS, OE, stage scores, command resilience in results/discussion/abstract/conclusion.
Skip method.tex (math formulas) and experimental_setup.tex (scenario params/θ/ε)."""
import re, os

SECTIONS_DIR = 'manuscript/sections'
PROCESS_FILES = {'results.tex', 'discussion.tex', 'abstract.tex', 'conclusion.tex'}

def convert_line(line):
    """Convert metric scores. Uses context: only converts numbers that are
    clearly MS/OE/stage/command_resilience scores. Skips LER, time, params."""
    
    def is_metric(m, val):
        """Check context before and after the match."""
        if val < 0.01:
            return False
        # Check 30 chars before
        before = line[max(0,m.start()-60):m.start()]
        after = line[m.end():m.end()+40]
        around = before + ' ' + after
        
        # Skip if in math mode (formulas)
        if before.count('$') % 2 == 1:
            return False
        
        # Skip if this is a LER row context  
        if re.search(r'LER|ler|loss.exchange|Loss.Exchange', around):
            if val > 0.9:  # MS/OE in LER context? unlikely if >0.9 and near LER
                # Check if we're in the MS column of a multi-column table
                if re.search(r'(MS|mission.success)', before[-50:]):
                    return True
                return False
        
        # Skip θ/ε parameter values
        if re.search(r'\\theta|\\epsilon|theta|epsilon', before[-30:]):
            return False
        
        # Skip SD values (they appear after ±)
        if re.search(r'\\pm\s*$', before.strip()):
            return False
        
        # Skip if this is a time value (hours)
        if re.search(r'\\,h|hours|Time|time', after[:30]):
            return False
        
        # Skip scenario threat/EW/time coefficients
        if re.search(r'Thr\.|threat|EW|Time\.pressure', before[-30:]):
            return False
        
        # Skip table column width specs
        if re.search(r'\{0\.\d+cm\}', line):
            return False
        
        # Skip severity scores in BottleneckDetector table
        if re.search(r'Severity|severity', before[-30:]):
            return False
        
        # Skip deficit values in BottleneckDetector
        if re.search(r'deficit', before[-30:]):
            return False
        
        # OK - this is likely a metric score
        return True
    
    def replacer(m):
        full = m.group(0)
        val = float(full)
        if not is_metric(m, val):
            return full
        
        new_val = val * 100
        decimals = len(full.split('.')[1])
        if decimals <= 2:
            return f'{new_val:.1f}'
        else:
            return f'{new_val:.2f}'
    
    return re.sub(r'(?<![0-9\.\-])(0\.[0-9]{2,4})(?![0-9])', replacer, line)


for fname in sorted(os.listdir(SECTIONS_DIR)):
    if fname not in PROCESS_FILES:
        continue
    
    path = os.path.join(SECTIONS_DIR, fname)
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    changed = 0
    for line in lines:
        nl = convert_line(line)
        if nl != line:
            changed += 1
            # Print the change for verification
            old_nums = re.findall(r'(?<![0-9\.\-])(0\.[0-9]{2,4})(?![0-9])', line)
            new_nums = re.findall(r'(?<![0-9\.\-])([1-9][0-9]*\.[0-9]+)(?![0-9])', nl)
            if old_nums and new_nums:
                print(f'  {fname}: {old_nums[0]} → {new_nums[0]}')
        new_lines.append(nl)
    
    if changed > 0:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f'{fname}: {changed} lines converted')
    else:
        print(f'{fname}: no changes needed')
