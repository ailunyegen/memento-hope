"""Convert remaining inline metric numbers in results.tex text only (not inside tabular)."""
import re

with open('manuscript/sections/results.tex', encoding='utf-8') as f:
    content = f.read()

# Split into tabular and non-tabular sections
# Only convert non-tabular text
parts = re.split(r'(\\begin\{tabular\}.*?\\end\{tabular\})', content, flags=re.DOTALL)

def convert_text(text):
    """Convert MS/OE/stage values in prose text to percentage scale."""
    # Match patterns like "mission success of 0.6036" or "MS = 0.6260" or "OE = 0.7009"
    # Also match ranges like "0.60--0.63"
    
    def replacer(m):
        full = m.group(0)
        val = float(full)
        if val < 0.01:  # skip tiny values (params)
            return full
        # Check context 
        ctx_start = max(0, m.start() - 40)
        ctx = text[ctx_start:m.end() + 20]
        # Skip parameter contexts
        if re.search(r'\\theta|\\epsilon|temperature|threat|pressure|scenario|coefficient|weight|penalty|severity|variance|boost|\\pm|SD|deficit', ctx, re.IGNORECASE):
            return full
        new_val = val * 100
        decimals = len(full.split('.')[1]) if '.' in full else 0
        if decimals <= 2:
            return f'{new_val:.1f}'
        else:
            return f'{new_val:.2f}'
    
    return re.sub(r'(?<![0-9\.\-])(0\.[0-9]{2,4})(?![0-9])', replacer, text)

new_parts = []
for i, part in enumerate(parts):
    if i % 2 == 0:  # outside tabular = prose text
        new_parts.append(convert_text(part))
    else:  # inside tabular = leave as-is (already manually converted)
        new_parts.append(part)

new_content = ''.join(new_parts)
with open('manuscript/sections/results.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Done converting prose text in results.tex')
