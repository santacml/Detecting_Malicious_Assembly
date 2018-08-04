import glob
import codecs
files = []

files.extend(glob.glob(r"D:\DAAACUMENTS\Research\SNN\REAL_PROGRAMS\ASM_REGWARE\*.asm"))

for file in files:
    print(files)
    lines = []
    with open(file, 'r' ,encoding='utf16') as f:
        lines = f.read()  
    
    #write output file
    with codecs.open(file, 'w', encoding='mbcs') as f:
        f.write(lines)