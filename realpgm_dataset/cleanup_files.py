import glob

'''
Trim first 2 lines of every file
This is because they all start with :

/home/michael/Desktop/MALWARE/zippedMalware/e9d7b791-b42e-11e7-b29d-80e65024849a.file:     file format pei-i386
D:\DAAACUMENTS\Research\SNN\REAL_PROGRAMS\REGWARE\fgrep.exe:     file format pei-i386

kinda gives it away, doesn't it

'''

files = []
files.extend(glob.glob(r"D:\DAAACUMENTS\Research\SNN\REAL_PROGRAMS\ASM_MALWARE\*.asm"))
files.extend(glob.glob(r"D:\DAAACUMENTS\Research\SNN\REAL_PROGRAMS\ASM_MALWARE_REST\*.asm"))
files.extend(glob.glob(r"D:\DAAACUMENTS\Research\SNN\REAL_PROGRAMS\ASM_REGWARE\*.asm"))
files.extend(glob.glob(r"D:\DAAACUMENTS\Research\SNN\REAL_PROGRAMS\ASM_REGWARE_REST\*.asm"))

for file in files:
    with open(file, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(file, 'w') as fout:
        fout.writelines(data[2:])
    
