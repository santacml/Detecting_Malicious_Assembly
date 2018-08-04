# To add or remove an app from the Startup tab, press the Windows Logo Key + R, type shell:startup, and then select OK.
# got exe's from robocopy "C:\Program Files (x86)" "D:\DAAACUMENTS\TEST" *.exe /S
# stopped it after 1k programs
import glob
import subprocess
import os
import sys

files = []
files.extend(glob.glob(r"D:\DAAACUMENTS\Research\SNN\REAL_PROGRAMS\REGWARE\*"))
#print(files)

options = "objdump -M intel -M --no-aliases -d --no-show-raw-insn "
ending = r" | perl -p -e 's/^\s+(\S+):\t//;'   > "

asmFolder = "D:\DAAACUMENTS\Research\SNN\REAL_PROGRAMS\ASM_REGWARE\\"

for file in files:
    print(file)
    head, tail = os.path.split(file)
    base = os.path.splitext(tail)[0]
    asmFile = os.path.join( asmFolder, base + ".asm")
    if os.path.exists(asmFile):
        continue # skip existing files... I guess.. because this takes forever
    print(options + file + ending + asmFile)
    # subprocess.call(options + file + ending + asmFile, shell=True)
    # must be run with powershell for piping
    process = subprocess.Popen(['powershell.exe', options + file + ending + asmFile], stdout=sys.stdout)
    process.wait()
    # break




# example command: 
# objdump -M intel -M --no-aliases -d --no-show-raw-insn D:\DAAACUMENTS\Research\SNN\REAL_PROGRAMS\REGWARE\LOOT.exe | perl -p -e 's/^\s+(\S+):\t//;'