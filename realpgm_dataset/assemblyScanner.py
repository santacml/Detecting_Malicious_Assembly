#!/usr/bin python3
import os
import random
import pickle

class ScanError(Exception): pass

stuff = "abcdefghijklmnopqrstuvwxyz0123456789.\\_$?@'"
breakers = "+-*/[]{}():,"
alphabet = "abcdefghijklmnopqrstuvwxyz"
nums = "0123456789"


class StateMachine(object):
    def __init__(self, name):
        self.states = {}
        self.name = name
        self.currStr = ""
        self.ended = False
        
    def accept(self, newChar):
        # self.currChar = char
        nextState = None
        
        for key, state in self.states.items():
            # print("checking key", key)
            # print(newChar)
            if newChar in key:
                # print("found " + repr(newChar) + " in " + key)
                state.currStr = self.currStr + newChar
                nextState = state
                # self.currStr = ""  #doesn't matter->gets overwritten 2 lines above when it matters
        
        if not nextState:
            nextState = self
            self.ended = True
        
        return nextState
        
    def clear(self):
        self.currStr = ""
        self.ended = False
        
        
    def terminate(self, retVal=None):
        if not retVal: 
            # retVal = (self.name, self.currStr) 
            retVal = self.currStr
        self.clear()
        return retVal
        
class Word(StateMachine):
    def __init__(self):
        StateMachine.__init__(self, "word")
        self.states = {
        }
        for thing in stuff:
            self.states[thing] = self
        
class End(StateMachine):
    def __init__(self, name):
        StateMachine.__init__(self, name)
        
    def terminate(self): # this is dangerous. All ends just use their name as the currstr
        self.currStr = self.name
        return super().terminate()
        
class String(StateMachine):
    def __init__(self):
        # DOES NOT ALLOW UNDERSCORE... ???
        StateMachine.__init__(self, "string")
        self.states = {
            ":": self,
            " ": self,
            "'": self,
            "\"": End(self.name)
        }
        for thing in stuff:
            self.states[thing] = self
        
    def terminate(self):
        raise ScanError("String must end with \"")
        
class FuncCall(StateMachine):
    def __init__(self):
        StateMachine.__init__(self, "funccall")
        self.states = {
            ":": self,
            " ": self,
            "'": self,
            "\"": self,
            "_": self,
            ">": End(self.name)
        }
        for thing in stuff+breakers+nums:  # really collect everything until end of func call
            self.states[thing] = self
        
    def terminate(self):
        raise ScanError("FuncCall must end with >")
        
class Comment(StateMachine):
    def __init__(self):
        StateMachine.__init__(self, "comment")
        self.states = {}
        
    def accept(self, newChar):
        #this will only happen if we are in a single line comment
        #accept everything, forever
        self.currStr = self.currStr + newChar
        nextState = self
        
        return nextState
        
    def terminate(self):
        # return super().terminate()
        return super().terminate()
        
        
        
class MasterMachine(StateMachine):
    def __init__(self):
        StateMachine.__init__(self, "do not see")
        self.states = {
            "\"": String(),
            # ",": End("comma"),
            # ":": End("colon"),
            "#": Comment(),
            ";": Comment(),
            "<": FuncCall()
        }
        for thing in stuff:
            self.states[thing] = Word()
        
        ''' theory is
        going along, see like [eax+0x90823]
        we want this to be [ eax + 0x90823 ]
        so word doesn't accept any of the breakers
        and then all the breakers go right to End
        also reduce redundancy with : and ,
        '''
        for thing in breakers:
            self.states[thing] = End(thing)
        
    def clear(self):
        self.currStr = ""
        self.ended = False
        return self
        
        
class Scanner(object):
    def __init__(self, file):
        self.file = file
        
    def scan(self):
        lines = []
        
        # with open(self.file, 'r' ,encoding='utf16') as f:
        with open(self.file, 'rb') as f:
            currLine = 0
            masterMachine = MasterMachine()
            machine = masterMachine.clear()
            for line in f:
                currCol = 0
                tokens = []
                try:
                    machine = masterMachine.clear()
                    
                    currLine += 1
                    line = line.lower()
                    # print(repr(line))
                    # print(line)
                    for char in line.decode('ascii'):
                        currCol += 1
                        if char in ("\r\n", "\r", "\n"): continue #gets rid of newlines!
                        
                        
                        
                        # print("now accepting", char)
                        machine = machine.accept(char)
                        
                        if machine.ended: 
                            # print(machine.ended, machine.currStr, char)
                            # comments will only ever be at the end of a line... redundant
                            #idk, whatever
                            if machine.currStr and machine.name is not "comment" and machine.name is not "string":
                                tokens.append(machine.terminate())
                            
                            
                            machine = masterMachine.clear().accept(char)
                            if not machine.currStr and char  not in [" ","\t"] :  
                                # token not found, also discard whitespace
                                raise ScanError("Unexpected Token", char)
                                
                            # print("made new machine with: " + char)
                    
                    
                    # at end of line, terminate machine
                    if  machine.currStr and machine.name is not "comment" and machine.name is not "string":
                        tokens.append(machine.terminate())
                    
                    lines.append(tokens)
                except ScanError as e:
                    print("Encountered error while scanning line: " + str(currLine )+ ".")
                    print(e)
                    print(line.decode('ascii'), " " * (currCol-2) + "^")
                    print()
                    # return
                    
                    # continue onwards, can't account for everything, tired of dealing with this
                    
            # this is specifically for files that end in a block comment... 
            # I guess I don't really need this...
            if (machine.name == "block_comment") and machine.currStr:
                tokens.append(machine.terminate())
            
        return lines

# lines = Scanner(r"D:\DAAACUMENTS\Research\SNN\complexPrograms\complexRegular\assembly\pgm1.asm").scan()
# for line in lines[0:10]: print(line)
# for line in lines: print(line)
