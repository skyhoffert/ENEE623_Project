# Sky Hoffert
# ENEE623 Project main class.

import matplotlib.pyplot as plt
import msvcrt
import numpy as np
import threading
from queue import Queue

from util import Log

chQ = Queue()
rxQ = Queue()
kbQ = Queue()
KILL = False

def main():
    Log("Main program started.")

    Fs = 44100
    tx = Tx()
    ch = Ch()
    rx = Rx()
    kb = threading.Thread(target=ThreadKB)
    kb.start()

    global KILL
    global kbQ

    t = 0
    while not KILL:
        if not kbQ.empty():
            act = kbQ.get()

            if act == "t":
                tx.Toggle()

        tx.Tick(t)
        ch.Tick(t)
        rx.Tick(t)
        t += 1/Fs

    kb.join()

    Log("Main program ended.")

def ThreadKB():
    global KILL
    global kbQ
    while not KILL:
        inp = input()
        if inp == "q" or inp == "quit":
            KILL = True
        else:
            kbQ.put(inp)

def AWGN(v):
    return np.random.normal(0,v,1)[0]

class Tx:
    def __init__(self):
        global chQ
        self._txQ = chQ
        self._active = False

    def Toggle(self):
        self._active = not self._active

    def Tick(self, t):
        if self._active:
            self._txQ.put(0)

class Ch:
    def __init__(self):
        global chQ
        self._rxQ = chQ
        global rxQ
        self._txQ = rxQ

        self._rxs = 0

    def Tick(self, t):
        if not self._rxQ.empty():
            rx = self._rxQ.get()

            self._rxs += 1

            if self._rxs > 100000:
                Log(f"ch got 100000 rxs")
                self._rxs = 0
        else:
            self._txQ.put(AWGN(1))

class Rx:
    def __init__(self):
        global rxQ
        self._rxQ = rxQ

    def Tick(self, t):
        if not self._rxQ.empty():
            rx = self._rxQ.get()

if __name__ == "__main__":
    main()
