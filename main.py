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
            I = np.sin(2*np.pi*1.1*t)
            I = 1 if I > 0 else -1
            Q = np.sin(2*np.pi*1.5*t)
            Q = 1 if Q > 0 else -1
            samp = I*np.cos(2*np.pi*10*t) + Q*np.sin(2*np.pi*10*t)
            
            self._txQ.put(samp)

class Ch:
    def __init__(self):
        global chQ
        self._rxQ = chQ
        global rxQ
        self._txQ = rxQ

        self._rxs = 0

        self._P_n = 0.01

    def Tick(self, t):
        if not self._rxQ.empty():
            rx = self._rxQ.get()
            self._txQ.put(rx + AWGN(self._P_n))

            self._rxs += 1

            if self._rxs > 100000:
                self._rxs = 0
                # DEBUG: log here?
        else:
            self._txQ.put(AWGN(self._P_n))

class Rx:
    def __init__(self):
        global rxQ
        self._rxQ = rxQ

        self._nvals = 16384
        self._vals = np.zeros(self._nvals)
        self._ivals = 0

        plt.ion()
        plt.plot(self._vals)
        plt.pause(0.0001)

        self._refresh_len = 100000
        self._refresh_timer = self._refresh_len

    def Tick(self, t):
        if not self._rxQ.empty():
            rx = self._rxQ.get()
            self._vals[self._ivals] = rx

            self._ivals += 1
            self._refresh_timer -= 1

            if self._ivals >= self._nvals:
                self._ivals = 0
                plt.clf()
                plt.plot(self._vals)
                plt.pause(0.0001)

                if self._refresh_timer <= 0:
                    self._refresh_timer = self._refresh_len

if __name__ == "__main__":
    main()
