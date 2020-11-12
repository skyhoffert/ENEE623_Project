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

Fs = 44100.23
fc = 1000
b = np.genfromtxt("filter.csv", delimiter=",")

np.random.seed(123456)
SEQ = np.random.randint(2, size=16384)
Log(str(SEQ[0:10]))
bpSym = 2
spSym = 10000

def main():
    Log("Main program started.")

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
            elif act == "r":
                rx.Toggle()

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

        self._iseq = 0

        global bpSym
        global spSym
        global SEQ
        global Fs
        self._seq = SEQ
        self._nseq = len(self._seq)

        self._bpSym = bpSym
        self._spSym = spSym
        self._sec_per_sym = self._spSym / Fs

    def Toggle(self):
        self._active = not self._active

    def Tick(self, t):
        if self._active:
            # Get next symbol from PRBS sequence, at iseq and bpSym in len
            self._iseq = int(np.floor(t / self._sec_per_sym) * self._bpSym) % self._nseq

            I = self._seq[self._iseq] * 2 - 1
            Q = self._seq[self._iseq+1] * 2 - 1
            samp = I*np.cos(2*np.pi*fc*t) + Q*np.sin(2*np.pi*fc*t)
            
            self._txQ.put((t,samp))

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
            self._txQ.put((rx[0], rx[1] + AWGN(self._P_n)))

            self._rxs += 1

            if self._rxs > 100000:
                self._rxs = 0
                # DEBUG: log here?
        else:
            self._txQ.put((t, AWGN(self._P_n)))

class Rx:
    def __init__(self):
        global rxQ
        self._rxQ = rxQ

        self._nvals = pow(2,15)
        self._vals = np.zeros(self._nvals)
        self._tvals = np.zeros(self._nvals)
        self._ivals = 0

        self._I_rcv = np.zeros(self._nvals)
        self._Q_rcv = np.zeros(self._nvals)
        self._I = np.zeros(self._nvals)
        self._Q = np.zeros(self._nvals)

        self._paused = False
        
        plt.figure()
        plt.ion()
        self._Plot()
        
        self._bpSym = bpSym
        self._spSym = spSym
        self._sec_per_sym = self._spSym / Fs
        self._isymsamples = 0
        self._symsamps = np.zeros(self._spSym)

    def Toggle(self):
        self._paused = not self._paused

    def _Plot(self):
        plt.clf()
        plt.subplot(221); plt.plot(self._tvals, self._vals); plt.ylim(-1,1)
        plt.subplot(222); plt.plot(self._I_rcv); plt.ylim(-1,1)
        plt.subplot(223); plt.plot(self._I); plt.ylim(-1,1)
        plt.subplot(224); plt.plot(self._I, self._Q, "."); plt.xlim(-1,1); plt.ylim(-1,1)
        plt.pause(0.1)

    def Tick(self, t):
        if not self._rxQ.empty():
            rx = self._rxQ.get()
            self._tvals[self._ivals] = rx[0]
            self._vals[self._ivals] = rx[1]

            self._ivals += 1

            if self._ivals >= self._nvals:
                self._ivals = 0

                self._I_rcv = self._vals * np.cos(2*np.pi*fc*self._tvals)
                self._Q_rcv = self._vals * np.sin(2*np.pi*fc*self._tvals)

                N = len(self._I_rcv)
                M = len(b)
                self._I = np.convolve(self._I_rcv, b)[M:N]
                self._Q = np.convolve(self._Q_rcv, b)[M:N]

                if not self._paused:
                    self._Plot()

if __name__ == "__main__":
    main()
