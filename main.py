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
txQ = Queue()
KILL = False

Fs = 44100.23
fc = 1000
b = np.genfromtxt("filter.csv", delimiter=",")

np.random.seed(123456)
SEQ = np.random.randint(2, size=16384)
bpSym = 2
spSym = 5000

constellation = [
    np.array([ 1.0,  1.0]), # 0
    np.array([-1.0,  1.0]), # 1
    np.array([-1.0, -1.0]), # 2
    np.array([ 1.0, -1.0]), # 3
]

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

            for k in act.split(" "):
                if k == "t":
                    tx.Toggle()
                    rx.SetTransmitterActive(tx.IsActive())
                elif k == "r":
                    rx.Toggle()

        tx.Tick(t)
        ch.Tick(t)
        rx.Tick(t)
        t += 1/Fs

        # DEBUG: is this needed? avoid overflow for various calculations
        if t > 100*Fs:
            t = 0

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
        global txQ
        self._txQ = chQ
        self._active = False
        self._rxQ = txQ

        self._iseq = 0

        global bpSym
        global spSym
        global SEQ
        global Fs
        global constellation
        self._seq = SEQ
        self._nseq = len(self._seq)

        self._bpSym = bpSym
        self._spSym = spSym
        self._sec_per_sym = self._spSym / Fs

        self._constellation = constellation

    def Toggle(self):
        self._active = not self._active
    
    def IsActive(self):
        return self._active

    def Tick(self, t):
        if self._active:
            # Get next symbol from PRBS sequence, at iseq. bpSym in len
            self._iseq = int(np.floor(t / self._sec_per_sym) * self._bpSym) % self._nseq

            vals = self._seq[self._iseq:self._iseq+self._bpSym]
            v = 0 # this is the decimal value of the next symbol
            for i in range(0,self._bpSym):
                v += vals[i] * 2**(self._bpSym-i-1)
            sym = self._constellation[v]

            I = sym[0]
            Q = sym[1]
            samp = I*np.cos(2*np.pi*fc*t) + Q*np.sin(2*np.pi*fc*t)
            
            self._txQ.put((t,samp))
        
        if not self._rxQ.empty():
            rx = self._rxQ.get()
            Log("tx got SNR=" + str(rx["SNR"]) + ", BER=" + str(rx["BER"]))

class Ch:
    def __init__(self):
        global chQ
        self._rxQ = chQ
        global rxQ
        self._txQ = rxQ

        self._rxs = 0

        self._P_n = 0.05

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
        global txQ
        self._rxQ = rxQ
        self._txQ = txQ

        self._nvals = pow(2,15)
        self._vals = np.zeros(self._nvals)
        self._tvals = np.zeros(self._nvals)
        self._ivals = 0

        self._I_rcv = np.zeros(self._nvals)
        self._Q_rcv = np.zeros(self._nvals)
        self._I = np.zeros(self._nvals)
        self._Q = np.zeros(self._nvals)

        self._paused = False

        global bpSym
        global spSym
        global SEQ
        global Fs
        global constellation

        self._seq = SEQ
        self._nseq = len(self._seq)
        
        self._bpSym = bpSym
        self._spSym = spSym
        self._sec_per_sym = self._spSym / Fs
        self._isymsamples = 0
        self._Isymsamps = np.zeros(self._spSym)
        self._Qsymsamps = np.zeros(self._spSym)

        self._transmitter_active = False
        self._powers_noise = []
        self._powers_signal = []
        self._P_s_n = 0 # this is an averaged value of Power of signal + noise
        self._P_n = 0 # this is an averaged value of Power of noise
        self._nSNR_averages = 5

        self._bits_total = 1
        self._bits_correct = 1
        self._BER = 1

        self._constellation = constellation
        
        plt.figure()
        plt.ion()
        self._Plot()

    def SetTransmitterActive(self, b):
        self._transmitter_active = b

    def Toggle(self):
        self._paused = not self._paused

    def _Plot(self):
        plt.clf()
        plt.subplot(221); plt.plot(self._tvals, self._vals); plt.ylim(-5,5); plt.title("Received Signal")
        plt.subplot(222); plt.plot(np.abs(np.fft.fft(self._vals, 1024))); plt.title("DFT of Received Signal")
        plt.subplot(223); plt.plot(self._I); plt.plot(self._Q); plt.ylim(-1.2,1.2); plt.title("Baseband IQ Data Over Time")
        plt.subplot(224); plt.plot(self._I, self._Q, "."); plt.xlim(-1.7,1.7); plt.ylim(-1.7,1.7); plt.title("IQ Diagram")
        plt.pause(0.1)

    def Tick(self, t):
        if not self._rxQ.empty():
            rx = self._rxQ.get()
            self._tvals[self._ivals] = rx[0]
            self._vals[self._ivals] = rx[1]
            
            self._ivals += 1

            if self._ivals >= self._nvals:
                self._ivals = 0

                power = 1 / self._nvals * np.sum(np.square(self._vals))
                power_dB = 10*np.log10(power)
                Log("power = " + str(power_dB) + " dB")

                if self._transmitter_active:
                    self._powers_signal.append(power)
                    self._P_s_n = np.mean(self._powers_signal)
                else:
                    self._powers_noise.append(power)
                    self._P_n = np.mean(self._powers_noise)

                # Only keep a few of the most recent entries in either array for SNR
                if len(self._powers_noise) > self._nSNR_averages:
                    del self._powers_noise[0]
                if len(self._powers_signal) > self._nSNR_averages:
                    del self._powers_signal[0]
                
                if self._P_s_n > self._P_n and self._P_n > 0:
                    SNR = (self._P_s_n - self._P_n) / self._P_n
                    SNR_dB = 10*np.log10(SNR)
                    Log("SNR = " + str(SNR_dB) + " dB")

                self._I_rcv = 2 * self._vals * np.cos(2*np.pi*fc*self._tvals)
                self._Q_rcv = 2 * self._vals * np.sin(2*np.pi*fc*self._tvals)

                N = len(self._I_rcv)
                M = len(b)
                self._I = np.convolve(self._I_rcv, b)[M:N]
                self._Q = np.convolve(self._Q_rcv, b)[M:N]
                
                nCorrect = 0
                nTotal = 0
                for i in range(0, len(self._I)):
                    nt = self._tvals[i]
                    idx = int(np.floor(nt / self._sec_per_sym * self._spSym)) % self._spSym

                    # at the start of a new symbol
                    if idx < self._isymsamples:
                        v = np.array([np.mean(self._Isymsamps), np.mean(self._Qsymsamps)])
                        dists = np.zeros(len(self._constellation))

                        for (i,c) in enumerate(self._constellation):
                            dist = np.linalg.norm(v - c)
                            dists[i] = dist
                        guess_val = np.argmin(dists)

                        # TODO: not sure why the "- self._bpsym" is needed here. Investigate
                        iseq = int(np.floor(nt / self._sec_per_sym) * self._bpSym) % self._nseq - self._bpSym
                        target = self._seq[iseq:iseq+self._bpSym]
                        
                        if len(target) > 0:
                            target_val = 0 # this is the decimal value of the next symbol
                            for i in range(0,self._bpSym):
                                target_val += target[i] * 2**(self._bpSym-i-1)

                            nTotal += 1
                            if target_val == guess_val:
                                nCorrect += 1

                    self._Isymsamps[idx] = self._I[i]
                    self._Qsymsamps[idx] = self._Q[i]
                    self._isymsamples = idx
                    
                # Calculate BER
                if self._transmitter_active:
                    self._bits_total += nTotal * self._bpSym
                    self._bits_correct += nCorrect * self._bpSym
                    self._BER = 1 - self._bits_correct / self._bits_total

                    # Output how many symbols were correct
                    Log("Got " + str(nCorrect) + "/" + str(nTotal) + " correct this time.")
                    Log("BER = " + str(self._BER))
                    Log("-----------------------------------------------------")
                    self._txQ.put({"SNR":SNR_dB, "BER":self._BER})

                if not self._paused:
                    self._Plot()

if __name__ == "__main__":
    main()
