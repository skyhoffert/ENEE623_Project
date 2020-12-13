# Sky Hoffert
# ENEE623 Project main class.

import matplotlib.pyplot as plt
import msvcrt
import numpy as np
import threading
import time
from queue import Queue

from util import Log

kbQ = Queue()
KILL = False

Fs = 44100.23
fc = 1000
b = np.genfromtxt("filter.csv", delimiter=",")

np.random.seed(123456)
SEQ = np.random.randint(2, size=16384)
np.random.seed(int(time.time()))

SCORE_WEIGHTS = {
    "BER": 0,
    "SNR": 0,
    "TxPut": 1,
    "Pavg": 5,
}
M_MUTATION_CHANCE = 0.2 # Chance for M to change by +/- 1.
SPSYM_STDDEV = 20 # stddev for SpSym value (tests suggested 50 was a good value)
CPT_STDDEV = 0.2 # stddev for constellation point value (with base power of 1.0)

def main():
    Log("Main program started.")

    kb = threading.Thread(target=ThreadKB)
    kb.start()

    # constellation_bpsk = [
    #     np.array([1.0,0.0]), # 0
    #     np.array([-1.0,0.0]) # 1
    # ]
    # constellation_qpsk = [
    #     np.array([ 1.0,  1.0]), # 0
    #     np.array([-1.0,  1.0]), # 1
    #     np.array([-1.0, -1.0]), # 2
    #     np.array([ 1.0, -1.0]), # 3
    # ]

    # M = RandomM()
    # nc = RandomConstellation(M)
    # Log("randM=" + str(M))
    # x,y = ConstellationToXY(nc)
    # plt.plot(x,y,"*")
    # plt.show()

    nPop = 6
    population = []
    for i in range(0,nPop):
        M = RandomM()
        population.append(Individual(M, RandomSpSym(), RandomConstellation(M)))
    nGen = 0

    tReport = 20
    tTurnOnTx = 5 # this depends on nSNR_averages
    TxEnabled = False

    global KILL
    global kbQ
    
    # Prepare the matplotlib figre window to be always up.
    plt.figure()
    plt.ion()
    reports = []

    t = 0
    while not KILL:
        if not kbQ.empty():
            act = kbQ.get()

            for k in act.split(" "):
                if k == "t":
                    for ind in population:
                        ind.TxToggle()
                elif k == "r":
                    for ind in population:
                        ind.RxToggle()

        for ind in population:
            ind.Tick(t)

        dT = 1/Fs
        t += dT

        # DEBUG: is this needed? avoid overflow for various calculations
        if t > 100*Fs:
            t = 0

        # Here, a generation reproduces
        if t > tReport:
            # clear figure before overwriting
            plt.clf()

            Log("---- Generation " + str(nGen) + " ----")

            reports = []

            for i,ind in enumerate(population):
                rep = ind.Report()
                Log(str(i) + ": " + str(rep))
                Log("Score=" + str(ScoreReport(rep)))
                reports.append(rep)
                
                x,y = ConstellationToXY(rep["constellation"])
                plt.subplot(331 + i); plt.plot(x,y,"*"); plt.title("Individual " + str(i))
                plt.pause(0.01)
            
            # Reset the simulation for the next generation
            t = 0
            TxEnabled = False
            population = NewGeneration(reports)
            nGen += 1
        
        # Handle automation actions that run the GA
        if TxEnabled == False:
            if t >= tTurnOnTx:
                Log("---- Enabled Transmitter ----")
                TxEnabled = True
                for ind in population:
                    ind.TxToggle()

    # If we reach here, the program has been halted -> save the population

    kb.join()

    f = open("file.txt", "w")
    for i,ind in enumerate(population):
        f.write(str(ind.Report()))
        f.write("\n")
    f.close()

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

def RandomM(t="uniform"):
    if t == "uniform":
        return 2**np.random.randint(1,5)

def RandomConstellationPoint(pmax=1.0):
    return np.array([np.random.uniform(-pmax,pmax), np.random.uniform(-pmax,pmax)])

def RandomConstellation(M):
    c = []
    for i in range(0,M):
        c.append(RandomConstellationPoint())
    return c

def RandomSpSym():
    return np.random.randint(1,200) # TODO: is this a good range? arbitrarily chosen

def ConstellationToXY(c):
    xs = []
    ys = []
    for pt in c:
        xs.append(pt[0])
        ys.append(pt[1])
    return (xs,ys)

def ScoreReport(rep):
    # We care about BER, SNR, TxPut, and Pavg
    return 1/rep["BER"] * SCORE_WEIGHTS["BER"] + \
        rep["SNR"] * SCORE_WEIGHTS["SNR"] + \
        rep["TxPut"] * SCORE_WEIGHTS["TxPut"] + \
        1/rep["Pavg"] * SCORE_WEIGHTS["Pavg"]

def NewGeneration(reps):
    scores = np.zeros(len(reps))
    for i,rep in enumerate(reps):
        scores[i] = ScoreReport(rep)
    rankings = np.flipud(np.argsort(scores))
    Log("rankings:" + str(rankings))
    
    newpop = []

    # Keep the top half of the individuals
    nKeep = round(len(rankings)/2 + 0.1)
    for i,r in enumerate(rankings):
        if i > nKeep-1:
            break
        newpop.append(Individual(reps[r]["M"], reps[r]["spSym"], reps[r]["constellation"]))

    # Breed and mutate for the rest of the population
    nNew = len(rankings) - nKeep
    for i in range(0,nNew):
        # Randomly choose mate1
        mate1 = np.random.randint(0,nKeep)

        # Randomly choose mate2, don't choose mate1 as mate2
        mate2 = mate1
        while mate2 == mate1:
            mate2 = np.random.randint(0,nKeep)
        
        # Decide what value of M to take
        b1 = np.log2(reps[rankings[mate1]]["M"])
        b2 = np.log2(reps[rankings[mate2]]["M"])
        # average the two Ms, select closest value
        b = round((b1 + b2)/2)
        # randomly add or subtract
        if np.random.uniform() < M_MUTATION_CHANCE:
            Log("---- Mutation of M from " + str(2**b), end="")
            b += (2 * np.random.randint(0,2)) - 1
            Log(" to " + str(2**b) + " ----")
        M = 2**b
        # M must be at least 2
        if M < 2:
            M = 2

        # Decide what value of SpSym to take
        s = round((reps[rankings[mate1]]["spSym"] + reps[rankings[mate1]]["spSym"])/2 + np.random.normal(0,SPSYM_STDDEV))
        if s < 2:
            s = 2

        # Decide what constellation to use
        wParent = mate1
        if np.random.uniform() > 0.5:
            wParent = mate2
        constellation = []
        for j,c in enumerate(reps[rankings[wParent]]["constellation"]):
            constellation.append(np.array(c))
        for j in range(0,len(constellation)):
            c = constellation[j]
            constellation[j][0] = c[0] + np.random.normal(0,CPT_STDDEV)
            constellation[j][1] = c[1] + np.random.normal(0,CPT_STDDEV)
        nPtsNeeded = M - len(constellation)
        for j in range(0,nPtsNeeded+1):
            constellation.append(RandomConstellationPoint())

        newpop.append(Individual(M, s, constellation))

    return newpop

class Individual:
    def __init__(self, M, spSym, constellation):
        self._M = M
        self._bpSym = int(np.log2(self._M))
        self._spSym = spSym

        self._constellation = constellation

        self._chQ = Queue()
        self._rxQ = Queue()
        self._txQ = Queue()

        self._tx = Tx(self._chQ, self._txQ, self._M, self._spSym, self._constellation)
        self._ch = Ch(self._rxQ, self._chQ)
        self._rx = Rx(self._txQ, self._rxQ, self._M, self._spSym, self._constellation)

    def Tick(self, t):
        self._tx.Tick(t)
        self._ch.Tick(t)
        self._rx.Tick(t)

    def TxToggle(self):
        self._tx.Toggle()
        self._rx.SetTransmitterActive(self._tx.IsActive())

    def RxToggle(self):
        self._rx.Toggle()

    def Report(self):
        return {
            "M": self._M,
            "spSym": self._spSym,
            "constellation": self._constellation,
            "BER": self._tx.GetBER(),
            "SNR": self._tx.GetSNR(),
            "TxPut": self._tx.GetTxPut(),
            "Pavg": self._tx.GetPavg(),
        }

class Tx:
    def __init__(self, txQ, rxQ, M, spSym, constellation):
        self._active = False

        self._txQ = txQ
        self._rxQ = rxQ

        self._iseq = 0

        self._seq = SEQ
        self._nseq = len(self._seq)

        self._M = M
        self._bpSym = int(np.log2(self._M))
        self._spSym = spSym
        self._sec_per_sym = self._spSym / Fs

        self._constellation = constellation

        self._BER = 1
        self._SNR = 0
        self._txPut = int(Fs * self._bpSym / self._spSym) # bps

        # calculate average power based on constellation
        self._Pavg = 0
        for i,c in enumerate(self._constellation):
            self._Pavg += c[0]**2 + c[1]**2
        self._Pavg /= self._M

    def GetBER(self):
        return self._BER

    def GetSNR(self):
        return self._SNR

    def GetTxPut(self):
        return int(self._txPut * (1-self._BER)) # Depends on BER, in units of bps

    def GetPavg(self):
        return self._Pavg

    def Toggle(self):
        self._active = not self._active
    
    def IsActive(self):
        return self._active

    def Tick(self, t):
        if self._active:
            # Get next symbol from PRBS sequence, at iseq. bpSym in len
            self._iseq = int(np.floor(t / self._sec_per_sym) * self._bpSym) % self._nseq

            vals = np.array(self._seq[self._iseq:self._iseq+self._bpSym])

            # Needed in case bpsym is odd, don't go over the end of SEQ
            needvals = self._bpSym - len(vals)
            if needvals > 0:
                for i in range(0,needvals):
                    vals = np.append(vals, self._seq[i])

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
            self._BER = rx["BER"]
            self._SNR = rx["SNR"]

class Ch:
    def __init__(self, txQ, rxQ):
        self._rxQ = rxQ
        self._txQ = txQ

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
    def __init__(self, txQ, rxQ, M, spSym, constellation):
        self._rxQ = rxQ
        self._txQ = txQ

        self._nvals = pow(2,14)
        self._vals = np.zeros(self._nvals)
        self._tvals = np.zeros(self._nvals)
        self._ivals = 0

        self._I_rcv = np.zeros(self._nvals)
        self._Q_rcv = np.zeros(self._nvals)
        self._I = np.zeros(self._nvals)
        self._Q = np.zeros(self._nvals)

        self._paused = True

        self._seq = SEQ
        self._nseq = len(self._seq)
        
        self._M = M
        self._bpSym = int(np.log2(self._M))
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
        
        # plt.figure()
        # plt.ion()
        # self._Plot()

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
                # Log("power = " + str(power_dB) + " dB")

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
                    # Log("SNR = " + str(SNR_dB) + " dB")

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
                    
                # TODO: fix BER, currently broken
                # TODO: add metric for Power utilization! how much energy influences a constellation

                # Calculate BER
                if self._transmitter_active:
                    self._bits_total += nTotal * self._bpSym
                    self._bits_correct += nCorrect * self._bpSym
                    self._BER = 1 - self._bits_correct / self._bits_total

                    # Output how many symbols were correct
                    # Log("Got " + str(nCorrect) + "/" + str(nTotal) + " correct this time.")
                    # Log("BER = " + str(self._BER))
                    # Log("-----------------------------------------------------")
                    self._txQ.put({"SNR":SNR_dB, "BER":self._BER})

                if not self._paused:
                    self._Plot()

if __name__ == "__main__":
    main()
