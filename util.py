# Sky Hoffert
# Utility stuff for ENEE623 Project.

import socket
import sys
import threading

PORT_TX_TO_CH = 5000
PORT_CH_TO_RX = 5001

Fs = 44100

def Log(s, end="\n"):
    sys.stdout.write(s)
    sys.stdout.write(end)
    sys.stdout.flush()

def ConstellationToXY(c):
    xs = []
    ys = []
    for pt in c:
        xs.append(pt["I"])
        ys.append(pt["Q"])
    return (xs,ys)