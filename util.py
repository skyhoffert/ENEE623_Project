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

class Connection_Server:
    def __init__(self, p):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._addr = ("localhost", p)
        self._sock.bind(self._addr)

        self._rxQ = []
        self._txQ = []

    def Recv(self):
        pass
