
import matplotlib.pyplot as plt
import numpy as np
import sys
import json

from util import Log, ConstellationToXY

def main():
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        Log("ERR. Incorrent number of args.")
        return

    if sys.argv[1] != "-f":
        Log("ERR. No '-f' param given.")
        return

    wp = -1
    if len(sys.argv) > 3:
        if sys.argv[3] != "-n":
            Log("ERR. No -n param given")
            return 
        wp = int(sys.argv[4])

    f = open(sys.argv[2], "r")
    rep = json.load(f)
    # Log(json.dumps(rep, indent=2, sort_keys=True))

    Log("Plotting generation " + str(rep["Reports"][wp]["Generation"]))
    for i,r in enumerate(rep["Reports"][wp]["Reports"]):
        x,y = ConstellationToXY(r["constellation"])
        plt.subplot(331+i); plt.plot(x,y,"*"); plt.xlim([-2,2]); plt.ylim([-2,2]); plt.grid(); plt.title("Individual " + str(i))
    plt.show()

if __name__ == "__main__":
    main()
