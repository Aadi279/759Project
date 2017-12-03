from threadSim import Grid
import numpy as np
from sys import maxint


def localScans(self, inp, outp, blockSums, N):
    # Global and local thread indices
    gtid = self.threadId + self.blockId * self.blockDim
    tid = self.threadId

    if gtid < N:
        self.shared[self.threadId] = inp[gtid]
    yield 0

    stride = 1
    scanStart = 0
    scanEnd = -1
    while stride < self.blockDim:
        numberOfElementsWereScanningThisRound = self.blockDim / stride
        scanEnd = scanStart + numberOfElementsWereScanningThisRound
        if tid < numberOfElementsWereScanningThisRound/2:
            self.shared[scanEnd + tid] = self.shared[scanStart + tid*2] + self.shared[scanStart + tid*2 + 1]
        stride *= 2
        scanStart = scanEnd
        yield 1

    if tid == 0:
        blockSums[self.blockId] = self.shared[self.blockDim*2-2]
    yield 2

    stride = 1
    tierStart = self.blockDim*2 - 2
    sum = 0
    remainder = tid + 1
    offsetAggregator = 0
    while stride <= self.blockDim:
        twoPow = self.blockDim / stride
        tierIndex = remainder / twoPow - 1

        if tierIndex >= 0:
            toAdd = self.shared[tierStart + tierIndex + offsetAggregator]
            sum += toAdd
            remainder -= twoPow * (tierIndex+1)
            offsetAggregator += (tierIndex+1)


        offsetAggregator *= 2
        stride *= 2
        tierStart -= stride

    if gtid < N:
        outp[gtid] = sum

    yield maxint

def addBlockScans(self, inp, blockScans, N):
    gtid = self.threadId + self.blockId * self.blockDim
    if self.blockId == 0:
        yield maxint

    if gtid < N:
        inp[gtid] += blockScans[self.blockId-1]

    yield maxint


def scanSeq(inp, outp, N):
    sum = 0
    i = 0
    while i < N:
        sum += inp[i]
        outp[i] = sum
        i += 1

N = 130
threadsPerBlock = 8


gridSize = N / threadsPerBlock + 1
sharedMem = 2 * threadsPerBlock

x_inp = np.array(range(N))
x_outp = np.array([0]*N)
x_blockSums = np.array([0]*gridSize)

# Sample for reference
x_outp_ref = np.array([0]*N)
scanSeq(x_inp, x_outp_ref, N)

def recursiveScan(curInp, curN):
    nextN = curN/threadsPerBlock+1
    blockSumBuff = np.array([0]*nextN)
    outp = np.array([0]*curN)
    if curN < threadsPerBlock:
        Grid(localScans, [curInp, outp, blockSumBuff, curN], 1, threadsPerBlock, sharedMem).run()
    else:
        Grid(localScans, [curInp, outp, blockSumBuff, curN], nextN, threadsPerBlock, sharedMem).run()
        blockScan = recursiveScan(blockSumBuff, nextN)
        Grid(addBlockScans, [outp, blockScan, curN], nextN, threadsPerBlock, 0).run()

    #deallocate
    return outp

x_outp = recursiveScan(x_inp, N)

test = np.all(x_outp == x_outp_ref)

pass
