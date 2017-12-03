import numpy as np
from sys import maxint

class Thread: # Simulated thread
    def __init__(self, kernel, kargs, blockId, blockDim, threadId, numBlocks, sharedMemRef):
        self.blockId = blockId
        self.threadId = threadId
        self.shared = sharedMemRef
        self.kargs = kargs
        self.kernel = kernel
        self.blockDim = blockDim
        self.gridDim = numBlocks
        self.start()

    def start(self):
        self.runningThread = self.kernel(self, *self.kargs)

    def step(self):
        # Blocks should not call step when threads have completed (yielded maxint)
        try:
            return self.runningThread.next()
        except StopIteration as e:
            print("Remember: Kerenel functions should yield sequential integers beginning with 0. Make sure you end the kernel function by yielding maxint")
            raise e


class Block:
    def __init__(self, kernel, kargs, blockId, threadsPerBlock, numBlocks, memSize):
        self.shared = np.array([0] * memSize)
        self.threads = []
        self.syncVals = np.array([-1] * threadsPerBlock) # keeps track of the last syncThreads each thread has seen
        for threadId in range(threadsPerBlock):
            self.threads.append(Thread(kernel, kargs, blockId, threadsPerBlock, threadId, numBlocks, self.shared))

        for thread in self.threads:
            thread.start()

    def step(self):
        # Halt the furthest progressed threads to let the rest catch up
        maxSyncVal = max(self.syncVals)

        # Check if all syncVals are the same
        # Signifies that we are on the same '__syncThreads()' call in a cuda program
        # This isn't fool-proof at all, and could use some tuning to make it robust
        allEq = len(set(self.syncVals)) <= 1

        for i, thread in enumerate(self.threads):
            if self.syncVals[i] != maxint and ((self.syncVals[i] < maxSyncVal) or allEq):
                self.syncVals[i] = thread.step()

        # Check if we're all done
        allEq = len(set(self.syncVals)) <= 1
        if allEq and (self.syncVals[0] == maxint):
            return "done"
        else:
            return "running"



class Grid: # Simulated thread pool
    def __init__(self, kernel, kargs, numBlocks, threadsPerBlock, memSize):
        self.blocks = []
        self.numBlocks = numBlocks
        for blockId in range(numBlocks):
            self.blocks.append(Block(kernel, kargs, blockId, threadsPerBlock, numBlocks, memSize))

    def step(self):
        blockStatuses = []
        for block in self.blocks:
            blockStatuses.append(block.step())

        if (len(set(blockStatuses)) <= 1) and (blockStatuses[0] == "done"):
            return "allDone"
        else:
            return "running"

    def run(self):
        while(self.step() != "allDone"):
            pass




def multiSyncThreads_sample(self):
    i = 0
    ctid = self.threadId + self.blockId * self.blockDim
    while(i < ctid):
        i += 1
        self.shared[ctid] = ctid
        print(self.threadId, self.blockId)
        yield 0

    print(ctid, "finished!")
    yield maxint

# Sample Usage: g = Grid(arguments_sample, [inp,outp], 2, 2, 10); g.step()
def arguments_sample(self, inp, out):
    ctid = self.threadId + self.blockId * self.blockDim
    out[ctid] = inp[ctid]
    yield maxint


if __name__ == "__main__":
    inp = [1] * 4
    outp = [0] * 4
    # g = Grid(arguments_sample, [inp,outp], 2, 2, 10)
    g = Grid(multiSyncThreads_sample, [], 2, 2, 10)
    g.run()
    pass
