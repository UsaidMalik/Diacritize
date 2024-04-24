from collections import deque

class IndexableQueue:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize)

    def put(self, item):
        if len(self.queue) == self.maxsize:
            self.queue.popleft()  # remove the oldest item
        self.queue.append(item)

    def pop(self):
        return self.queue.popleft()
    
    def full(self):
        return len(self.queue) == self.maxsize

    def __getitem__(self, index):
        if isinstance(index, slice):
            returned = []
            step = 1 if not index.step else index.step
            for i in range(index.start, index.stop, step):
                returned.append(self.queue[i])
            return returned
        else:
            return self.queue[index]

    def __len__(self):
        return len(self.queue)
    
    def __str__(self) -> str:
        queue : str = "["
        for i in range(self.maxsize):
            if i < len(self.queue):
                queue += f' {self.queue[i]},'
            else:
                queue += f' ...,'
        queue += "]"
        return queue
