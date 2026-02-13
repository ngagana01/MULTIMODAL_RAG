class ChatMemory:
    def __init__(self, size=5):
        self.size=size
        self.history=[]

    def add(self,q,a):
        self.history.append((q,a))
        self.history=self.history[-self.size:]

    def context(self):
        return "\n".join([f"Q:{q}\nA:{a}" for q,a in self.history])
