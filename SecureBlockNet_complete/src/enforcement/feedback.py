class PolicyFeedback:
    def __init__(self, adaptation=0.1): self.state=0.0; self.adaptation=adaptation
    def update(self, adjustment):
        self.state=(1-self.adaptation)*self.state+self.adaptation*float(adjustment)
        return self.state
