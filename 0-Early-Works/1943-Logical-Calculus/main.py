class Neuron:
    def __init__(self, inhibitory_inputs=[], excitatory_sets=[]) -> None:
        self.inhibitory_inputs = inhibitory_inputs
        self.excitatory_sets = excitatory_sets
        self.out = False

    def activate(self):
        self.out = True

    def deactivate(self):
        self.out = False

    def __repr__(self) -> str:
        inhibitory_repr = [e.out for e in self.inhibitory_inputs]

        excitatory_repr = [[e.out for e in s] for s in self.excitatory_sets]

        return (
            f"Inhibitory Inputs: {inhibitory_repr}\n"
            f"Excitatory Sets: {excitatory_repr}\n"
            f"Output: {self.out}\n ----"
        )

    def call(self):
        output = True
        for excitatory_sets in self.excitatory_sets:
            set_output = True
            for excitatory in excitatory_sets:
                set_output = set_output and excitatory.out
            output = output or set_output

        for inhibitory_input in self.inhibitory_inputs:
            output = output and not inhibitory_input.out
        self.out = output


n1 = Neuron()
n2 = Neuron()
n3 = Neuron()
n4 = Neuron()

nOut = Neuron([n4], [[n1, n2], [n3]])


def reset():
    n1.deactivate()
    n2.deactivate()
    n3.deactivate()
    n4.deactivate()


n1.activate()
n2.activate()

nOut.call()

print(nOut)

reset()

n1.activate()
n3.activate()

nOut.call()

print(nOut)

reset()

n1.activate()
n3.activate()
n4.activate()

nOut.call()

print(nOut)
