"""
delay_buffer_generators.py
Copyright (C) 2021 Andrew J. P. Garner

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

---

Transition tensor indices are specified in the form: [x, j, i] where
    x labels the output symbol,
    j is the destination machine state,
    i is the source machine state.
The value of the tensor at such an index gives the conditional probability
(conditioned on the source machine state).

"""
import math
import numpy as np


def stationary_distribution(sm: np.array):
    the_vector = None
    evals, evecs = np.linalg.eig(sm)
    for index, value in enumerate(evals):
        if np.isclose(np.real_if_close(value), 1):
            the_vector = np.real_if_close(evecs[:, index])
            break
    if the_vector is None:
        raise Exception("Matrix did not have an eigenvalue close to 1")

    return the_vector / np.sum(the_vector)


def shannon_entropy(dist: np.array):
    def xlog2x(val):
        if val > 0:
            return val * math.log2(val)
        return 0

    output = 0
    for v in dist:
        output -= xlog2x(v)
    return output


class HMMProcess:
    def __init__(self, proc: tuple, eps=None, compound_memory=None):
        self.tx, self.name = proc
        assert (self.tx.ndim == 3)
        assert (self.tx.shape[1] == self.tx.shape[2])

        self.num_outputs = self.tx.shape[0]
        self.num_memory_states = self.tx.shape[1]
        self.stochastic_matrix = np.sum(self.tx, 0)
        self.emission_matrix = np.sum(self.tx, 1)

        self.stationary_memory = stationary_distribution(self.stochastic_matrix)

        if eps is None:
            self.eps = self
        else:
            self.eps = eps

        if compound_memory is None:
            self.mem_compound = {x: tuple([x])
                                 for x in range(self.num_memory_states)}
        else:
            self.mem_compound = compound_memory

    def memory_entropy(self):
        return shannon_entropy(self.stationary_memory)

    def __str__(self):
        def str_mem_compound(compound):
            smout = "("
            for x in compound[0:-1]:
                smout += str(x) + ", "
            smout += "S" + str(compound[-1]) + ")"
            return smout

        output = self.name + ":\n"
        for i in range(self.num_memory_states):
            output += "Mem" + str(i) + " = "
            output += str_mem_compound(self.mem_compound[i])
            output += "  [P=%f]:\n" % (self.stationary_memory[i])
            targets = np.argwhere(self.tx[:, :, i])
            for target in targets:
                output_symbol = target[0]
                target_mem_state = target[1]
                decoded_target = self.mem_compound[target_mem_state]
                probability = self.tx[output_symbol, target_mem_state, i]

                output += " -> Mem" + str(target_mem_state) + " = "
                output += str_mem_compound(decoded_target)
                output += "  [Out=" + str(output_symbol) + ", "
                output += "p=" + str(probability) + "]\n"

        return output

    def entropy_rate(self):
        conditional_entropy = np.zeros(self.eps.num_memory_states)
        for i in range(self.eps.num_memory_states):
            conditional_entropy[i] = shannon_entropy(np.sum(self.eps.tx[:, :, i], 1))
        return np.inner(self.eps.stationary_memory, conditional_entropy)

    def next_step_entropy(self):
        # Contract memory states with probability of being in that memory state
        # then "flatten" to get joint distribution over final state and outputs.
        next_step_matrix = np.tensordot(self.tx,
                                        self.stationary_memory,
                                        axes=([2], [0])).flatten()
        return shannon_entropy(next_step_matrix)  # H(R_1 X_1)

    def generative_dissipation(self):
        return (self.memory_entropy() + self.entropy_rate()) \
               - self.next_step_entropy()

    def delay_by_one(self, verbose: bool = False):

        # Enumerate non-zero sequences for each causal state
        delayed_mem_states = {}
        state_num = 0
        if verbose:
            print("Input memory states: " + str(self.mem_compound))
        for mem_sequence in self.mem_compound.values():
            causal_state = mem_sequence[-1]
            destination_list = np.argwhere(self.eps.tx[:, :, causal_state])
            for destination in destination_list:
                delayed_output = destination[0]
                final_causal_state = destination[1]
                memkey = (tuple([delayed_output])
                          + mem_sequence[0:-1] + tuple([final_causal_state]))
                if memkey not in delayed_mem_states:
                    delayed_mem_states[memkey] = state_num
                    state_num += 1

        inv_mem_states = {v: k for k, v in delayed_mem_states.items()}
        if verbose:
            print("Delayed Mem States: " + str(inv_mem_states))

        # Create transition matrix
        num_delay_states = len(delayed_mem_states)
        next_tx = np.zeros([self.num_outputs, num_delay_states, num_delay_states])
        for source_key, source_index in delayed_mem_states.items():
            if verbose:
                print(str(source_index) + ": " + str(source_key))

            output_symbol = source_key[-2]
            init_causal_state = source_key[-1]
            destination_list = np.argwhere(self.eps.tx[:, :, init_causal_state])
            for destination in destination_list:
                final_mem_state = destination[1]
                delayed_symbol = destination[0]
                destination_key = (tuple([delayed_symbol])
                                   + source_key[0:-2] + tuple([final_mem_state]))
                destination_index = delayed_mem_states[destination_key]
                prob = self.eps.tx[destination[0], destination[1], init_causal_state]
                if verbose:
                    print("-> " + str(destination_key) + " = " + str(destination_index) +
                          " (Out=" + str(output_symbol) + ", p=" + str(prob) + ")")
                next_tx[output_symbol, destination_index, source_index] = prob

        # Make new name
        delay_length = (len(inv_mem_states[0])-1)
        delayed_name = self.eps.name + " [Delay = " + str(delay_length) + "]"

        return HMMProcess((next_tx, delayed_name), self.eps, inv_mem_states)


def make_delay_chain(input_process, delay_length: int, verbose:bool = False):
    output = np.zeros(delay_length + 1)
    the_epsilon = input_process
    process = the_epsilon
    for i in range(delay_length + 1):
        if i != 0:
            if verbose:
                print("Delaying by " + str(i) + "...")
            process = process.delay_by_one(verbose)
        if verbose:
            print("Calculating entropies...")
        output[i] = process.generative_dissipation()
    return output



