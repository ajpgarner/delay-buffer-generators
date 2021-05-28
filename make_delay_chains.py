"""
make_delay_chains.py
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
"""
from delay_buffer_generators import *


def even_process(p: float):
    output = np.zeros([2, 2, 2])
    output[0, 0, 0] = p
    output[1, 1, 0] = 1 - p
    output[1, 0, 1] = 1
    name = "Even process (p = " + str(p) + ")"
    return HMMProcess((output, name))


def perturbed_coin(p: float):
    output = np.zeros([2, 2, 2])
    output[0, 0, 0] = 1 - p
    output[0, 0, 1] = p
    output[1, 1, 0] = p
    output[1, 1, 1] = 1 - p

    name = "Perturbed coin (p = " + str(p) + ")"
    return HMMProcess((output, name))


def restricted_golden_mean(k: int):
    output = np.zeros([2, k + 1, k + 1])
    output[0, 1, 0] = 0.5
    output[1, 0, 0] = 0.5
    for i in range(1, k + 1):
        output[1, (i + 1) % (k + 1), i] = 1
    name = "Restricted golden mean (K = " + str(k) + ")"
    return HMMProcess((output, name))


def nemo_process(p: float, q: float):
    output = np.zeros([2, 3, 3])
    output[1, 0, 0] = p
    output[0, 1, 0] = 1 - p
    output[0, 2, 1] = 1
    output[0, 0, 2] = 1 - q
    output[1, 0, 2] = q
    name = "Nemo process (p = " + str(p) + ", q = " + str(q) + ")"
    return HMMProcess((output, name))

# %% Script


max_delay = 15

rgm_chain = make_delay_chain(restricted_golden_mean(3), max_delay)
pc_chain = make_delay_chain(perturbed_coin(0.05), max_delay)
nemo_chain = make_delay_chain(nemo_process(0.4, 0.1), max_delay)
even_chain = make_delay_chain(even_process(0.3), max_delay)

np.save("rgm_chain.npy", rgm_chain)
np.save("pc_chain.npy", pc_chain)
np.save("nemo_chain.npy", nemo_chain)
np.save("even_chain.npy", even_chain)
