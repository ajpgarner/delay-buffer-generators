"""
plot_delay_chains.py
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
import numpy as np
import matplotlib.pyplot as plt

delay_length = np.arange(16, dtype=int)
# even_chain = np.load("even_chain.npy")
pc_chain = np.load("pc_chain.npy")
# rgm_chain = np.load("rgm_chain.npy")
nemo_chain = np.load("nemo_chain.npy")

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(delay_length, pc_chain, label="Perturbed coin (p=0.05)")
ax.plot(delay_length, nemo_chain, label="Nemo process (p=0.4, q=0.1)")
ax.set_xlabel("Delay length (steps)")
ax.set_ylabel("Dissipation per output (kT)")
ax.legend()
plt.xlim([0, 15])
plt.ylim([0, 0.3])
fig.show()
fig.savefig("delay_dissipation.pdf", format="pdf")
