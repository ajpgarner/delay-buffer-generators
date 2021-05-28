# Delay Buffer Generators
The scripts in this repository calculate the delay buffer 
generators, and their associated thermal dissipation, from 
a given specification of the process's epsilon machine.

### `delay_buffer_generators.py` 
`delay_buffer_generators.py` defines the class `HMMProcess` 
and some example processes. `HMMProcess` reflects a Hidden
Markov Model of a process. It should be constructed by
supplying a tensor description of its epsilon-machine 
transitions (for details, see comments in `delay_buffer_generators.py`, 
or the examples in `make_delay_chains.py`).

`HMMProcess` has several member functions, the most important being
`delay_by_one()`, which creates a new model producing
the same process, but with an additional delay of one step.
`generative_dissipation()` calculates the unavoidable
thermal dissipation (in units of kT) when the model is used
to generate the pattern.
The class also has a ``__str__`` overload, such that 
"``print(model)``" outputs a human-readable description
of the model's memory and transition structure.

Finally, `make_delay_chain(process, length)`, takes a  `HMMProcess`
"process" and number indicating the maximum delay buffer length. It 
then returns an array, where index `i` indicates the generative 
dissipation associated with a delay of `i` (index `0` indicates the
epsilon-machine dissipation). 

### `make_delay_chains.py`
`make_delay_chains.py` calculates the dissipation for the
Even Process, the Perturbed Coin, the Restricted 
Golden Mean, and the Nemo Process, when produced by a generator
that delays the output by between 0 and 15 steps.


### `plot_delay_chains.py`
`plot_delay_chains.py` synthesizes the data produced by 
`make_delay_chains.py` into a chart using matplotlib.


## Citing this project
If you use this script or a derivative thereof in academic
research, I kindly ask that you consider citing the paper 
for which this script was originally developed:

*Exact citation to be determined.*

