# composite-fermions-mh
Monte Carlo Simulations of Fractional Quantum Hall States using the Composite Fermion approach
following LogDensityProblems.jl and AdvancedMH.jl projects, the code is organized as follows (to be updated):
1. A system object is created for every MCMC run (chain). 
2. Associated with a system object of a given type is a logpdf function. 
3. Metropolis-Hastings algorithm is used as the basis for MCMC.
4. For thermalization, we use simulated annealing with a cooling schedule.
5. Sampling is done in batches. Associated with a batch run, is a samples object. 
6. We adapt the step size at every iteration according to ARM scheme to achieve an ideal acceptance ratio.
7. Generic proposal distributions are allowed. As long as they are markovian.

To avoid non-markovian bias, it is important that the batch size is greater than or equal to the square root of total number of samples.
