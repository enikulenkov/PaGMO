Algorithms
==========

A Quick Look
------------

Algorithms in PyGMO are objects, constructed and then used to optimize a problem via their evolve method. The user can
implement his own algorithm in Python (in which case they need to derive from :class:`PyGMO.algorithm.base`), but we
also provide a number of algorithms that are considered useful for general purposes. Each algorithm can be associated only to
problems of certain types: (Continuous, Integer or Mixed Integer)-(Constrained, Unconstrained)-(Single, Multi-objective).


Heuristic Optimization
^^^^^^^^^^^^^^^^^^^^^^
================================== ========================================= =============== ===========================================
Common Name                        Name in PyGMO                             Type            Comments
================================== ========================================= =============== ===========================================
Differential Evolution (DE)        :class:`PyGMO.algorithm.de`                    C-U-S 
Particle Swarm Optimization (PSO)  :class:`PyGMO.algorithm.pso`                   C-U-S      steady-state
Particle Swarm Optimization (PSO)  :class:`PyGMO.algorithm.pso_gen`               C-U-S      generational (also problems deriving from base_stochastic)
Simple Genetic Algorithm (SGA)     :class:`PyGMO.algorithm.sga`                  MI-U-S 
Corana's Simulated Annealing (SA)  :class:`PyGMO.algorithm.sa_corana`             C-U-S 
Artificial Bee Colony (ABC)        :class:`PyGMO.algorithm.bee_colony`            C-U-S 
Cross-Entropy Method (CE)          :class:`PyGMO.algorithm.py_cross_entropy`      C-U-S      Written in Python
Improved Harmony Search (IHS)      :class:`PyGMO.algorithm.ihs`                  MI-U-M      Integer and Multiobjetive not tested yet
Monte Carlo Search (MC)            :class:`PyGMO.algorithm.monte_carlo`          MI-C-S
Monte Carlo Search (MC)            :class:`PyGMO.algorithm.py_example`           MI-C-S      Written in Python
NSGA-II	                                                                                     Coming soon!!! (under testing)
Covariance Matrix Adaptation-ES 	                                                     Coming soon!!! (under testing) 
Ant Colony Optimization                                                                      Planned
================================== ========================================= =============== ===========================================

Meta-algorithms 
^^^^^^^^^^^^^^^
================================== ========================================= =============== ===========================================
Common Name                        Name in PyGMO                             Type            Comments
================================== ========================================= =============== ===========================================
Monotonic Basin Hopping (MBH)      :class:`PyGMO.algorithm.mbh`                    N/A          
Multistart (MS)                    :class:`PyGMO.algorithm.ms`                     N/A      
Penalty Function (PF)                                                                        Planned 
Augmented Lagrangian (AL)                                                                    Planned
================================== ========================================= =============== ===========================================

Local optimization 
^^^^^^^^^^^^^^^^^^
================================== ========================================= =============== =====================================================================
Common Name                        Name in PyGMO                             Type            Comments
================================== ========================================= =============== =====================================================================
Compass Search (CS)                :class:`PyGMO.algorithm.cs`                    C-U-S 
Nelder-Mead simplex                :class:`PyGMO.algorithm.scipy_fmin`            C-U-S      SciPy required. Minimization assumed
Nelder-Mead simplex                :class:`PyGMO.algorithm.gsl_nm`                C-U-S      Requires PyGMO to be compiled with GSL option. Minimization assumed
Nelder-Mead simplex variant 2      :class:`PyGMO.algorithm.gsl_nm2`               C-U-S      Requires PyGMO to be compiled with GSL option. Minimization assumed
Nelder-Mead simplex variant 2r     :class:`PyGMO.algorithm.gsl_nm2rand`           C-U-S      Requires PyGMO to be compiled with GSL option. Minimization assumed
Subplex (a Nelder-Mead variant)    :class:`PyGMO.algorithm.nlopt_sbplx`           C-C-S      Requires PyGMO to be compiled with nlopt option. Minimization assumed
L-BFGS-B                           :class:`PyGMO.algorithm.scipy_l_bfgs_b`        C-U-S      SciPy required. Minimization assumed
BFGS                               :class:`PyGMO.algorithm.gsl_bfgs2`             C-U-S      Requires PyGMO to be compiled with GSL option. Minimization assumed
BFGS 2                             :class:`PyGMO.algorithm.gsl_bfgs`              C-U-S      Requires PyGMO to be compiled with GSL option. Minimization assumed
Sequential Least SQuares Prog.     :class:`PyGMO.algorithm.scipy_slsqp`           C-C-S      SciPy required. Minimization assumed
Truncated Newton Method            :class:`PyGMO.algorithm.scipy_tnc`             C-U-S      SciPy required. Minimization assumed
Conjugate Gradient (fr)            :class:`PyGMO.algorithm.gsl_fr`                C-U-S      Requires PyGMO to be compiled with GSL option. Minimization assumed
Conjugate Gradient (pr)            :class:`PyGMO.algorithm.gsl_pr`                C-U-S      Requires PyGMO to be compiled with GSL option. Minimization assumed
COBYLA                             :class:`PyGMO.algorithm.scipy_cobyla`          C-C-S      SciPy required. Minimization assumed
COBYLA                             :class:`PyGMO.algorithm.nlopt_cobyla`          C-C-S      Requires PyGMO to be compiled with nlopt option. Minimization assumed
BOBYQA                             :class:`PyGMO.algorithm.nlopt_bobyqa`          C-C-S      Requires PyGMO to be compiled with nlopt option. Minimization assumed
SNOPT                              :class:`PyGMO.algorithm.snopt`                 C-C-S      Requires PyGMO to be compiled with snopt option. Minimization assumed
IPOPT                              :class:`PyGMO.algorithm.ipopt`                 C-C-S      Requires PyGMO to be compiled with ipopt option. Minimization assumed
================================== ========================================= =============== =====================================================================

Detailed Documentation
----------------------
.. autoclass:: PyGMO.algorithm._base()

   .. automethod:: PyGMO.algorithm._base.evolve

.. autoclass:: PyGMO.algorithm.base()

.. autoclass:: PyGMO.algorithm.de

   .. automethod:: PyGMO.algorithm.de.__init__

.. autoclass:: PyGMO.algorithm.pso

   .. automethod:: PyGMO.algorithm.pso.__init__

.. autoclass:: PyGMO.algorithm.pso_gen

   .. automethod:: PyGMO.algorithm.pso_gen.__init__

.. autoclass:: PyGMO.algorithm.sga

   .. automethod:: PyGMO.algorithm.sga.__init__

   .. attribute:: PyGMO.algorithm.sga.mutation.RANDOM

     Random mutation (width is set by the width argument in :class:`PyGMO.algorithm.sga`)

   .. attribute:: PyGMO.algorithm.sga.mutation.GAUSSIAN

     Gaussian mutation (bell shape standard deviation is set by the width argument in :class:`PyGMO.algorithm.sga` multiplied by the box-bounds width)

   .. attribute:: PyGMO.algorithm.sga.selection.ROULETTE

     Roulette selection mechanism

   .. attribute:: PyGMO.algorithm.sga.selection.BEST20

     Best 20% individuals are inserted over and over again

   .. attribute:: PyGMO.algorithm.sga.crossover.BINOMIAL

     Binomial crossover

   .. attribute:: PyGMO.algorithm.sga.crossover.EXPONENTIAL

     Exponential crossover

.. autoclass:: PyGMO.algorithm.sa_corana

   .. automethod:: PyGMO.algorithm.sa_corana.__init__

.. autoclass:: PyGMO.algorithm.bee_colony

   .. automethod:: PyGMO.algorithm.bee_colony.__init__

.. autoclass:: PyGMO.algorithm.ms

   .. automethod:: PyGMO.algorithm.ms.__init__

   .. attribute:: PyGMO.algorithm.ms.screen_output

      When True, the algorithms produces output on screen 

   .. attribute:: PyGMO.algorithm.ms.algorithm

      Algorithm to be multistarted

.. autoclass:: PyGMO.algorithm.mbh

   .. automethod:: PyGMO.algorithm.mbh.__init__

   .. attribute:: PyGMO.algorithm.mbh.screen_output

      When True, the algorithms produces output on screen 

   .. attribute:: PyGMO.algorithm.mbh.algorithm

      Algorithm to perform mbh 'local' search

.. autoclass:: PyGMO.algorithm.cs

   .. automethod:: PyGMO.algorithm.cs.__init__

.. autoclass:: PyGMO.algorithm.ihs

   .. automethod:: PyGMO.algorithm.ihs.__init__

.. autoclass:: PyGMO.algorithm.monte_carlo

   .. automethod:: PyGMO.algorithm.monte_carlo.__init__

.. autoclass:: PyGMO.algorithm.py_example

   .. automethod:: PyGMO.algorithm.py_example.__init__

.. autoclass:: PyGMO.algorithm.py_cross_entropy

   .. automethod:: PyGMO.algorithm.py_cross_entropy.__init__

.. autoclass:: PyGMO.algorithm.scipy_fmin

   .. automethod:: PyGMO.algorithm.scipy_fmin.__init__

.. autoclass:: PyGMO.algorithm.scipy_l_bfgs_b

   .. automethod:: PyGMO.algorithm.scipy_l_bfgs_b.__init__

.. autoclass:: PyGMO.algorithm.scipy_slsqp

   .. automethod:: PyGMO.algorithm.scipy_slsqp.__init__

.. autoclass:: PyGMO.algorithm.scipy_tnc

   .. automethod:: PyGMO.algorithm.scipy_tnc.__init__

   .. attribute:: PyGMO.algorithm.scipy_tnc.screen_output

      When True, the algorithms produces output on screen 

.. autoclass:: PyGMO.algorithm.scipy_cobyla

   .. automethod:: PyGMO.algorithm.scipy_cobyla.__init__

   .. attribute:: PyGMO.algorithm.scipy_cobyla.screen_output

      When True, the algorithms produces output on screen 

.. autoclass:: PyGMO.algorithm.nlopt_cobyla

   .. automethod:: PyGMO.algorithm.nlopt_cobyla.__init__

.. autoclass:: PyGMO.algorithm.nlopt_bobyqa

   .. automethod:: PyGMO.algorithm.nlopt_bobyqa.__init__

.. autoclass:: PyGMO.algorithm.nlopt_sbplx

   .. automethod:: PyGMO.algorithm.nlopt_sbplx.__init__

.. autoclass:: PyGMO.algorithm.gsl_nm2rand

   .. automethod:: PyGMO.algorithm.gsl_nm2rand.__init__

.. autoclass:: PyGMO.algorithm.gsl_nm2

   .. automethod:: PyGMO.algorithm.gsl_nm2.__init__

.. autoclass:: PyGMO.algorithm.gsl_nm

   .. automethod:: PyGMO.algorithm.gsl_nm.__init__

.. autoclass:: PyGMO.algorithm.gsl_pr

   .. automethod:: PyGMO.algorithm.gsl_pr.__init__

.. autoclass:: PyGMO.algorithm.gsl_fr

   .. automethod:: PyGMO.algorithm.gsl_fr.__init__

.. autoclass:: PyGMO.algorithm.gsl_bfgs2

   .. automethod:: PyGMO.algorithm.gsl_bfgs2.__init__

.. autoclass:: PyGMO.algorithm.gsl_bfgs

   .. automethod:: PyGMO.algorithm.gsl_bfgs.__init__

.. autoclass:: PyGMO.algorithm.snopt

   .. automethod:: PyGMO.algorithm.snopt.__init__

   .. attribute:: PyGMO.algorithm.snopt.screen_output

      When True, the algorithms produces output on screen 

.. autoclass:: PyGMO.algorithm.ipopt

   .. automethod:: PyGMO.algorithm.ipopt.__init

   .. attribute:: PyGMO.algorithm.ipopt.screen_output

      When True, the algorithms produces output on screen 
