/*****************************************************************************
 *   Copyright (C) 2004-2013 The PaGMO development team,                     *
 *   Advanced Concepts Team (ACT), European Space Agency (ESA)               *
 *   http://apps.sourceforge.net/mediawiki/pagmo                             *
 *   http://apps.sourceforge.net/mediawiki/pagmo/index.php?title=Developers  *
 *   http://apps.sourceforge.net/mediawiki/pagmo/index.php?title=Credits     *
 *   act@esa.int                                                             *
 *                                                                           *
 *   This program is free software; you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation; either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program; if not, write to the                           *
 *   Free Software Foundation, Inc.,                                         *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.               *
 *****************************************************************************/

#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/round.hpp>

#include <gsl/gsl_deriv.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>

#include <string>
#include <vector>
#include <algorithm>

#include "../exceptions.h"
#include "../population.h"
#include "../types.h"
#include "base.h"
#include "birmingham_ga.h"
#include "../problem/base_stochastic.h"

#define Z(i) ((i)*3 +2)
#define COORDS_CNT 3
#define  SQR(_a) ((_a)*(_a))

namespace pagmo { namespace algorithm {

birmingham_ga::birmingham_ga(const int gen,
      const double &crossover_rate,
      const double &binom_rate,
      const double &min_atom_dist,
	    mutation *muts,
      int mut_count,
      int elitism,
	    selection::type sel,
	    crossover::type cro,
      const double &max_coord,
      const double &bfgs_step_size,
      const double &bfgs_tol)
{
  m_gen = gen;
  m_crossover_rate = crossover_rate;
  m_binom_rate = binom_rate;
  m_min_atom_dist = min_atom_dist;
  m_elitism = elitism;
  m_selection_type = sel;
  m_crossover_type = cro;
  m_max_coord = max_coord;
  m_bfgs_step_size = bfgs_step_size;
  m_bfgs_tol = bfgs_tol;
  m_mut_count = mut_count;

  for (int i=0; i < mut_count; i++)
  {
    m_mutations[i] = muts[i];
  }
}

/// Clone method.
base_ptr birmingham_ga::clone() const
{
  return base_ptr(new birmingham_ga(*this));
}


bool birmingham_ga::check_cluster(decision_vector &x) const
{
  //for (population::size_type i=0; i < x.size(); i++)
  //{
    //if (abs(x[i]) > 2*m_max_coord)
    //{
      //return false;
    //}
  //}

  for (population::size_type i=0; i < x.size()/COORDS_CNT - 1; i++)
  {
    for (population::size_type j=i+1; j < x.size()/COORDS_CNT; j++)
    {
      double dist = SQR(x[i*COORDS_CNT] - x[j*COORDS_CNT]) +
                    SQR(x[i*COORDS_CNT+1] - x[j*COORDS_CNT+1]) +
                    SQR(x[i*COORDS_CNT+2] - x[j*COORDS_CNT+2]);

      if (dist < m_min_atom_dist*m_min_atom_dist)
      {
        return false;
      }
    }
  }

  return true;
}


void birmingham_ga::randomize_cluster(decision_vector &x)
{
  boost::uniform_real<double> rand_coord(-m_max_coord, m_max_coord);
  double step = 0.5;
  int step_count = m_max_coord * 2 / step;
  boost::random::mt19937 rng;
  boost::random::uniform_int_distribution<> rand_dist(0,step_count);
  boost::uniform_real<double> rand_angle(0, 2*3.1415);

  for (population::size_type i = 0; i < x.size(); i++)
  {
    x[i] = m_max_coord * (-1) + rand_dist(rng) * step;
  }
}


int compare_function(const void *a,const void *b)
{
  double *atom1 = (double *) a;
  double *atom2 = (double *) b;
  if (atom1[2] < atom2[2]) return -1;
  else if (atom1[2] > atom2[2]) return 1;
  return 0;
}


void matr_vector_mult(double *matr, double *vec)
{
  const int n = 3;
  double res[3];

  memset(res, 0, sizeof(res));

  for (int i = 0; i < n; i++)
  {
    for (int j=0; j < n; j++)
    {
      res[i] += matr[i*n + j]*vec[j];
    }
  }

  for (int i=0; i < n; i++)
  {
    vec[i] = res[i];
  }
}

void birmingham_ga::make_rotation(decision_vector &vec) const
{
  boost::uniform_real<double> rand_angle(0, 2*3.1415);
  double phi;
  double cos_phi;
  double sin_phi;
  const int n = 3;

  decision_vector vec_copy = vec;

  double rot_matr[3*3];

  /* Rotation along X axis */
  /* 1    0        0
   * 0 cos_phi -sin_phi
   * 0 sin_phi cos_phi */
  {
    phi = rand_angle(m_drng);
    cos_phi = cos(phi);
    sin_phi = sin(phi);

    memset(rot_matr, 0, sizeof(rot_matr));
    rot_matr[0] = 1;
    rot_matr[n+1] = cos_phi;
    rot_matr[n+2] = -sin_phi;
    rot_matr[2*n+1] = sin_phi;
    rot_matr[2*n+2] = cos_phi;

    for (population::size_type i=0; i < vec.size(); i+=3)
    {
      matr_vector_mult(rot_matr, &vec[i]);
    }
  }

  /* Rotation along Y axis */
  /* cos_phi  0  sin_phi 
   * 0        1  0 
   * -sin_phi  0  cos_phi */
  {
    phi = rand_angle(m_drng);
    cos_phi = cos(phi);
    sin_phi = sin(phi);

    memset(rot_matr, 0, sizeof(rot_matr));
    rot_matr[0] = cos_phi;
    rot_matr[2] = sin_phi;
    rot_matr[n+1] = 1;
    rot_matr[2*n] = -sin_phi;
    rot_matr[2*n+2] = cos_phi;

    for (population::size_type i=0; i < vec.size(); i+=3)
    {
      matr_vector_mult(rot_matr, &vec[i]);
    }
  }
}


/*
static inline int splice_atoms_cnt(decision_vector &vec, double z_coord, bool dir_up)
{
  int res = 0;

  for (population::size_type i = 0; i < vec.size()/3; i++)
  {
    res += (dir_up && (vec[i*3 + 2] >= z_coord)) || (!dir_up && (vec[i*3 +2] < z_coord));
  }

  return res;
}
*/

typedef struct objfun_params_s
{
  problem::base const *p;
  decision_vector x;
  fitness_vector f;
  double step_size;
}
objfun_params_t;

/// Objective function wrapper.
/**
 * @param[in] v pointer to the gsl_vector representing the decision vector.
 * @param[in] params pointer to extra parameters for the internal function.
 *
 * @return the fitness of the input decision vector.
 */
double objfun_wrapper(const gsl_vector *v, void *params)
{
  objfun_params_t *par = (objfun_params_t *)params;
  // Size of the continuous part of the problem.
  const problem::base::size_type cont_size = par->p->get_dimension() - par->p->get_i_dimension();
  // Fill up the continuous part of temporary storage with the contents of v.
  for (problem::base::size_type i = 0; i < cont_size; ++i) {
    par->x[i] = gsl_vector_get(v,i);
  }
  // Compute the objective function.
  par->p->objfun(par->f,par->x);
  return par->f[0];
}


void d_objfun(const gsl_vector *v, void *params, gsl_vector *df)
{
  objfun_params_t *par = (objfun_params_t *)params;

  const problem::base::size_type cont_size = par->p->get_dimension() - par->p->get_i_dimension();
  double *ret_df = (double *)malloc(cont_size * sizeof(double));

  for (problem::base::size_type i = 0; i < cont_size; ++i)
  {
    par->x[i] = gsl_vector_get(v,i);
  }

  par->p->d_objfun(&par->x[0], cont_size, ret_df);

  for (problem::base::size_type i = 0; i < cont_size; ++i)
  {
    gsl_vector_set(df, i, ret_df[i]);
  }
  free(ret_df);
}


// Simmultaneous function/derivative computation wrapper for the objective function.
void fd_objfun_wrapper(const gsl_vector *v, void *params, double *retval, gsl_vector *df)
{
  *retval = objfun_wrapper(v,params);
  d_objfun(v,params,df);
}


void minimize_cluster(decision_vector &vec1, const problem::base &prob)
{
  const gsl_multimin_fdfminimizer_type *minimizer_type;
  gsl_multimin_fdfminimizer *minimizer;
  gsl_vector *x = NULL; /* Starting point */
  double step_size = 0.01;
  double tol = 0.1;
  objfun_params_t params;
  const problem::base::size_type cont_size = prob.get_dimension() - prob.get_i_dimension();
  gsl_multimin_function_fdf gsl_func;
  double numdiff_step_size = 1e-8;
  int max_iter = 10;

  params.p = &prob;
  params.x.resize(vec1.size());
  params.f.resize(1);
  params.step_size = numdiff_step_size;

  gsl_func.n = boost::numeric_cast<std::size_t>(cont_size);
  gsl_func.f = &objfun_wrapper;
  gsl_func.df = &d_objfun;
  gsl_func.fdf = &fd_objfun_wrapper;
  gsl_func.params = (void *)&params;

  /* Alloc memory and initialize starting point */
  minimizer_type = gsl_multimin_fdfminimizer_vector_bfgs2;
  minimizer = gsl_multimin_fdfminimizer_alloc(minimizer_type, cont_size);

  x = gsl_vector_alloc(cont_size);

  for (population::size_type i = 0; i < vec1.size(); i++)
  {
    gsl_vector_set(x, i, vec1[i]);
  }

  /* Init the solver */
  gsl_multimin_fdfminimizer_set(minimizer, &gsl_func, x, step_size, tol);

  /* Iterate */
  int iter = 0;
  int status;
  try {
    do
    {
      ++iter;
      status = gsl_multimin_fdfminimizer_iterate(minimizer);
      if (status) {
        break;
      }
      status = gsl_multimin_test_gradient(minimizer->gradient, tol);
    } while (status == GSL_CONTINUE && iter < max_iter);
  } catch (const std::exception &e) {
    // Cleanup and re-throw.
    gsl_vector_free(x);
    gsl_multimin_fdfminimizer_free(minimizer);
    throw e;
  } catch (...) {
    // Cleanup and throw.
    gsl_vector_free(x);
    gsl_multimin_fdfminimizer_free(minimizer);
    pagmo_throw(std::runtime_error,"unknown exception caught in minimize_cluster");
  }
  // Free up resources.
  gsl_vector_free(x);
  gsl_multimin_fdfminimizer_free(minimizer);

  // Check the generated individual and change it to respect the bounds as necessary.
  for (problem::base::size_type i = 0; i < cont_size; ++i) {
    if (params.x[i] < prob.get_lb()[i]) {
      params.x[i] = prob.get_lb()[i];
    }
    if (params.x[i] > prob.get_ub()[i]) {
      params.x[i] = prob.get_ub()[i];
    }
  }

  vec1 = params.x;
}


void make_splice(decision_vector &vec1, decision_vector &vec2)
{
  int atoms_cnt = vec1.size()/3;
  int cur_idx1 = atoms_cnt/2;
  int cur_idx2 = atoms_cnt/2;
  int init_idx2;
  bool splice_found = false;

  /* Sort both decision vectors by Z coordinate */
  qsort(&vec1[0], atoms_cnt, 3*sizeof(double), compare_function);
  qsort(&vec2[0], atoms_cnt, 3*sizeof(double), compare_function);

  /* Find corresponding index to cur_idx1 at vec2 */
  /* TODO: Possible array bound error? */
  while (vec2[Z(cur_idx2+1)] < vec1[Z(cur_idx1)])
  {
    cur_idx2++;
  }

  while (vec2[Z(cur_idx2)] >= vec1[Z(cur_idx1)])
  {
    cur_idx2--;
  }

  //pagmo_assert(cur_idx2 >= 0);
  //
  if (cur_idx2 < 0)
  {
    return;
  }

  init_idx2 = cur_idx2;

  /* Search in forward direction from the middle */
  for ( ; cur_idx1 < atoms_cnt; cur_idx1++)
  {
    while ((vec2[Z(cur_idx2+1)] < vec1[Z(cur_idx1)]) && (cur_idx2 < atoms_cnt - 1))
    {
      cur_idx2++;
    }

    if (cur_idx2 > atoms_cnt - 1)
    {
      break;
    }

    if (cur_idx1 - 1 == cur_idx2)
    {
      splice_found = true;
      memcpy(&vec1[0], &vec2[0], cur_idx1*3*sizeof(double));
      break;
    }
  }

  /* Search in backwards direction */
  if (!splice_found)
  {
    for (cur_idx1 = atoms_cnt/2, cur_idx2 = init_idx2; cur_idx1 > 0; cur_idx1--)
    {
      while ((vec2[Z(cur_idx2)] >= vec1[Z(cur_idx1)]) && cur_idx2 > 0)
      {
        cur_idx2--;
      }

      if (cur_idx2 < 0)
      {
        break;
      }

      if (cur_idx1 - 1 == cur_idx2)
      {
        splice_found = true;
        memcpy(&vec1[0], &vec2[0], cur_idx1*3*sizeof(double));
        break;
      }
    }
  }
}


void birmingham_ga::do_cut_and_splice(decision_vector &vec1, decision_vector &vec2) const
{
  pagmo_assert(vec1.size() == vec2.size());

  make_rotation(vec1);
  make_rotation(vec2);

  /* Try to make splice operation */
  make_splice(vec1, vec2);
}


/// Evolve implementation.
/**
 * Run the simple genetic algorithm for the number of generations specified in the constructors.
 * At each improvment velocity is also updated.
 *
 * @param[in,out] pop input/output pagmo::population to be evolved.
 */

void birmingham_ga::evolve(population &pop) const
{
  // Let's store some useful variables.
  const problem::base &prob = pop.problem();
  const problem::base::size_type D = prob.get_dimension(), prob_c_dimension = prob.get_c_dimension(), prob_f_dimension = prob.get_f_dimension();
  const decision_vector &lb = prob.get_lb(), &ub = prob.get_ub();
  const population::size_type NP = pop.size();


  //We perform some checks to determine wether the problem/population are suitable for birmingham_ga
  if ( prob_c_dimension != 0 ) {
    pagmo_throw(value_error,"The problem is not box constrained and birmingham_ga is not suitable to solve it");
  }

  if ( prob_f_dimension != 1 ) {
    pagmo_throw(value_error,"The problem is not single objective and birmingham_ga is not suitable to solve it");
  }

  if (NP < 5) {
    pagmo_throw(value_error,"for birmingham_ga at least 5 individuals in the population are needed");
  }

  // Get out if there is nothing to do.
  if (m_gen == 0) {
    return;
  }
  // Some vectors used during evolution are allocated here.
  decision_vector dummy(D,0);      //used for initialisation purposes
  std::vector<decision_vector > X(NP,dummy), Xnew(NP,dummy);

  std::vector<fitness_vector > fit(NP);    //fitness

  fitness_vector bestfit;
  decision_vector bestX(D,0);

  std::vector<double> selectionfitness(NP), cumsum(NP), cumsumTemp(NP);
  std::vector <int> selection(2*NP); /* Parent pairs for selection */

  std::vector<int> fitnessID(NP);

  // Initialise the chromosomes and their fitness to that of the initial deme
  for (pagmo::population::size_type i = 0; i<NP; i++ ) {
    X[i]  =  pop.get_individual(i).cur_x;
    fit[i]  =  pop.get_individual(i).cur_f;
  }

  // Find the best member and store in bestX and bestfit
  double bestidx = pop.get_best_idx();
  bestX = pop.get_individual(bestidx).cur_x;
  bestfit = pop.get_individual(bestidx).cur_f;


  // Main birmingham_ga loop
  for (int j = 0; j<m_gen; j++) {

    switch (m_selection_type)
    {
      case selection::ROULETTE: {
        //We scale all fitness values from 0 (worst) to absolute value of the best fitness
        fitness_vector worstfit=fit[0];
        for (pagmo::population::size_type i = 1; i < NP;i++) {
          if (prob.compare_fitness(worstfit,fit[i]))
          {
            worstfit=fit[i];
          }
        }

        for (pagmo::population::size_type i = 0; i < NP; i++) {
          selectionfitness[i] = fabs(worstfit[0] - fit[i][0]);
        }

        // We build and normalise the cumulative sum
        cumsumTemp[0] = selectionfitness[0];
        for (pagmo::population::size_type i = 1; i< NP; i++) {
          cumsumTemp[i] = cumsumTemp[i - 1] + selectionfitness[i];
        }
        for (pagmo::population::size_type i = 0; i < NP; i++) {
          cumsum[i] = cumsumTemp[i]/cumsumTemp[NP-1];
        }

        //we throw a dice and pick up the corresponding index
        double r2;
        for (pagmo::population::size_type i = 0; i < selection.size(); i++) {
          r2 = m_drng();
          for (pagmo::population::size_type j = 0; j < NP; j++) {
            if (cumsum[j] > r2) {
              selection[i]=j;
              break;
            }
          }
        }
      }
      break;

      case selection::TOURNAMENT:
      {
        population::size_type selected_number = 0;

        while (selected_number < selection.size())
        {
          int tournament_pool[NP];
          int tournament_pool_cnt = 0;
          double rand_double;

          memset(tournament_pool, 0, sizeof(tournament_pool));

          /* Construct tournament pool */
          for (population::size_type i = 0; i < NP; i++)
          {
            rand_double = m_drng();

            if (rand_double > 0.8)
            {
              tournament_pool[tournament_pool_cnt++] = i;
            }
          }

          if (tournament_pool_cnt < 2)
          {
            /* We are unlucky to construct tournament pool */
            continue;
          }

          /* Find two best parents from tournament pool */
          {
            fitness_vector threshold_fit;
            int parent1 = tournament_pool[0];
            int parent2 = tournament_pool[1];

            if (prob.compare_fitness(fit[parent2], fit[parent1]))
            {
              int tmp = parent1;
              parent1 = parent2;
              parent2 = tmp;
            }

            threshold_fit = fit[parent2];

            for (int i = 1; i < tournament_pool_cnt; i++)
            {
              if (prob.compare_fitness(fit[tournament_pool[i]],threshold_fit))
              {
                if (prob.compare_fitness(fit[tournament_pool[i]],fit[parent1]))
                {
                  parent2 = parent1;
                  parent1 = tournament_pool[i];
                }
                else
                {
                  parent2 = tournament_pool[i];
                }

                threshold_fit = fit[parent2];
              }
            }

            selection[selected_number++] = parent1;
            selection[selected_number++] = parent2;
          }
        }
      }
      break;
    }

    //Xnew stores the new selected generation of chromosomes
    for (pagmo::population::size_type i = 0; i < NP; i++) {
      Xnew[i]=X[selection[i]];
    }

    //2 - Crossover
    {
      decision_vector  member1,member2;
      pagmo::population::size_type i = 0;

      while (i < selection.size())
      {
        member1 = X[selection[i]];
        member2 = X[selection[i+1]];
        //and we operate crossover
        switch (m_crossover_type)
        {
            //0 - binomial crossover
          case crossover::BINOMIAL:
          {
            size_t n = boost::uniform_int<int>(0,D-1)(m_urng);
            for (size_t L = 0; L < D; ++L) { /* perform D binomial trials */
              if ((m_drng() < m_crossover_rate) || L + 1 == D) { /* change at least one parameter */
                member1[n] = member2[n];
              }
              n = (n+1)%D;
            }
          }
          break;

          case crossover::CUT_AND_SPLICE:
          {
            decision_vector tmp1;
            decision_vector tmp2;
            do
            {
              tmp1 = member1;
              tmp2 = member2;
              this->do_cut_and_splice(tmp1, tmp2);

              if (!check_cluster(tmp1))
              {
                continue;
              }

              minimize_cluster(tmp1, prob);
            }
            while (!check_cluster(tmp1));

            member1 = tmp1;
          }
          break;
        }
        Xnew[i/2] = member1;
        i += 2;
      }
    }

    //3 - Mutation
    //switch (m_mut.m_type) {
      //case mutation::GAUSSIAN:
      //{
        //boost::normal_distribution<double> dist;
        //boost::variate_generator<boost::lagged_fibonacci607 &, boost::normal_distribution<double> > delta(m_drng,dist);
        //for (pagmo::problem::base::size_type k = 0; k < Dc;k++) { //for each continuous variable
          //double std = (ub[k]-lb[k]) * m_mut.m_width;
          //for (pagmo::population::size_type i = 0; i < NP;i++) { //for each individual
            //if (m_drng() < m_m) {
              //double mean = Xnew[i][k];
              //double tmp = (delta() * std + mean);
              //if ( (tmp < ub[k]) &&  (tmp > lb[k]) ) Xnew[i][k] = tmp;
            //}
          //}
        //}
        //for (pagmo::problem::base::size_type k = Dc; k < D;k++) { //for each integer variable
          //double std = (ub[k]-lb[k]) * m_mut.m_width;
          //for (pagmo::population::size_type i = 0; i < NP;i++) { //for each individual
            //if (m_drng() < m_m) {
              //double mean = Xnew[i][k];
              //double tmp = boost::math::iround(delta() * std + mean);
              //if ( (tmp < ub[k]) &&  (tmp > lb[k]) ) Xnew[i][k] = tmp;
            //}
          //}
        //}
      //}
      //break;

      //case mutation::RANDOM:
      //{
        //for (pagmo::population::size_type i = 0; i < NP;i++) {
          //for (pagmo::problem::base::size_type j = 0; j < Dc;j++) { //for each continuous variable
            //if (m_drng() < m_m) {
              //Xnew[i][j] = boost::uniform_real<double>(lb[j],ub[j])(m_drng);
            //}
          //}
          //for (pagmo::problem::base::size_type j = Dc; j < D;j++) {//for each integer variable
            //if (m_drng() < m_m) {
              //Xnew[i][j] = boost::uniform_int<int>(lb[j],ub[j])(m_urng);
            //}
          //}
        //}
      //}
      //break;

      //case mutation::ATOMIC:
      //{
        //[> no mutation <]
      //}
      //break;
    //}

    //4 - Evaluate the new population (deterministic problem)
    for (pagmo::population::size_type i = 0; i < NP;i++) {
      prob.objfun(fit[i],Xnew[i]);
      dummy = Xnew[i];
      std::transform(dummy.begin(), dummy.end(), pop.get_individual(i).cur_x.begin(), dummy.begin(),std::minus<double>());
      //updates x and v (cache avoids to recompute the objective function and constraints)
      pop.set_x(i,Xnew[i]);
      pop.set_v(i,dummy);
      if (prob.compare_fitness(fit[i], bestfit)) {
        bestfit = fit[i];
        bestX = Xnew[i];
      }
    }
    
    //5 - Reinsert best individual every m_elitism generations
    if (j % m_elitism == 0) {
      int worst=0;
      for (pagmo::population::size_type i = 1; i < NP;i++) {
        if ( prob.compare_fitness(fit[worst],fit[i]) ) worst=i;
      }
      Xnew[worst] = bestX;
      fit[worst] = bestfit;
      dummy = Xnew[worst];
      std::transform(dummy.begin(), dummy.end(), pop.get_individual(worst).cur_x.begin(), dummy.begin(),std::minus<double>());
      //updates x and v (cache avoids to recompute the objective function)
      pop.set_x(worst,Xnew[worst]);
      pop.set_v(worst,dummy);
    }
    X = Xnew;
  } // end of main birmingham_ga loop
}

/// Algorithm name
std::string birmingham_ga::get_name() const
{
  return "Birmingham Genetic Algorithm for molecular clusters";
}

/// Extra human readable algorithm info.
/**
 * Will return a formatted string displaying the parameters of the algorithm.
 */
std::string birmingham_ga::human_readable_extra() const
{
  std::ostringstream s;
  s << "gen:" << m_gen << ' ';
  s << "CR:" << m_crossover_rate << ' ';
  s << "elitism:" << m_elitism << ' ';

  s << "selection:";
  switch (m_selection_type) {
    case selection::ROULETTE: {
          s << "ROULETTE "; 
          break;
          }
    case selection::TOURNAMENT: {
          s << "TOURNAMENT "; 
          break;
          }
  }
  s << "crossover:";
  switch (m_crossover_type) {
    case crossover::BINOMIAL: {
          s << "BINOMIAL "; 
          break;
          }
    case crossover::CUT_AND_SPLICE: {
          s << "CUT_AND_SPLICE "; 
          break;
          }
  }

  return s.str();
}

}} //namespaces

//BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::algorithm::birmingham_ga);
