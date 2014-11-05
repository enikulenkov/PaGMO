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

#ifndef PAGMO_ALGORITHM_birmingham_ga_H
#define PAGMO_ALGORITHM_birmingham_ga_H

#include "../config.h"
#include "../problem/base.h"
#include "../serialization.h"
#include "base.h"

namespace pagmo { namespace algorithm {

/// The Birmingham Genetic Algorithm
/**
 * Genetic algorithms are very popular algorithms used widely by people of very different backgrounds.
 * As a consequence there are a large number of different implementations and toolboxes that are available
 * and can be used to construct a genetic algorithm. We decided not to choose one of these and, instead, to
 * provide only a basic implementation of the algorithm implementing a floating point encoding (not binary)
 * and some common mutation and crossover strategies, hence the name Simple Genetic Algorithm.
 *
 * Mutation is gaussian or random, crossover exponential or binomial and selection is tournament or
 * roulette wheel.
 *
 * The algorithm works on single objective, box constrained problems. The mutation operator acts
 * differently on continuous and discrete variables.
 *
 * @author Dario Izzo (dario.izzo@googlemail.com)
 *
 */

class __PAGMO_VISIBLE birmingham_ga: public base
{
public:
  /// Selection info
  struct selection {
    /// Selection type, best 20% or roulette
    enum type {ROULETTE = 1, TOURNAMENT = 2};
  };
  /// Mutation operator info
  struct mutation {
      /// Mutation type
      enum type {MOVE, ROTATE, REPLACE, MUTATIONS_CNT};
      /// Mutation type
      type type;
      /// Mutation width
      double probability;
  };

  /// Crossover operator info
  struct crossover {
    /// Crossover type, binomial or "cut and splice"
    enum type {BINOMIAL = 0, CUT_AND_SPLICE = 2};
  };
  birmingham_ga(const int gen,
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
      const double &bfgs_tol);
  base_ptr clone() const;
  void evolve(population &) const;
  std::string get_name() const;
  void randomize_cluster(decision_vector &x) const;
protected:
  std::string human_readable_extra() const;
private:
  void do_cut_and_splice(decision_vector &vec1, decision_vector &vec2) const;
  bool check_cluster(decision_vector &x) const;
  void make_rotation(decision_vector &vec) const;
  void possibly_mutate(decision_vector &x) const;
  void mutation_rotate(decision_vector &x) const;
  void mutation_move(decision_vector &x) const;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int)
  {
    ar & boost::serialization::base_object<base>(*this);
    ar & const_cast<int &>(m_gen);
    ar & const_cast<double &>(m_crossover_rate);
    ar & const_cast<int &>(m_elitism);
    ar & const_cast<selection::type &>(m_selection_type);
    ar & const_cast<crossover::type &>(m_crossover_type);
  }  
  //Number of generations
  int m_gen;
  //Crossover rate
  double m_crossover_rate;
  double m_binom_rate;

  //Elitism (number of generations after which to reinsert the best)
  int m_elitism;
  double m_min_atom_dist;
  selection::type m_selection_type;
  crossover::type m_crossover_type;
  double m_max_coord;
  double m_bfgs_step_size;
  double m_bfgs_tol;
  //Possible mutations
  mutation m_mutations[mutation::MUTATIONS_CNT];
  int m_mut_count;
};

}} //namespaces

BOOST_CLASS_EXPORT_KEY(pagmo::algorithm::birmingham_ga);

#endif // PAGMO_ALGORITHM_birmingham_ga_H
