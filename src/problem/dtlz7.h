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

#ifndef PAGMO_PROBLEM_DTLZ7_H
#define PAGMO_PROBLEM_DTLZ7_H

#include <string>

#include "../serialization.h"
#include "../types.h"
#include "base_dtlz.h"

namespace pagmo{ namespace problem {

/// DTLZ7 problem
/**
 *
 * This is a box-constrained continuous n-dimensional multi-objecive problem, scalable in fitness dimension.
 *
 * This problem has disconnected Pareto-optimal regions in the search space.
 *
 * The dimension of the decision space is k + fdim - 1, whereas fdim is the number of objectives and k a paramter.
 *
 * @see K. Deb, L. Thiele, M. Laumanns, E. Zitzler, Scalable test problems for evoulationary multiobjective optimization
 * @author Marcus Maertens (mmarcusx@gmail.com)
 */

class __PAGMO_VISIBLE dtlz7 : public base_dtlz
{
	public:
		dtlz7(int = 20, fitness_vector::size_type = 3);
		base_ptr clone() const;
		std::string get_name() const;
	protected:
		void objfun_impl(fitness_vector &, const decision_vector &) const;
	private:
		double g_func(const decision_vector &) const;
		double h_func(const fitness_vector &, const double) const;
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive &ar, const unsigned int)
		{
			ar & boost::serialization::base_object<base>(*this);
		}
};

}} //namespaces

BOOST_CLASS_EXPORT_KEY(pagmo::problem::dtlz7);

#endif // PAGMO_PROBLEM_DTLZ7_H