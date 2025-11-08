"""
Copyright (c) 2012 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
"""

from abc import abstractmethod
from typing import TypeVar, Generic
from .EdgeFunctions import EdgeFunctions
from .EdgeFunction import EdgeFunction
from .MeetLattice import MeetLattice

# Type variables
N = TypeVar('N')  # The type of nodes in the interprocedural control-flow graph
D = TypeVar('D')  # The type of data-flow facts to be computed by the tabulation problem
M = TypeVar('M')  # The type of objects used to represent methods
V = TypeVar('V')  # The type of values to be computed along flow edges
I = TypeVar('I')  # The type of inter-procedural control-flow graph being used


class IDETabulationProblem(Generic[N, D, M, V, I]):
    """
    Defines an IDE tabulation problem as presented in the Sagiv, Reps, Horwitz 1996 
    (SRH96) paper. An IDE tabulation problem extends an IFDSTabulationProblem
    by allowing additional values to be computed along flow functions: each domain value
    of type D maps at any program point to a value of type V. The functions describe how
    values are transformed when moving from one statement to another.
    
    The problem further defines a MeetLattice, which describes how values of
    type V are merged (via a meet operation) when multiple values are possible.
    
    Type Parameters:
        N: The type of nodes in the interprocedural control-flow graph. Typically Unit.
        D: The type of data-flow facts to be computed by the tabulation problem.
        M: The type of objects used to represent methods. Typically SootMethod.
        V: The type of values to be computed along flow edges.
        I: The type of inter-procedural control-flow graph being used.
    """

    @abstractmethod
    def edge_functions(self) -> 'EdgeFunctions[N, D, M, V]':
        """
        Returns the edge functions that describe how V-values are transformed along
        flow function edges.
        
        Returns:
            EdgeFunctions object that provides edge functions for different types of edges
        """
        pass

    @abstractmethod
    def meet_lattice(self) -> 'MeetLattice[V]':
        """
        Returns the lattice describing how to compute the meet of two V values.
        
        Returns:
            MeetLattice object that defines the meet operation for values
        """
        pass

    @abstractmethod
    def all_top_function(self) -> 'EdgeFunction[V]':
        """
        Returns a function mapping everything to top.
        
        Returns:
            An EdgeFunction that maps all values to the top element of the lattice
        """
        pass


"""\n===== ORIGINAL JAVA FOR REFERENCE =====\n/*******************************************************************************
 * Copyright (c) 2012 Eric Bodden.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Eric Bodden - initial API and implementation
 ******************************************************************************/
package heros;

/**
 * Defines an IDE tabulation problem as presented in the Sagiv, Reps, Horwitz 1996 
 * (SRH96) paper. An IDE tabulation problem extends an {@link IFDSTabulationProblem}
 * by allowing additional values to be computed along flow functions: each domain value
 * of type D maps at any program point to a value of type V. The functions describe how
 * values are transformed when moving from one statement to another.
 * 
 * The problem further defines a {@link MeetLattice}, which describes how values of
 * type V are merged (via a meet operation) when multiple values are possible.
 *
 * @param <N> The type of nodes in the interprocedural control-flow graph. Typically {@link Unit}.
 * @param <D> The type of data-flow facts to be computed by the tabulation problem.
 * @param <M> The type of objects used to represent methods. Typically {@link SootMethod}.
 * @param <V> The type of values to be computed along flow edges.
 * @param <I> The type of inter-procedural control-flow graph being used.
 */
public interface IDETabulationProblem<N,D,M,V,I extends InterproceduralCFG<N,M>> extends IFDSTabulationProblem<N,D,M,I>{

	/**
	 * Returns the edge functions that describe how V-values are transformed along
	 * flow function edges.
	 */
	EdgeFunctions<N,D,M,V> edgeFunctions();
	
	/**
	 * Returns the lattice describing how to compute the meet of two V values.
	 */
	MeetLattice<V> meetLattice();

	/**
	 * Returns a function mapping everything to top.
	 */	
	EdgeFunction<V> allTopFunction(); 
}
\n"""