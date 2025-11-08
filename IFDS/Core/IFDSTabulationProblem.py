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
from typing import TypeVar, Generic, Dict, Set
from .FlowFunctions import FlowFunctions

# Type variables
N = TypeVar('N')  # The type of nodes in the interprocedural control-flow graph
D = TypeVar('D')  # The type of data-flow facts to be computed by the tabulation problem
M = TypeVar('M')  # The type of objects used to represent methods
I = TypeVar('I')  # The type of inter-procedural control-flow graph being used


class IFDSTabulationProblem(Generic[N, D, M, I]):
    """
    A tabulation problem for solving in an IFDSSolver as described
    by the Reps, Horwitz, Sagiv 1995 (RHS95) paper.
    
    Type Parameters:
        N: The type of nodes in the interprocedural control-flow graph. Typically Unit.
        D: The type of data-flow facts to be computed by the tabulation problem.
        M: The type of objects used to represent methods. Typically SootMethod.
        I: The type of inter-procedural control-flow graph being used.
    """

    @abstractmethod
    def flow_functions(self) -> 'FlowFunctions[N, D, M]':
        """
        Returns a set of flow functions. Those functions are used to compute data-flow facts
        along the various kinds of control flows.
        
        NOTE: this method could be called many times. Implementations of this
        interface should therefore cache the return value!
        
        Returns:
            FlowFunctions object that provides flow functions for different types of edges
        """
        pass

    @abstractmethod
    def interprocedural_cfg(self) -> I:
        """
        Returns the interprocedural control-flow graph which this problem is computed over.
        
        NOTE: this method could be called many times. Implementations of this
        interface should therefore cache the return value!
        
        Returns:
            The interprocedural control-flow graph
        """
        pass

    @abstractmethod
    def initial_seeds(self) -> Dict[N, Set[D]]:
        """
        Returns initial seeds to be used for the analysis. This is a mapping of statements 
        to initial analysis facts.
        
        Returns:
            A dictionary mapping nodes to sets of initial data-flow facts
        """
        pass

    @abstractmethod
    def zero_value(self) -> D:
        """
        This must be a data-flow fact of type D, but must NOT be part of the domain of 
        data-flow facts. Typically this will be a singleton object of type D that is used 
        for nothing else. It must hold that this object does not equal any object within 
        the domain.
        
        NOTE: this method could be called many times. Implementations of this
        interface should therefore cache the return value!
        
        Returns:
            The special zero value used in the analysis
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

import heros.solver.IFDSSolver;

import java.util.Map;
import java.util.Set;



/**
 * A tabulation problem for solving in an {@link IFDSSolver} as described
 * by the Reps, Horwitz, Sagiv 1995 (RHS95) paper.
 *
 * @param <N> The type of nodes in the interprocedural control-flow graph. Typically {@link Unit}.
 * @param <D> The type of data-flow facts to be computed by the tabulation problem.
 * @param <M> The type of objects used to represent methods. Typically {@link SootMethod}.
 * @param <I> The type of inter-procedural control-flow graph being used.
 */
public interface IFDSTabulationProblem<N,D,M, I extends InterproceduralCFG<N,M>> extends SolverConfiguration {

	/**
	 * Returns a set of flow functions. Those functions are used to compute data-flow facts
	 * along the various kinds of control flows.
     *
	 * <b>NOTE:</b> this method could be called many times. Implementations of this
	 * interface should therefore cache the return value! 
	 */
	FlowFunctions<N,D,M> flowFunctions();
	
	/**
	 * Returns the interprocedural control-flow graph which this problem is computed over.
	 * 
	 * <b>NOTE:</b> this method could be called many times. Implementations of this
	 * interface should therefore cache the return value! 
	 */
	I interproceduralCFG();
	
	/**
	 * Returns initial seeds to be used for the analysis. This is a mapping of statements to initial analysis facts.
	 */
	Map<N,Set<D>> initialSeeds();
	
	/**
	 * This must be a data-flow fact of type {@link D}, but must <i>not</i>
	 * be part of the domain of data-flow facts. Typically this will be a
	 * singleton object of type {@link D} that is used for nothing else.
	 * It must holds that this object does not equals any object 
	 * within the domain.
	 *
	 * <b>NOTE:</b> this method could be called many times. Implementations of this
	 * interface should therefore cache the return value! 
	 */
	D zeroValue();

}
\n"""