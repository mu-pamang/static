"""
Copyright (c) 2012 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
"""

from typing import TypeVar, Generic, Set
from .BiDiIDESolver import BiDiIDESolver, AbstractionWithSourceStmt
from .IFDSSolver import IFDSSolver, BinaryDomain
from .JoinHandlingNode import JoinHandlingNode
from Core.IFDSTabulationProblem import IFDSTabulationProblem
from .BiDiIDESolver import BiDiIDESolver, AbstractionWithSourceStmt


# Type variables
N = TypeVar('N')  # Node type
D = TypeVar('D', bound='JoinHandlingNode')  # Data-flow abstraction type (must implement JoinHandlingNode)
M = TypeVar('M')  # Method type
I = TypeVar('I')  # InterproceduralCFG type


class BiDiIFDSSolver(BiDiIDESolver[N, D, M, BinaryDomain, I], Generic[N, D, M, I]):
    """
    Special IFDS solver that solves the analysis problem inside out, i.e., from further 
    down the call stack to further up the call stack. This can be useful for taint 
    analysis problems that track flows in two directions.
    
    The solver is instantiated with two analyses, one computed forward and one backward.
    Both analysis problems must be unbalanced, i.e., must return True for 
    followReturnsPastSeeds().
    
    The solver executes both analyses in lockstep: when one analysis reaches an unbalanced 
    return edge (signified by a ZERO source value), the solver pauses this analysis until 
    the other analysis reaches the same unbalanced return (if ever). The result is that 
    the analyses will never diverge.
    
    This solver requires data-flow abstractions that implement the JoinHandlingNode 
    interface such that data-flow values can be linked to form reportable paths.
    
    Args:
        N: Node type (see IFDSSolver)
        D: Data-flow abstraction that must implement the JoinHandlingNode interface
        M: Method type (see IFDSSolver)
        I: InterproceduralCFG type (see IFDSSolver)
    """
    
    def __init__(self, forward_problem: 'IFDSTabulationProblem[N, D, M, I]',
                 backward_problem: 'IFDSTabulationProblem[N, D, M, I]'):
        """
        Instantiate a BiDiIFDSSolver with the associated forward and backward problem.
        
        Args:
            forward_problem: The forward IFDS tabulation problem
            backward_problem: The backward IFDS tabulation problem
        """
        # Convert IFDS problems to IDE problems using IFDSSolver helper
        forward_ide_problem = IFDSSolver.create_ide_tabulation_problem(forward_problem)
        backward_ide_problem = IFDSSolver.create_ide_tabulation_problem(backward_problem)
        
        # Call parent BiDiIDESolver constructor
        super().__init__(forward_ide_problem, backward_ide_problem)
    
    def fw_ifds_result_at(self, stmt: N) -> Set[D]:
        """
        Get the forward IFDS results at the given statement.
        
        Args:
            stmt: The statement node to query
            
        Returns:
            Set of data-flow abstractions that hold at the statement in forward analysis
        """
        annotated_results = self.fw_solver.results_at(stmt).keys()
        return self._extract_results(annotated_results)
    
    def bw_ifds_result_at(self, stmt: N) -> Set[D]:
        """
        Get the backward IFDS results at the given statement.
        
        Args:
            stmt: The statement node to query
            
        Returns:
            Set of data-flow abstractions that hold at the statement in backward analysis
        """
        annotated_results = self.bw_solver.results_at(stmt).keys()
        return self._extract_results(annotated_results)
    
    def _extract_results(self, annotated_results: Set[AbstractionWithSourceStmt]) -> Set[D]:
        """
        Extract plain abstractions from annotated results.
        
        Args:
            annotated_results: Set of AbstractionWithSourceStmt objects
            
        Returns:
            Set of plain data-flow abstractions (without source statement annotations)
        """
        result = set()
        for abstraction_with_source_stmt in annotated_results:
            result.add(abstraction_with_source_stmt.get_abstraction())
        return result

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
package heros.solver;

import heros.EdgeFunction;
import heros.FlowFunction;
import heros.FlowFunctions;
import heros.IFDSTabulationProblem;
import heros.InterproceduralCFG;
import heros.solver.IFDSSolver.BinaryDomain;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import com.google.common.collect.Maps;

/**
 * This is a special IFDS solver that solves the analysis problem inside out, i.e., from further down the call stack to
 * further up the call stack. This can be useful, for instance, for taint analysis problems that track flows in two directions.
 * 
 * The solver is instantiated with two analyses, one to be computed forward and one to be computed backward. Both analysis problems
 * must be unbalanced, i.e., must return <code>true</code> for {@link IFDSTabulationProblem#followReturnsPastSeeds()}.
 * The solver then executes both analyses in lockstep, i.e., when one of the analyses reaches an unbalanced return edge (signified
 * by a ZERO source value) then the solver pauses this analysis until the other analysis reaches the same unbalanced return (if ever).
 * The result is that the analyses will never diverge, i.e., will ultimately always only propagate into contexts in which both their
 * computed paths are realizable at the same time.
 * 
 * This solver requires data-flow abstractions that implement the {@link LinkedNode} interface such that data-flow values can be linked to form
 * reportable paths.  
 *
 * @param <N> see {@link IFDSSolver}
 * @param <D> A data-flow abstraction that must implement the {@link LinkedNode} interface such that data-flow values can be linked to form
 * 				reportable paths.
 * @param <M> see {@link IFDSSolver}
 * @param <I> see {@link IFDSSolver}
 */
public class BiDiIFDSSolver<N, D extends JoinHandlingNode<D>, M, I extends InterproceduralCFG<N, M>> extends BiDiIDESolver<N, D, M, BinaryDomain, I> {


	/**
	 * Instantiates a {@link BiDiIFDSSolver} with the associated forward and backward problem.
	 */
	public BiDiIFDSSolver(IFDSTabulationProblem<N,D,M,I> forwardProblem, IFDSTabulationProblem<N,D,M,I> backwardProblem) {
		super(IFDSSolver.createIDETabulationProblem(forwardProblem), IFDSSolver.createIDETabulationProblem(backwardProblem));
	}
	
	public Set<D> fwIFDSResultAt(N stmt) {
		return extractResults(fwSolver.resultsAt(stmt).keySet());
	}
	
	public Set<D> bwIFDSResultAt(N stmt) {
		return extractResults(bwSolver.resultsAt(stmt).keySet());
	}

	private Set<D> extractResults(Set<AbstractionWithSourceStmt> annotatedResults) {
		Set<D> res = new HashSet<D>();		
		for (AbstractionWithSourceStmt abstractionWithSourceStmt : annotatedResults) {
			res.add(abstractionWithSourceStmt.getAbstraction());
		}
		return res;
	}
	
}
\n"""