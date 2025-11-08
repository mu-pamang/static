"""
Copyright (c) 2012 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Collection, Set

# Type variables
N = TypeVar('N')  # Nodes in the CFG, typically Unit or Block
M = TypeVar('M')  # Method representation


class InterproceduralCFG(ABC, Generic[N, M]):
    """
    An interprocedural control-flow graph.
    
    Type Parameters:
        N: Nodes in the CFG, typically Unit or Block
        M: Method representation
    """

    @abstractmethod
    def get_method_of(self, n: N) -> M:
        """
        Returns the method containing a node.
        
        Args:
            n: The node for which to get the parent method
            
        Returns:
            The method containing the node
        """
        pass

    @abstractmethod
    def get_preds_of(self, u: N) -> List[N]:
        """
        Returns the predecessor nodes.
        
        Args:
            u: The node for which to get predecessors
            
        Returns:
            List of predecessor nodes
        """
        pass

    @abstractmethod
    def get_succs_of(self, n: N) -> List[N]:
        """
        Returns the successor nodes.
        
        Args:
            n: The node for which to get successors
            
        Returns:
            List of successor nodes
        """
        pass

    @abstractmethod
    def get_callees_of_call_at(self, n: N) -> Collection[M]:
        """
        Returns all callee methods for a given call.
        
        Args:
            n: The call site node
            
        Returns:
            Collection of callee methods
        """
        pass

    @abstractmethod
    def get_callers_of(self, m: M) -> Collection[N]:
        """
        Returns all caller statements/nodes of a given method.
        
        Args:
            m: The method
            
        Returns:
            Collection of caller nodes
        """
        pass

    @abstractmethod
    def get_calls_from_within(self, m: M) -> Set[N]:
        """
        Returns all call sites within a given method.
        
        Args:
            m: The method
            
        Returns:
            Set of call site nodes within the method
        """
        pass

    @abstractmethod
    def get_start_points_of(self, m: M) -> Collection[N]:
        """
        Returns all start points of a given method. There may be
        more than one start point in case of a backward analysis.
        
        Args:
            m: The method
            
        Returns:
            Collection of start point nodes
        """
        pass

    @abstractmethod
    def get_return_sites_of_call_at(self, n: N) -> Collection[N]:
        """
        Returns all statements to which a call could return.
        In the RHS paper, for every call there is just one return site.
        We, however, use as return site the successor statements, of which
        there can be many in case of exceptional flow.
        
        Args:
            n: The call site node
            
        Returns:
            Collection of return site nodes
        """
        pass

    @abstractmethod
    def is_call_stmt(self, stmt: N) -> bool:
        """
        Returns True if the given statement is a call site.
        
        Args:
            stmt: The statement to check
            
        Returns:
            True if stmt is a call statement, False otherwise
        """
        pass

    @abstractmethod
    def is_exit_stmt(self, stmt: N) -> bool:
        """
        Returns True if the given statement leads to a method return
        (exceptional or not). For backward analyses may also be start statements.
        
        Args:
            stmt: The statement to check
            
        Returns:
            True if stmt is an exit statement, False otherwise
        """
        pass

    @abstractmethod
    def is_start_point(self, stmt: N) -> bool:
        """
        Returns True if this is a method's start statement. For backward analyses
        those may also be return or throws statements.
        
        Args:
            stmt: The statement to check
            
        Returns:
            True if stmt is a start point, False otherwise
        """
        pass

    @abstractmethod
    def all_non_call_start_nodes(self) -> Set[N]:
        """
        Returns the set of all nodes that are neither call nor start nodes.
        
        Returns:
            Set of all non-call, non-start nodes
        """
        pass

    @abstractmethod
    def is_fall_through_successor(self, stmt: N, succ: N) -> bool:
        """
        Returns whether succ is the fall-through successor of stmt,
        i.e., the unique successor that is reached when stmt
        does not branch.
        
        Args:
            stmt: The statement
            succ: The potential successor
            
        Returns:
            True if succ is the fall-through successor of stmt, False otherwise
        """
        pass

    @abstractmethod
    def is_branch_target(self, stmt: N, succ: N) -> bool:
        """
        Returns whether succ is a branch target of stmt.
        
        Args:
            stmt: The statement
            succ: The potential branch target
            
        Returns:
            True if succ is a branch target of stmt, False otherwise
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

import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * An interprocedural control-flow graph.
 * 
 * @param <N> Nodes in the CFG, typically {@link Unit} or {@link Block}
 * @param <M> Method representation
 */
public interface InterproceduralCFG<N,M>  {
	
	/**
	 * Returns the method containing a node.
	 * @param n The node for which to get the parent method
	 */
	public M getMethodOf(N n);

	public List<N> getPredsOf(N u);
	
	/**
	 * Returns the successor nodes.
	 */
	public List<N> getSuccsOf(N n);

	/**
	 * Returns all callee methods for a given call.
	 */
	public Collection<M> getCalleesOfCallAt(N n);

	/**
	 * Returns all caller statements/nodes of a given method.
	 */
	public Collection<N> getCallersOf(M m);

	/**
	 * Returns all call sites within a given method.
	 */
	public Set<N> getCallsFromWithin(M m);

	/**
	 * Returns all start points of a given method. There may be
	 * more than one start point in case of a backward analysis.
	 */
	public Collection<N> getStartPointsOf(M m);

	/**
	 * Returns all statements to which a call could return.
	 * In the RHS paper, for every call there is just one return site.
	 * We, however, use as return site the successor statements, of which
	 * there can be many in case of exceptional flow.
	 */
	public Collection<N> getReturnSitesOfCallAt(N n);

	/**
	 * Returns <code>true</code> if the given statement is a call site.
	 */
	public boolean isCallStmt(N stmt);

	/**
	 * Returns <code>true</code> if the given statement leads to a method return
	 * (exceptional or not). For backward analyses may also be start statements.
	 */
	public boolean isExitStmt(N stmt);
	
	/**
	 * Returns true is this is a method's start statement. For backward analyses
	 * those may also be return or throws statements.
	 */
	public boolean isStartPoint(N stmt);
	
	/**
	 * Returns the set of all nodes that are neither call nor start nodes.
	 */
	public Set<N> allNonCallStartNodes();
	
	/**
	 * Returns whether succ is the fall-through successor of stmt,
	 * i.e., the unique successor that is be reached when stmt
	 * does not branch.
	 */
	public boolean isFallThroughSuccessor(N stmt, N succ);
	
	/**
	 * Returns whether succ is a branch target of stmt. 
	 */
	public boolean isBranchTarget(N stmt, N succ);

}
\n"""