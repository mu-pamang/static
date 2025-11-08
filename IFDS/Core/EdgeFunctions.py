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
from typing import TypeVar, Generic, Optional

# Type variables
N = TypeVar('N')  # The type of nodes in the interprocedural control-flow graph
D = TypeVar('D')  # The type of data-flow facts to be computed by the tabulation problem
M = TypeVar('M')  # The type of objects used to represent methods
V = TypeVar('V')  # The type of values to be computed along flow edges


class EdgeFunctions(ABC, Generic[N, D, M, V]):
    """
    Classes implementing this interface provide a range of edge functions used to
    compute a V-type value for each of the finitely many D-type values reachable
    in the program.
    
    Type Parameters:
        N: The type of nodes in the interprocedural control-flow graph.
           Typically Unit in Soot.
        D: The type of data-flow facts to be computed by the tabulation problem.
        M: The type of objects used to represent methods. Typically SootMethod.
        V: The type of values to be computed along flow edges.
    """
    
    @abstractmethod
    def get_normal_edge_function(self, curr: N, curr_node: D, succ: N, succ_node: D) -> 'EdgeFunction[V]':
        """
        Returns the function that computes how the V-typed value changes when
        being propagated from src_node at statement src to tgt_node at statement tgt.
        
        This handles normal intraprocedural flow within a method (e.g., from one
        statement to its successor in the control flow graph).
        
        Args:
            curr: The statement from which the flow originates.
            curr_node: The D-type value with which the source value is associated.
            succ: The target statement of the flow.
            succ_node: The D-type value with which the target value will be associated.
            
        Returns:
            An EdgeFunction that transforms values along this normal flow edge.
        """
        pass
    
    @abstractmethod
    def get_call_edge_function(self, call_stmt: N, src_node: D, 
                               destination_method: M, dest_node: D) -> 'EdgeFunction[V]':
        """
        Returns the function that computes how the V-typed value changes when
        being propagated along a method call.
        
        This handles the flow from a call site into the entry of the called method.
        
        Args:
            call_stmt: The call statement from which the flow originates.
            src_node: The D-type value with which the source value is associated.
            destination_method: A concrete destination method of the call.
            dest_node: The D-type value with which the target value will be
                      associated at the side of the callee.
                      
        Returns:
            An EdgeFunction that transforms values along this call edge.
        """
        pass
    
    @abstractmethod
    def get_return_edge_function(self, call_site: Optional[N], callee_method: M, 
                                exit_stmt: N, exit_node: D, 
                                return_site: Optional[N], ret_node: D) -> 'EdgeFunction[V]':
        """
        Returns the function that computes how the V-typed value changes when
        being propagated along a method exit (return or throw).
        
        This handles the flow from a method exit back to the call site.
        
        Args:
            call_site: One of all the call sites in the program that called the
                      method from which the exit_stmt is actually returning. This
                      information can be exploited to compute a value that depends on
                      information from before the call.
                      Note: This value might be None if using a tabulation problem
                      with followReturnsPastSeeds() returning True in a situation
                      where the call graph does not contain a caller for the method
                      that is returned from.
            callee_method: The method from which we are exiting.
            exit_stmt: The exit statement from which the flow originates.
            exit_node: The D-type value with which the source value is associated.
            return_site: One of the possible successor statements of a caller to the
                        method we are exiting from.
                        Note: This value might be None if using a tabulation problem
                        with followReturnsPastSeeds() returning True in a situation
                        where the call graph does not contain a caller for the method
                        that is returned from.
            ret_node: The D-type value with which the target value will be
                     associated at the return_site.
                     
        Returns:
            An EdgeFunction that transforms values along this return edge.
        """
        pass
    
    @abstractmethod
    def get_call_to_return_edge_function(self, call_site: N, call_node: D, 
                                        return_site: N, return_side_node: D) -> 'EdgeFunction[V]':
        """
        Returns the function that computes how the V-typed value changes when
        being propagated from a method call to one of its intraprocedural successors.
        
        This handles the flow that bypasses the called method, modeling data flow
        that is not affected by the call (e.g., local variables not passed to the method).
        
        Args:
            call_site: The call statement from which the flow originates.
            call_node: The D-type value with which the source value is associated.
            return_site: One of the possible successor statements of a call statement.
            return_side_node: The D-type value with which the target value will be
                            associated at the return_site.
                            
        Returns:
            An EdgeFunction that transforms values along this call-to-return edge.
        """
        pass


# Example concrete implementation for reference
class IdentityEdgeFunctions(EdgeFunctions[N, D, M, V]):
    """
    A simple implementation that returns identity edge functions for all edges.
    This is useful as a base case or for IFDS problems (where V is binary).
    """
    
    def __init__(self, identity_function: 'EdgeFunction[V]'):
        """
        Initialize with an identity edge function.
        
        Args:
            identity_function: The identity edge function to return for all edges
        """
        self.identity_function = identity_function
    
    def get_normal_edge_function(self, curr: N, curr_node: D, 
                                 succ: N, succ_node: D) -> 'EdgeFunction[V]':
        """Return identity function for normal edges."""
        return self.identity_function
    
    def get_call_edge_function(self, call_stmt: N, src_node: D, 
                              destination_method: M, dest_node: D) -> 'EdgeFunction[V]':
        """Return identity function for call edges."""
        return self.identity_function
    
    def get_return_edge_function(self, call_site: Optional[N], callee_method: M, 
                                exit_stmt: N, exit_node: D, 
                                return_site: Optional[N], ret_node: D) -> 'EdgeFunction[V]':
        """Return identity function for return edges."""
        return self.identity_function
    
    def get_call_to_return_edge_function(self, call_site: N, call_node: D, 
                                        return_site: N, return_side_node: D) -> 'EdgeFunction[V]':
        """Return identity function for call-to-return edges."""
        return self.identity_function


class AllTopEdgeFunctions(EdgeFunctions[N, D, M, V]):
    """
    An implementation that returns all-top edge functions for all edges.
    This represents "no information" and is useful as a default or base case.
    """
    
    def __init__(self, all_top_function: 'EdgeFunction[V]'):
        """
        Initialize with an all-top edge function.
        
        Args:
            all_top_function: The all-top edge function to return for all edges
        """
        self.all_top_function = all_top_function
    
    def get_normal_edge_function(self, curr: N, curr_node: D, 
                                 succ: N, succ_node: D) -> 'EdgeFunction[V]':
        """Return all-top function for normal edges."""
        return self.all_top_function
    
    def get_call_edge_function(self, call_stmt: N, src_node: D, 
                              destination_method: M, dest_node: D) -> 'EdgeFunction[V]':
        """Return all-top function for call edges."""
        return self.all_top_function
    
    def get_return_edge_function(self, call_site: Optional[N], callee_method: M, 
                                exit_stmt: N, exit_node: D, 
                                return_site: Optional[N], ret_node: D) -> 'EdgeFunction[V]':
        """Return all-top function for return edges."""
        return self.all_top_function
    
    def get_call_to_return_edge_function(self, call_site: N, call_node: D, 
                                        return_site: N, return_side_node: D) -> 'EdgeFunction[V]':
        """Return all-top function for call-to-return edges."""
        return self.all_top_function


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
 * Classes implementing this interface provide a range of edge functions used to
 * compute a V-type value for each of the finitely many D-type values reachable
 * in the program.
 * 
 * @param <N>
 *            The type of nodes in the interprocedural control-flow graph.
 *            Typically {@link Unit}.
 * @param <D>
 *            The type of data-flow facts to be computed by the tabulation
 *            problem.
 * @param <M>
 *            The type of objects used to represent methods. Typically
 *            {@link SootMethod}.
 * @param <V>
 *            The type of values to be computed along flow edges.
 */
public interface EdgeFunctions<N, D, M, V> {

	/**
	 * Returns the function that computes how the V-typed value changes when
	 * being propagated from srcNode at statement src to tgtNode at statement
	 * tgt.
	 * 
	 * @param curr
	 *            The statement from which the flow originates.
	 * @param currNode
	 *            The D-type value with which the source value is associated.
	 * @param succ
	 *            The target statement of the flow.
	 * @param succNode
	 *            The D-type value with which the target value will be
	 *            associated.
	 */
	public EdgeFunction<V> getNormalEdgeFunction(N curr, D currNode, N succ, D succNode);

	/**
	 * Returns the function that computes how the V-typed value changes when
	 * being propagated along a method call.
	 * 
	 * @param callStmt
	 *            The call statement from which the flow originates.
	 * @param srcNode
	 *            The D-type value with which the source value is associated.
	 * @param destinationMethod
	 *            A concrete destination method of the call.
	 * @param destNode
	 *            The D-type value with which the target value will be
	 *            associated at the side of the callee.
	 */
	public EdgeFunction<V> getCallEdgeFunction(N callStmt, D srcNode, M destinationMethod, D destNode);

	/**
	 * Returns the function that computes how the V-typed value changes when
	 * being propagated along a method exit (return or throw).
	 * 
	 * @param callSite
	 *            One of all the call sites in the program that called the
	 *            method from which the exitStmt is actually returning. This
	 *            information can be exploited to compute a value that depend on
	 *            information from before the call.
	 *            <b>Note:</b> This value might be <code>null</code> if
	 *            using a tabulation problem with {@link IFDSTabulationProblem#followReturnsPastSeeds()}
	 *            returning <code>true</code> in a situation where the call graph
	 *            does not contain a caller for the method that is returned from.
	 * @param calleeMethod
	 *            The method from which we are exiting.
	 * @param exitStmt
	 *            The exit statement from which the flow originates.
	 * @param exitNode
	 *            The D-type value with which the source value is associated.
	 * @param returnSite
	 *            One of the possible successor statements of a caller to the
	 *            method we are exiting from.
	 *            <b>Note:</b> This value might be <code>null</code> if
	 *            using a tabulation problem with {@link IFDSTabulationProblem#followReturnsPastSeeds()}
	 *            returning <code>true</code> in a situation where the call graph
	 *            does not contain a caller for the method that is returned from.
	 * @param tgtNode
	 *            The D-type value with which the target value will be
	 *            associated at the returnSite.
	 */
	public EdgeFunction<V> getReturnEdgeFunction(N callSite, M calleeMethod, N exitStmt, D exitNode, N returnSite, D retNode);

	/**
	 * Returns the function that computes how the V-typed value changes when
	 * being propagated from a method call to one of its intraprocedural
	 * successor.
	 * 
	 * @param callSite
	 *            The call statement from which the flow originates.
	 * @param callNode
	 *            The D-type value with which the source value is associated.
	 * @param returnSite
	 *            One of the possible successor statements of a call statement.
	 * @param returnSideNode
	 *            The D-type value with which the target value will be
	 *            associated at the returnSite.
	 */
	public EdgeFunction<V> getCallToReturnEdgeFunction(N callSite, D callNode, N returnSite, D returnSideNode);

}
\n"""