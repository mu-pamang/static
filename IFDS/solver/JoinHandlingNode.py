"""
Copyright (c) 2014 Johannes Lerch.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Johannes Lerch - initial API and implementation
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple, Any

# Type variable
T = TypeVar('T')


class JoinKey:
    """
    A JoinKey object used to identify which node abstractions require manual join handling.
    """
    
    def __init__(self, *elements: Any):
        """
        Initialize a JoinKey with the given elements.
        
        Args:
            *elements: Passed elements must be immutable with respect to their 
                      hash and equality implementations.
        """
        self.elements: Tuple[Any, ...] = elements
    
    def __hash__(self) -> int:
        """
        Compute hash code for the JoinKey.
        
        Returns:
            Hash code based on the elements
        """
        return hash(self.elements)
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality with another JoinKey.
        
        Args:
            other: The other object to compare with
            
        Returns:
            True if equal, False otherwise
        """
        if self is other:
            return True
        if other is None:
            return False
        if not isinstance(other, JoinKey):
            return False
        return self.elements == other.elements
    
    def __repr__(self) -> str:
        """
        String representation of the JoinKey.
        
        Returns:
            String showing the elements
        """
        return f"JoinKey{self.elements}"


class JoinHandlingNode(ABC, Generic[T]):
    """
    Interface for nodes that require manual join handling.
    
    Type Parameters:
        T: The type of the node abstraction
    """
    
    @abstractmethod
    def handle_join(self, joining_node: T) -> bool:
        """
        Handle the join with another node that was propagated to the same target.
        
        Args:
            joining_node: The node abstraction that was propagated to the same target 
                         after this node.
                         
        Returns:
            True if the join could be handled and no further propagation of the 
            joining_node is necessary, otherwise False meaning the node should be 
            propagated by the solver.
        """
        pass
    
    @abstractmethod
    def create_join_key(self) -> JoinKey:
        """
        Create a JoinKey object used to identify which node abstractions require 
        manual join handling. For nodes with equal JoinKey instances, handle_join() 
        will be called.
        
        Returns:
            A JoinKey object for this node
        """
        pass
    
    @abstractmethod
    def set_calling_context(self, calling_context: T):
        """
        Set the calling context for this node.
        
        Args:
            calling_context: The calling context to set
        """
        pass


"""\n===== ORIGINAL JAVA FOR REFERENCE =====\n/*******************************************************************************
 * Copyright (c) 2014 Johannes Lerch.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Johannes Lerch - initial API and implementation
 ******************************************************************************/
package heros.solver;

import java.util.Arrays;


public interface JoinHandlingNode<T> {

	/**
	 * 
	 * @param joiningNode the node abstraction that was propagated to the same target after {@code this} node.
	 * @return true if the join could be handled and no further propagation of the {@code joiningNode} is necessary, otherwise false meaning 
	 * the node should be propagated by the solver.
	 */
	public boolean handleJoin(T joiningNode);
	
	/**
	 * 
	 * @return a JoinKey object used to identify which node abstractions require manual join handling. 
	 * For nodes with {@code equal} JoinKey instances {@link #handleJoin(JoinHandlingNode)} will be called.
	 */
	public JoinKey createJoinKey();
	
	public void setCallingContext(T callingContext);
	
	public static class JoinKey {
		private Object[] elements;

		/**
		 * 
		 * @param elements Passed elements must be immutable with respect to their hashCode and equals implementations.
		 */
		public JoinKey(Object... elements) {
			this.elements = elements;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + Arrays.hashCode(elements);
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			JoinKey other = (JoinKey) obj;
			if (!Arrays.equals(elements, other.elements))
				return false;
			return true;
		}
	}
}
\n"""