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
from typing import TypeVar, Generic

# Type variable for values computed along flow edges
V = TypeVar('V')


class EdgeFunction(ABC, Generic[V]):
    """
    An edge function computes how a V-type value changes when flowing from one
    super-graph node to another. See Sagiv, Reps, Horwitz 1996.
    
    NOTE: Methods defined on this type may be called simultaneously by different threads.
    Hence, classes implementing this interface should synchronize accesses to
    any mutable shared state.
    
    Type Parameters:
        V: The type of values to be computed along flow edges.
    """
    
    @abstractmethod
    def compute_target(self, source: V) -> V:
        """
        Computes the value resulting from applying this function to source.
        
        Args:
            source: The source value to transform
            
        Returns:
            The target value after applying this edge function
        """
        pass
    
    @abstractmethod
    def compose_with(self, second_function: 'EdgeFunction[V]') -> 'EdgeFunction[V]':
        """
        Composes this function with the second_function, effectively returning
        a summary function that maps sources to targets exactly as if
        first this function had been applied and then the second_function.
        
        This represents functional composition: (f ∘ g)(x) = f(g(x))
        
        Args:
            second_function: The function to compose with this one
            
        Returns:
            A new EdgeFunction representing the composition of this function
            followed by second_function
        """
        pass
    
    @abstractmethod
    def meet_with(self, other_function: 'EdgeFunction[V]') -> 'EdgeFunction[V]':
        """
        Returns a function that represents the (element-wise) meet
        of this function with other_function. Naturally, this is a
        symmetric operation.
        
        The meet operation combines two edge functions according to the
        meet operation of the underlying lattice.
        
        Args:
            other_function: The function to meet with this one
            
        Returns:
            A new EdgeFunction representing the meet of this function
            and other_function
            
        See Also:
            MeetLattice.meet(Object, Object)
        """
        pass
    
    @abstractmethod
    def equal_to(self, other: 'EdgeFunction[V]') -> bool:
        """
        Returns True if this function represents exactly the same 
        source to target mapping as other.
        
        This is semantic equality - two functions are equal if they
        produce the same output for all possible inputs, not necessarily
        if they are the same object.
        
        Args:
            other: The other edge function to compare with
            
        Returns:
            True if this function is semantically equal to other, False otherwise
        """
        pass
    
    # Optional: Add __eq__ for Python-style equality, delegating to equal_to
    def __eq__(self, other) -> bool:
        """
        Python equality operator. Delegates to equal_to for semantic equality.
        
        Args:
            other: The other object to compare with
            
        Returns:
            True if other is an EdgeFunction and equal_to returns True
        """
        if not isinstance(other, EdgeFunction):
            return False
        return self.equal_to(other)
    
    def __hash__(self):
        """
        Hash function for EdgeFunction. Should be overridden in concrete implementations
        to provide consistent hashing with equal_to semantics.
        
        Returns:
            Hash value for this edge function
        """
        # Default implementation - concrete classes should override
        return hash(self.__class__.__name__)


# Example concrete implementation for reference
class IdentityEdgeFunction(EdgeFunction[V]):
    """
    An edge function that returns its input unchanged (identity function).
    This is a common base case in edge function analysis.
    """
    
    def compute_target(self, source: V) -> V:
        """Return the source value unchanged."""
        return source
    
    def compose_with(self, second_function: EdgeFunction[V]) -> EdgeFunction[V]:
        """
        Composing with identity returns the second function.
        Identity is the neutral element for composition.
        """
        return second_function
    
    def meet_with(self, other_function: EdgeFunction[V]) -> EdgeFunction[V]:
        """
        Meeting with identity typically returns the other function,
        though this depends on the lattice structure.
        """
        return other_function
    
    def equal_to(self, other: EdgeFunction[V]) -> bool:
        """Check if other is also an identity function."""
        return isinstance(other, IdentityEdgeFunction)
    
    def __hash__(self):
        """All identity functions hash to the same value."""
        return hash("IdentityEdgeFunction")
    
    def __repr__(self):
        return "IdentityEdgeFunction()"


class AllTopEdgeFunction(EdgeFunction[V]):
    """
    An edge function that always returns the TOP value of the lattice.
    This typically represents "no information" or "unknown".
    """
    
    def __init__(self, top_value: V):
        """
        Initialize with the TOP value of the lattice.
        
        Args:
            top_value: The top element of the value lattice
        """
        self.top_value = top_value
    
    def compute_target(self, source: V) -> V:
        """Always return the top value."""
        return self.top_value
    
    def compose_with(self, second_function: EdgeFunction[V]) -> EdgeFunction[V]:
        """
        Composing all-top with any function results in all-top,
        since all-top produces top which maps to top under any function.
        """
        return self
    
    def meet_with(self, other_function: EdgeFunction[V]) -> EdgeFunction[V]:
        """
        Meeting with all-top returns the other function,
        as top is typically the neutral element for meet.
        """
        return other_function
    
    def equal_to(self, other: EdgeFunction[V]) -> bool:
        """Check if other is also an all-top function with the same top value."""
        return (isinstance(other, AllTopEdgeFunction) and 
                self.top_value == other.top_value)
    
    def __hash__(self):
        """Hash based on being an all-top function."""
        return hash(("AllTopEdgeFunction", self.top_value))
    
    def __repr__(self):
        return f"AllTopEdgeFunction(top_value={self.top_value})"

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
 * An edge function computes how a V-type value changes when flowing from one
 * super-graph node to another. See Sagiv, Reps, Horwitz 1996.
 * 
 * <b>NOTE:</b> Methods defined on this type may be called simultaneously by different threads.
 * Hence, classes implementing this interface should synchronize accesses to
 * any mutable shared state.
 *  
 * @param <V> The type of values to be computed along flow edges.
 */
public interface EdgeFunction<V> {

	/**
	 * Computes the value resulting from applying this function to source.
	 */
	V computeTarget(V source);
	
	/**
	 * Composes this function with the secondFunction, effectively returning
	 * a summary function that maps sources to targets exactly as if
	 * first this function had been applied and then the secondFunction. 
	 */
	EdgeFunction<V> composeWith(EdgeFunction<V> secondFunction);
	
	/**
	 * Returns a function that represents that (element-wise) meet
	 * of this function with otherFunction. Naturally, this is a
	 * symmetric operation.
	 * @see MeetLattice#meet(Object, Object)
	 */
	EdgeFunction<V> meetWith(EdgeFunction<V> otherFunction);
	
	/**
	 * Returns true is this function represents exactly the same 
	 * source to target mapping as other.
	 */
	public boolean equalTo(EdgeFunction<V> other);

}
\n"""