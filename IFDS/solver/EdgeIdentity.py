"""
Copyright (c) 2012 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
"""

from typing import TypeVar, Generic
from Core.EdgeFunction import EdgeFunction
from .AllBottom import AllBottom
from .AllTop import AllTop


# Type variable for values computed along flow edges
V = TypeVar('V')


class EdgeIdentity(EdgeFunction[V], Generic[V]):
    """
    The identity function on graph edges.
    
    This is a singleton class that represents the identity transformation
    where the output equals the input for all values.
    
    Type Parameters:
        V: The type of values to be computed along flow edges.
    """
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        """
        Ensure only one instance exists (singleton pattern).
        Use v() class method instead of direct instantiation.
        """
        if cls._instance is None:
            cls._instance = super(EdgeIdentity, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Private constructor. Use v() instead."""
        pass
    
    def compute_target(self, source: V) -> V:
        """
        Computes the target value by returning the source unchanged.
        
        Args:
            source: The source value
            
        Returns:
            The same value as the source (identity transformation)
        """
        return source
    
    def compose_with(self, second_function: EdgeFunction[V]) -> EdgeFunction[V]:
        """
        Composes this identity function with another function.
        
        Since this is the identity function, composing with any function f
        returns f itself: id ∘ f = f
        
        Args:
            second_function: The function to compose with
            
        Returns:
            The second_function unchanged
        """
        return second_function
    
    def meet_with(self, other_function: EdgeFunction[V]) -> EdgeFunction[V]:
        """
        Returns the meet of this identity function with another function.
        
        The meet operation combines two edge functions. For identity:
        - meet(id, id) = id
        - meet(id, AllBottom) = AllBottom (bottom absorbs)
        - meet(id, AllTop) = id (identity is more precise than top)
        - For other functions, delegate to the other function's meet_with
        
        Args:
            other_function: The function to meet with
            
        Returns:
            The result of the meet operation
        """
        # Check if it's the same instance or semantically equal
        if other_function is self or other_function.equal_to(self):
            return self
        
        # Import here to avoid circular dependencies
        from .AllBottom import AllBottom
        from .AllTop import AllTop
        
        # If other is AllBottom, bottom absorbs everything
        if isinstance(other_function, AllBottom):
            return other_function
        
        # If other is AllTop, identity is more precise
        if isinstance(other_function, AllTop):
            return self
        
        # Don't know how to meet; hence ask other function to decide on this
        return other_function.meet_with(self)
    
    def equal_to(self, other: EdgeFunction[V]) -> bool:
        """
        Check if this function is equal to another function.
        
        Since this is a singleton, equality is based on identity (same object).
        
        Args:
            other: The other edge function to compare
            
        Returns:
            True if other is the same singleton instance, False otherwise
        """
        return other is self
    
    @classmethod
    def v(cls) -> 'EdgeIdentity':
        """
        Get the singleton instance of EdgeIdentity.
        
        This is the preferred way to obtain an EdgeIdentity instance.
        
        Returns:
            The singleton EdgeIdentity instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __str__(self) -> str:
        """
        String representation of the identity function.
        
        Returns:
            "id" to represent the identity function
        """
        return "id"
    
    def __repr__(self) -> str:
        """
        Detailed string representation.
        
        Returns:
            A string showing this is an EdgeIdentity
        """
        return "EdgeIdentity()"
    
    def __hash__(self):
        """
        Hash function for EdgeIdentity.
        
        Since this is a singleton, all instances have the same hash.
        
        Returns:
            A consistent hash value for all EdgeIdentity instances
        """
        return hash("EdgeIdentity")
    
    # Prevent copying of singleton
    def __copy__(self):
        """Prevent copying - return the singleton instance."""
        return self
    
    def __deepcopy__(self, memo):
        """Prevent deep copying - return the singleton instance."""
        return self


# Convenience function to get the singleton instance
def edge_identity() -> EdgeIdentity:
    """
    Get the singleton EdgeIdentity instance.
    
    This is a convenience function equivalent to EdgeIdentity.v()
    
    Returns:
        The singleton EdgeIdentity instance
    """
    return EdgeIdentity.v()
    

"""\n===== ORIGINAL JAVA FOR REFERENCE =====\n/*******************************************************************************
 /*******************************************************************************
 * Copyright (c) 2012 Eric Bodden.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Eric Bodden - initial API and implementation
 ******************************************************************************/
package heros.edgefunc;

import heros.EdgeFunction;

/**
 * The identity function on graph edges
 * @param <V> The type of values to be computed along flow edges.
 */
public class EdgeIdentity<V> implements EdgeFunction<V> {
	
	@SuppressWarnings("rawtypes")
	private final static EdgeIdentity instance = new EdgeIdentity();
	
	private EdgeIdentity(){} //use v() instead

	public V computeTarget(V source) {
		return source;
	}

	public EdgeFunction<V> composeWith(EdgeFunction<V> secondFunction) {
		return secondFunction;
	}

	public EdgeFunction<V> meetWith(EdgeFunction<V> otherFunction) {
		if(otherFunction == this || otherFunction.equalTo(this)) return this;
		if(otherFunction instanceof AllBottom) {
			return otherFunction;
		}
		if(otherFunction instanceof AllTop) {
			return this;
		}
		//do not know how to meet; hence ask other function to decide on this
		return otherFunction.meetWith(this);
	}
	
	public boolean equalTo(EdgeFunction<V> other) {
		//singleton
		return other==this;
	}

	@SuppressWarnings("unchecked")
	public static <A> EdgeIdentity<A> v() {
		return instance;
	}

	public String toString() {
		return "id";
	}


}

\n"""