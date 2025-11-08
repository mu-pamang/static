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

# Type variables
N = TypeVar('N')  # The type of nodes in the interprocedural control-flow graph
D = TypeVar('D')  # The type of data-flow facts to be computed by the tabulation problem


class PathEdge(Generic[N, D]):
    """
    A path edge as described in the IFDS/IDE algorithms.
    The source node is implicit: it can be computed from the target by using the 
    InterproceduralCFG. Hence, we don't store it.
    
    Type Parameters:
        N: The type of nodes in the interprocedural control-flow graph. Typically Unit.
        D: The type of data-flow facts to be computed by the tabulation problem.
    """
    
    def __init__(self, d_source: D, target: N, d_target: D):
        """
        Initialize a PathEdge.
        
        Args:
            d_source: The fact at the source.
            target: The target statement.
            d_target: The fact at the target.
        """
        self.target = target
        self.d_source = d_source
        self.d_target = d_target
        
        # Pre-compute hash code
        prime = 31
        result = 1
        result = prime * result + (0 if d_source is None else hash(d_source))
        result = prime * result + (0 if d_target is None else hash(d_target))
        result = prime * result + (0 if target is None else hash(target))
        self._hash_code = result
    
    def get_target(self) -> N:
        """
        Get the target node.
        
        Returns:
            The target statement
        """
        return self.target
    
    def fact_at_source(self) -> D:
        """
        Get the fact at the source.
        
        Returns:
            The fact at the source
        """
        return self.d_source
    
    def fact_at_target(self) -> D:
        """
        Get the fact at the target.
        
        Returns:
            The fact at the target
        """
        return self.d_target
    
    def __hash__(self) -> int:
        """
        Get the pre-computed hash code.
        
        Returns:
            The hash code
        """
        return self._hash_code
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality with another PathEdge.
        
        Args:
            other: The other object to compare with
            
        Returns:
            True if equal, False otherwise
        """
        if self is other:
            return True
        if other is None:
            return False
        if not isinstance(other, PathEdge):
            return False
        
        if self.d_source is None:
            if other.d_source is not None:
                return False
        elif self.d_source != other.d_source:
            return False
        
        if self.d_target is None:
            if other.d_target is not None:
                return False
        elif self.d_target != other.d_target:
            return False
        
        if self.target is None:
            if other.target is not None:
                return False
        elif self.target != other.target:
            return False
        
        return True
    
    def __str__(self) -> str:
        """
        String representation of the PathEdge.
        
        Returns:
            String in the format "<dSource> -> <target,dTarget>"
        """
        return f"<{self.d_source}> -> <{self.target},{self.d_target}>"
    
    def __repr__(self) -> str:
        """
        Detailed string representation of the PathEdge.
        
        Returns:
            String showing the constructor call
        """
        return f"PathEdge({self.d_source!r}, {self.target!r}, {self.d_target!r})"


"""\n===== ORIGINAL SOURCE FOR REFERENCE =====\n/*******************************************************************************
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

import heros.InterproceduralCFG;

/**
 * A path edge as described in the IFDS/IDE algorithms.
 * The source node is implicit: it can be computed from the target by using the {@link InterproceduralCFG}.
 * Hence, we don't store it.
 *
 * @param <N> The type of nodes in the interprocedural control-flow graph. Typically {@link Unit}.
 * @param <D> The type of data-flow facts to be computed by the tabulation problem.
 */
public class PathEdge<N,D> {

	protected final N target;
	protected final D dSource, dTarget;
	protected final int hashCode;

	/**
	 * @param dSource The fact at the source.
	 * @param target The target statement.
	 * @param dTarget The fact at the target.
	 */
	public PathEdge(D dSource, N target, D dTarget) {
		super();
		this.target = target;
		this.dSource = dSource;
		this.dTarget = dTarget;
		
		final int prime = 31;
		int result = 1;
		result = prime * result + ((dSource == null) ? 0 : dSource.hashCode());
		result = prime * result + ((dTarget == null) ? 0 : dTarget.hashCode());
		result = prime * result + ((target == null) ? 0 : target.hashCode());
		this.hashCode = result;
	}
	
	public N getTarget() {
		return target;
	}

	public D factAtSource() {
		return dSource;
	}

	public D factAtTarget() {
		return dTarget;
	}

	@Override
	public int hashCode() {
		return hashCode;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		@SuppressWarnings("rawtypes")
		PathEdge other = (PathEdge) obj;
		if (dSource == null) {
			if (other.dSource != null)
				return false;
		} else if (!dSource.equals(other.dSource))
			return false;
		if (dTarget == null) {
			if (other.dTarget != null)
				return false;
		} else if (!dTarget.equals(other.dTarget))
			return false;
		if (target == null) {
			if (other.target != null)
				return false;
		} else if (!target.equals(other.target))
			return false;
		return true;
	}

	@Override
	public String toString() {
		StringBuffer result = new StringBuffer();
		result.append("<");
		result.append(dSource);
		result.append("> -> <");
		result.append(target.toString());
		result.append(",");
		result.append(dTarget);
		result.append(">");
		return result.toString();
	}

}
\n"""