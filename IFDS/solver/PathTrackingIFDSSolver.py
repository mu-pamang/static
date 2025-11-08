"""
Copyright (c) 2013 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
    '내가 지금까지 보낸거 종합적으로 피드백 해줘' 
"""

from typing import TypeVar, Generic, Dict
from threading import Lock
from dataclasses import dataclass
import warnings
from .LinkedNode import LinkedNode
from Core.IFDSTabulationProblem import IFDSTabulationProblem
from Core.EdgeFunction import EdgeFunction
from .IFDSSolver import BinaryDomain

# Type variables
N = TypeVar('N')
D = TypeVar('D', bound='LinkedNode')
M = TypeVar('M')
I = TypeVar('I')


@dataclass(frozen=True)
class CacheEntry(Generic[N, D]):
    """
    Cache entry for tracking propagated values.
    """
    n: N
    source_val: D
    target_val: D
    
    def __hash__(self) -> int:
        """
        Compute hash code for the cache entry.
        
        Returns:
            Hash code based on n, source_val, and target_val
        """
        prime = 31
        result = 1
        result = prime * result + (0 if self.source_val is None else hash(self.source_val))
        result = prime * result + (0 if self.target_val is None else hash(self.target_val))
        result = prime * result + (0 if self.n is None else hash(self.n))
        return result
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality with another cache entry.
        
        Args:
            other: The other object to compare with
            
        Returns:
            True if equal, False otherwise
        """
        if self is other:
            return True
        if other is None:
            return False
        if not isinstance(other, CacheEntry):
            return False
        
        if self.source_val is None:
            if other.source_val is not None:
                return False
        elif self.source_val != other.source_val:
            return False
        
        if self.target_val is None:
            if other.target_val is not None:
                return False
        elif self.target_val != other.target_val:
            return False
        
        if self.n is None:
            if other.n is not None:
                return False
        elif self.n != other.n:
            return False
        
        return True


class PathTrackingIFDSSolver(Generic[N, D, M, I]):
    """
    An IFDSSolver that tracks paths for reporting. To do so, it requires that 
    data-flow abstractions implement the LinkedNode interface.
    
    The solver implements a cache of data-flow facts for each statement and source value. 
    If for the same statement and source value the same target value is seen again 
    (as determined through a cache hit), then the solver propagates the cached value 
    but at the same time links both target values with one another.
    
    .. deprecated::
        Use JoinHandlingNodesIFDSSolver instead.
    
    Type Parameters:
        N: The type of nodes in the interprocedural control-flow graph
        D: The type of data-flow facts (must implement LinkedNode)
        M: The type of objects used to represent methods
        I: The type of inter-procedural control-flow graph being used
    
    Author:
        Eric Bodden
    """
    
    def __init__(self, ifds_problem: 'IFDSTabulationProblem[N, D, M, I]'):
        """
        Initialize the solver with the given IFDS problem.
        
        Issues a deprecation warning as this class is deprecated.
        
        Args:
            ifds_problem: The IFDS tabulation problem to solve
        """
        warnings.warn(
            "PathTrackingIFDSSolver is deprecated. Use JoinHandlingNodesIFDSSolver instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Call parent IFDSSolver constructor
        super().__init__(ifds_problem)
        
        self.cache: Dict[CacheEntry[N, D], D] = {}
        self.cache_lock = Lock()
    
    def propagate(self, source_val: D, target: N, target_val: D,
                 f: 'EdgeFunction[BinaryDomain]', related_call_site: N,
                 is_unbalanced_return: bool):
        """
        Propagate data-flow facts with path tracking.
        
        This method checks if the same target value has been seen before for the
        same statement and source value. If so, it links the existing target value
        with the new one as neighbors.
        
        Args:
            source_val: The source value
            target: The target node
            target_val: The target value
            f: The edge function
            related_call_site: The related call site
            is_unbalanced_return: Whether this is an unbalanced return
        """
        current_cache_entry = CacheEntry(target, source_val, target_val)
        
        propagate = False
        
        with self.cache_lock:
            if current_cache_entry in self.cache:
                existing_target_val = self.cache[current_cache_entry]
                if existing_target_val is not target_val:
                    existing_target_val.add_neighbor(target_val)
            else:
                self.cache[current_cache_entry] = target_val
                propagate = True
        
        if propagate:
            super().propagate(source_val, target, target_val, f, 
                            related_call_site, is_unbalanced_return)


"""\n===== ORIGINAL SOURCE FOR REFERENCE =====\n/*******************************************************************************
 * Copyright (c) 2013 Eric Bodden.
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
import heros.IFDSTabulationProblem;
import heros.InterproceduralCFG;

import java.util.Map;

import com.google.common.collect.Maps;

/**
 * An {@link IFDSSolver} that tracks paths for reporting. To do so, it requires that data-flow abstractions implement the LinkedNode interface.
 * The solver implements a cache of data-flow facts for each statement and source value. If for the same statement and source value the same
 * target value is seen again (as determined through a cache hit), then the solver propagates the cached value but at the same time links
 * both target values with one another.
 *  
 * @author Eric Bodden
 * @deprecated Use {@link JoinHandlingNodesIFDSSolver} instead.
 */
@Deprecated
public class PathTrackingIFDSSolver<N, D extends LinkedNode<D>, M, I extends InterproceduralCFG<N, M>> extends IFDSSolver<N, D, M, I> {

	public PathTrackingIFDSSolver(IFDSTabulationProblem<N, D, M, I> ifdsProblem) {
		super(ifdsProblem);
	}

	protected final Map<CacheEntry, LinkedNode<D>> cache = Maps.newHashMap();
	
	@Override
	protected void propagate(D sourceVal, N target, D targetVal, EdgeFunction<IFDSSolver.BinaryDomain> f, N relatedCallSite, boolean isUnbalancedReturn) {
		CacheEntry currentCacheEntry = new CacheEntry(target, sourceVal, targetVal);

		boolean propagate = false;
		synchronized (this) {
			if (cache.containsKey(currentCacheEntry)) {
				LinkedNode<D> existingTargetVal = cache.get(currentCacheEntry);
				if (existingTargetVal != targetVal)
					existingTargetVal.addNeighbor(targetVal);
			} else {
				cache.put(currentCacheEntry, targetVal);
				propagate = true;
			}
		}

		if (propagate)
			super.propagate(sourceVal, target, targetVal, f, relatedCallSite, isUnbalancedReturn);
		
	};
	
	
	private class CacheEntry {
		private N n;
		private D sourceVal;
		private D targetVal;

		public CacheEntry(N n, D sourceVal, D targetVal) {
			super();
			this.n = n;
			this.sourceVal = sourceVal;
			this.targetVal = targetVal;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((sourceVal == null) ? 0 : sourceVal.hashCode());
			result = prime * result + ((targetVal == null) ? 0 : targetVal.hashCode());
			result = prime * result + ((n == null) ? 0 : n.hashCode());
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
			@SuppressWarnings({ "unchecked" })
			CacheEntry other = (CacheEntry) obj;
			if (sourceVal == null) {
				if (other.sourceVal != null)
					return false;
			} else if (!sourceVal.equals(other.sourceVal))
				return false;
			if (targetVal == null) {
				if (other.targetVal != null)
					return false;
			} else if (!targetVal.equals(other.targetVal))
				return false;
			if (n == null) {
				if (other.n != null)
					return false;
			} else if (!n.equals(other.n))
				return false;
			return true;
		}
	}	
	


}
\n"""