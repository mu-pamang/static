"""
Copyright (c) 2013 Johannes Lerch.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Johannes Lerch - initial API and implementation
"""

from typing import TypeVar, Generic, Dict
from threading import Lock
from dataclasses import dataclass
from .IFDSSolver import BinaryDomain
from .JoinHandlingNode import JoinHandlingNode
from Core.EdgeFunction import EdgeFunction
from Core.IFDSTabulationProblem import IFDSTabulationProblem
from .JoinHandlingNode import JoinKey

# Type variables
N = TypeVar('N')
D = TypeVar('D', bound='JoinHandlingNode')
M = TypeVar('M')
I = TypeVar('I')


@dataclass(frozen=True)
class CacheEntry(Generic[N]):
    """
    Cache entry for tracking propagated values.
    """
    n: N
    source_key: 'JoinKey'
    target_key: 'JoinKey'
    
    def __hash__(self) -> int:
        """
        Compute hash code for the cache entry.
        
        Returns:
            Hash code based on n, source_key, and target_key
        """
        result = 1
        prime = 31
        result = prime * result + (0 if self.source_key is None else hash(self.source_key))
        result = prime * result + (0 if self.target_key is None else hash(self.target_key))
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
        
        if self.source_key is None:
            if other.source_key is not None:
                return False
        elif self.source_key != other.source_key:
            return False
        
        if self.target_key is None:
            if other.target_key is not None:
                return False
        elif self.target_key != other.target_key:
            return False
        
        if self.n is None:
            if other.n is not None:
                return False
        elif self.n != other.n:
            return False
        
        return True


class JoinHandlingNodesIFDSSolver(Generic[N, D, M, I]):
    """
    An IFDSSolver that tracks paths for reporting. To do so, it requires that 
    data-flow abstractions implement the JoinHandlingNode interface.
    
    The solver implements a cache of data-flow facts for each statement and source value. 
    If for the same statement and source value the same target value is seen again 
    (as determined through a cache hit), then the solver propagates the cached value 
    but at the same time links both target values with one another.
    
    Type Parameters:
        N: The type of nodes in the interprocedural control-flow graph
        D: The type of data-flow facts (must implement JoinHandlingNode)
        M: The type of objects used to represent methods
        I: The type of inter-procedural control-flow graph being used
    
    Author:
        Johannes Lerch
    """
    
    def __init__(self, ifds_problem: 'IFDSTabulationProblem[N, D, M, I]'):
        """
        Initialize the solver with the given IFDS problem.
        
        Args:
            ifds_problem: The IFDS tabulation problem to solve
        """
        # Call parent IFDSSolver constructor
        super().__init__(ifds_problem)
        
        self.cache: Dict[CacheEntry[N], D] = {}
        self.cache_lock = Lock()
    
    def propagate(self, source_val: D, target: N, target_val: D,
                 f: 'EdgeFunction[BinaryDomain]', related_call_site: N,
                 is_unbalanced_return: bool):
        """
        Propagate data-flow facts with join handling.
        
        This method checks if the same target value has been seen before for the
        same statement and source value. If so, it attempts to handle the join.
        
        Args:
            source_val: The source value
            target: The target node
            target_val: The target value
            f: The edge function
            related_call_site: The related call site
            is_unbalanced_return: Whether this is an unbalanced return
        """
        current_cache_entry = CacheEntry(
            target,
            source_val.create_join_key(),
            target_val.create_join_key()
        )
        
        propagate = False
        
        with self.cache_lock:
            if current_cache_entry in self.cache:
                existing_target_val = self.cache[current_cache_entry]
                if not existing_target_val.handle_join(target_val):
                    propagate = True
            else:
                self.cache[current_cache_entry] = target_val
                propagate = True
        
        if propagate:
            super().propagate(source_val, target, target_val, f, 
                            related_call_site, is_unbalanced_return)


"""\n===== ORIGINAL JAVA FOR REFERENCE =====\n/*******************************************************************************
 * Copyright (c) 2013Johannes Lerch.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Johannes Lerch - initial API and implementation
 ******************************************************************************/
package heros.solver;

import heros.EdgeFunction;
import heros.IFDSTabulationProblem;
import heros.InterproceduralCFG;
import heros.solver.JoinHandlingNode.JoinKey;

import java.util.Map;

import com.google.common.collect.Maps;

/**
 * An {@link IFDSSolver} that tracks paths for reporting. To do so, it requires that data-flow abstractions implement the LinkedNode interface.
 * The solver implements a cache of data-flow facts for each statement and source value. If for the same statement and source value the same
 * target value is seen again (as determined through a cache hit), then the solver propagates the cached value but at the same time links
 * both target values with one another.
 *  
 * @author Johannes Lerch
 */
public class JoinHandlingNodesIFDSSolver<N, D extends JoinHandlingNode<D>, M, I extends InterproceduralCFG<N, M>> extends IFDSSolver<N, D, M, I> {

	public JoinHandlingNodesIFDSSolver(IFDSTabulationProblem<N, D, M, I> ifdsProblem) {
		super(ifdsProblem);
	}

	protected final Map<CacheEntry, JoinHandlingNode<D>> cache = Maps.newHashMap();
	
	@Override
	protected void propagate(D sourceVal, N target, D targetVal, EdgeFunction<IFDSSolver.BinaryDomain> f, N relatedCallSite, boolean isUnbalancedReturn) {
		CacheEntry currentCacheEntry = new CacheEntry(target, sourceVal.createJoinKey(), targetVal.createJoinKey());

		boolean propagate = false;
		synchronized (this) {
			if (cache.containsKey(currentCacheEntry)) {
				JoinHandlingNode<D> existingTargetVal = cache.get(currentCacheEntry);
				if(!existingTargetVal.handleJoin(targetVal)) {
					propagate = true;
				}
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
		private JoinKey sourceKey;
		private JoinKey targetKey;

		public CacheEntry(N n, JoinKey sourceKey, JoinKey targetKey) {
			super();
			this.n = n;
			this.sourceKey = sourceKey;
			this.targetKey = targetKey;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((sourceKey == null) ? 0 : sourceKey.hashCode());
			result = prime * result + ((targetKey == null) ? 0 : targetKey.hashCode());
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
			if (sourceKey == null) {
				if (other.sourceKey != null)
					return false;
			} else if (!sourceKey.equals(other.sourceKey))
				return false;
			if (targetKey == null) {
				if (other.targetKey != null)
					return false;
			} else if (!targetKey.equals(other.targetKey))
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