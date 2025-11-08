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
from typing import Set, TypeVar, Generic


# Type variable for data-flow facts
D = TypeVar('D')


class FlowFunction(ABC, Generic[D]):
    """
    A flow function computes which of the finitely many D-type values are reachable
    from the current source values. Typically there will be one such function
    associated with every possible control flow.
    
    NOTE: To be able to produce **deterministic benchmarking results**, we have found that
    it helps to return sets with preserved insertion order from compute_targets(). This is
    because the duration of IDE's fixed point iteration may depend on the iteration order.
    Within the solver, we have tried to fix this order as much as possible, but the
    order, in general, does also depend on the order in which the result set
    of compute_targets() is traversed.
    
    NOTE: Methods defined on this type may be called simultaneously by different threads.
    Hence, classes implementing this interface should synchronize accesses to
    any mutable shared state.
    
    Type Parameters:
        D: The type of data-flow facts to be computed by the tabulation problem.
    """
    
    @abstractmethod
    def compute_targets(self, source: D) -> Set[D]:
        """
        Returns the target values reachable from the source.
        
        Args:
            source: The source data-flow fact
            
        Returns:
            Set of target data-flow facts reachable from the source
        """
        pass