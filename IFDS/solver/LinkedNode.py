"""
Copyright (c) 2013 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import warnings

# Type variable
D = TypeVar('D')


class LinkedNode(ABC, Generic[D]):
    """
    A data-flow fact that can be linked with other equal facts.
    Equality and hash-code operations must NOT take the linking data structures into account!
    
    .. deprecated::
        Use JoinHandlingNode instead.
    
    Type Parameters:
        D: The type of the data-flow fact
    """
    
    def __init__(self):
        """Initialize LinkedNode and issue deprecation warning."""
        warnings.warn(
            "LinkedNode is deprecated. Use JoinHandlingNode instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    @abstractmethod
    def add_neighbor(self, original_abstraction: D):
        """
        Links this node to a neighbor node, i.e., to an abstraction that would have been merged
        with this one if paths were not being tracked.
        
        Args:
            original_abstraction: The neighbor abstraction to link to
        """
        pass
    
    @abstractmethod
    def set_calling_context(self, calling_context: D):
        """
        Set the calling context for this node.
        
        Args:
            calling_context: The calling context to set
        """
        pass


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

/**
 * A data-flow fact that can be linked with other equal facts.
 * Equality and hash-code operations must <i>not</i> take the linking data structures into account!
 * 
 * @deprecated Use {@link JoinHandlingNode} instead.
 */
@Deprecated
public interface LinkedNode<D> {
	/**
	 * Links this node to a neighbor node, i.e., to an abstraction that would have been merged
	 * with this one of paths were not being tracked.
	 */
	public void addNeighbor(D originalAbstraction);
	
	public void setCallingContext(D callingContext);
}\n"""