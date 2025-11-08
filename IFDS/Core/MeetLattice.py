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

# Type variable
V = TypeVar('V')


class MeetLattice(ABC, Generic[V]):
    """
    This class defines a lattice in terms of its top and bottom elements
    and a meet operation. This is meant to be a complete lattice, with a unique 
    top and bottom element.
    
    Type Parameters:
        V: The domain type for this lattice.
    """
    
    @abstractmethod
    def top_element(self) -> V:
        """
        Returns the unique top element of this lattice.
        
        Returns:
            The top element of the lattice
        """
        pass
    
    @abstractmethod
    def bottom_element(self) -> V:
        """
        Returns the unique bottom element of this lattice.
        
        Returns:
            The bottom element of the lattice
        """
        pass
    
    @abstractmethod
    def meet(self, left: V, right: V) -> V:
        """
        Computes the meet of left and right.
        
        Note that:
            meet(top, x) = meet(x, top) = x
            meet(bottom, x) = meet(x, bottom) = bottom
        
        Args:
            left: The left operand
            right: The right operand
            
        Returns:
            The meet of left and right
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

/**
 * This class defines a lattice in terms of its top and bottom elements
 * and a meet operation. This is meant to be a complete lattice, with a unique top and bottom element. 
 *
 * @param <V> The domain type for this lattice.
 */
public interface MeetLattice<V> {
	
	/**
	 * Returns the unique top element of this lattice.
	 */
	V topElement();
	
	/**
	 * Returns the unique bottom element of this lattice.
	 */
	V bottomElement();
	
	/**
	 * Computes the meet of left and right. Note that <pre>meet(top,x) = meet(x,top) = x</pre> and
	 * <pre>meet(bottom,x) = meet(x,bottom) = bottom</pre>. 
	 */
	V meet(V left, V right);

}
\n"""