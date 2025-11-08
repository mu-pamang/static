"""
Copyright (c) 2012 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
"""

from typing import TypeVar, Generic, Optional

# Type variables
T = TypeVar('T')
U = TypeVar('U')


class Pair(Generic[T, U]):
    """
    A generic pair class that holds two values.
    
    Copied from soot.toolkits.scalar
    
    Type Parameters:
        T: The type of the first element
        U: The type of the second element
    """
    
    def __init__(self, o1: Optional[T] = None, o2: Optional[U] = None):
        """
        Initialize a Pair with two values.
        
        Args:
            o1: The first element (default: None)
            o2: The second element (default: None)
        """
        self.o1 = o1
        self.o2 = o2
        self._hash_code = 0
    
    def __hash__(self) -> int:
        """
        Compute hash code for the Pair.
        
        Returns:
            Hash code based on o1 and o2
        """
        if self._hash_code != 0:
            return self._hash_code
        
        prime = 31
        result = 1
        result = prime * result + (0 if self.o1 is None else hash(self.o1))
        result = prime * result + (0 if self.o2 is None else hash(self.o2))
        self._hash_code = result
        
        return self._hash_code
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality with another Pair.
        
        Args:
            other: The other object to compare with
            
        Returns:
            True if equal, False otherwise
        """
        if self is other:
            return True
        if other is None:
            return False
        if not isinstance(other, Pair):
            return False
        
        if self.o1 is None:
            if other.o1 is not None:
                return False
        elif self.o1 != other.o1:
            return False
        
        if self.o2 is None:
            if other.o2 is not None:
                return False
        elif self.o2 != other.o2:
            return False
        
        return True
    
    def __str__(self) -> str:
        """
        String representation of the Pair.
        
        Returns:
            String in the format "Pair o1,o2"
        """
        return f"Pair {self.o1},{self.o2}"
    
    def __repr__(self) -> str:
        """
        Detailed string representation of the Pair.
        
        Returns:
            String showing the type and values
        """
        return f"Pair({self.o1!r}, {self.o2!r})"
    
    def get_o1(self) -> Optional[T]:
        """
        Get the first element.
        
        Returns:
            The first element
        """
        return self.o1
    
    def get_o2(self) -> Optional[U]:
        """
        Get the second element.
        
        Returns:
            The second element
        """
        return self.o2
    
    def set_o1(self, no1: T):
        """
        Set the first element and reset hash code.
        
        Args:
            no1: The new first element
        """
        self.o1 = no1
        self._hash_code = 0
    
    def set_o2(self, no2: U):
        """
        Set the second element and reset hash code.
        
        Args:
            no2: The new second element
        """
        self.o2 = no2
        self._hash_code = 0
    
    def set_pair(self, no1: T, no2: U):
        """
        Set both elements and reset hash code.
        
        Args:
            no1: The new first element
            no2: The new second element
        """
        self.o1 = no1
        self.o2 = no2
        self._hash_code = 0


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

//copied from soot.toolkits.scalar
public class Pair<T, U> {
	protected T o1;
	protected U o2;
	
	protected int hashCode = 0;

	public Pair() {
		o1 = null;
		o2 = null;
	}

	public Pair(T o1, U o2) {
		this.o1 = o1;
		this.o2 = o2;
	}

	@Override
	public int hashCode() {
		if (hashCode != 0)
			return hashCode;
		
		final int prime = 31;
		int result = 1;
		result = prime * result + ((o1 == null) ? 0 : o1.hashCode());
		result = prime * result + ((o2 == null) ? 0 : o2.hashCode());
		hashCode = result;
		
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
		Pair other = (Pair) obj;
		if (o1 == null) {
			if (other.o1 != null)
				return false;
		} else if (!o1.equals(other.o1))
			return false;
		if (o2 == null) {
			if (other.o2 != null)
				return false;
		} else if (!o2.equals(other.o2))
			return false;
		return true;
	}

	public String toString() {
		return "Pair " + o1 + "," + o2;
	}

	public T getO1() {
		return o1;
	}

	public U getO2() {
		return o2;
	}

	public void setO1(T no1) {
		o1 = no1;
		hashCode = 0;
	}

	public void setO2(U no2) {
		o2 = no2;
		hashCode = 0;
	}

	public void setPair(T no1, U no2) {
		o1 = no1;
		o2 = no2;
		hashCode = 0;
	}

}
\n"""