"""
Copyright (c) 2012 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
"""

from typing import TypeVar, Generic, Dict, Set, Tuple
from threading import Lock
from collections import defaultdict
from Core.EdgeFunction import EdgeFunction

# Type variables
N = TypeVar('N')
D = TypeVar('D')
L = TypeVar('L')


class JumpFunctions(Generic[N, D, L]):
    """
    The IDE algorithm uses a list of jump functions. Instead of a list, we use a set of three
    maps that are kept in sync. This allows for efficient indexing: the algorithm accesses
    elements from the list through three different indices.
    
    This class is thread-safe.
    
    Type Parameters:
        N: The type of nodes in the control-flow graph
        D: The type of data-flow facts
        L: The type of values in the lattice
    """
    
    def __init__(self, all_top: 'EdgeFunction[L]'):
        """
        Initialize the JumpFunctions with the all-top edge function.
        
        Args:
            all_top: The edge function representing all-top
        """
        self.all_top = all_top
        self._lock = Lock()
        
        # Mapping from target node and value to a list of all source values and associated functions
        # where the list is implemented as a mapping from the source value to the function
        # we exclude empty default functions
        # Structure: {(target, targetVal): {sourceVal: EdgeFunction}}
        self.non_empty_reverse_lookup: Dict[Tuple[N, D], Dict[D, 'EdgeFunction[L]']] = {}
        
        # Mapping from source value and target node to a list of all target values and associated functions
        # where the list is implemented as a mapping from the target value to the function
        # we exclude empty default functions
        # Structure: {(sourceVal, target): {targetVal: EdgeFunction}}
        self.non_empty_forward_lookup: Dict[Tuple[D, N], Dict[D, 'EdgeFunction[L]']] = {}
        
        # A mapping from target node to a list of triples consisting of source value,
        # target value and associated function; the triple is implemented by a table
        # we exclude empty default functions
        # Structure: {target: {(sourceVal, targetVal): EdgeFunction}}
        self.non_empty_lookup_by_target_node: Dict[N, Dict[Tuple[D, D], 'EdgeFunction[L]']] = {}
    
    def add_function(self, source_val: D, target: N, target_val: D, 
                    function: 'EdgeFunction[L]'):
        """
        Records a jump function. The source statement is implicit.
        
        Args:
            source_val: The source value
            target: The target node
            target_val: The target value
            function: The edge function
        """
        assert source_val is not None
        assert target is not None
        assert target_val is not None
        assert function is not None
        
        # We do not store the default function (all-top)
        if function.equal_to(self.all_top):
            return
        
        with self._lock:
            # Update reverse lookup
            key = (target, target_val)
            if key not in self.non_empty_reverse_lookup:
                self.non_empty_reverse_lookup[key] = {}
            self.non_empty_reverse_lookup[key][source_val] = function
            
            # Update forward lookup
            key = (source_val, target)
            if key not in self.non_empty_forward_lookup:
                self.non_empty_forward_lookup[key] = {}
            self.non_empty_forward_lookup[key][target_val] = function
            
            # Update lookup by target node
            if target not in self.non_empty_lookup_by_target_node:
                self.non_empty_lookup_by_target_node[target] = {}
            self.non_empty_lookup_by_target_node[target][(source_val, target_val)] = function
    
    def reverse_lookup(self, target: N, target_val: D) -> Dict[D, 'EdgeFunction[L]']:
        """
        Returns, for a given target statement and value all associated
        source values, and for each the associated edge function.
        The return value is a mapping from source value to function.
        
        Args:
            target: The target node
            target_val: The target value
            
        Returns:
            A dictionary mapping source values to edge functions
        """
        assert target is not None
        assert target_val is not None
        
        with self._lock:
            key = (target, target_val)
            res = self.non_empty_reverse_lookup.get(key)
            if res is None:
                return {}
            return dict(res)  # Return a copy for thread safety
    
    def forward_lookup(self, source_val: D, target: N) -> Dict[D, 'EdgeFunction[L]']:
        """
        Returns, for a given source value and target statement all
        associated target values, and for each the associated edge function.
        The return value is a mapping from target value to function.
        
        Args:
            source_val: The source value
            target: The target node
            
        Returns:
            A dictionary mapping target values to edge functions
        """
        assert source_val is not None
        assert target is not None
        
        with self._lock:
            key = (source_val, target)
            res = self.non_empty_forward_lookup.get(key)
            if res is None:
                return {}
            return dict(res)  # Return a copy for thread safety
    
    def lookup_by_target(self, target: N) -> Set[Tuple[D, D, 'EdgeFunction[L]']]:
        """
        Returns for a given target statement all jump function records with this target.
        The return value is a set of records of the form (sourceVal, targetVal, edgeFunction).
        
        Args:
            target: The target node
            
        Returns:
            A set of tuples containing (source_val, target_val, edge_function)
        """
        assert target is not None
        
        with self._lock:
            table = self.non_empty_lookup_by_target_node.get(target)
            if table is None:
                return set()
            
            res = set()
            for (source_val, target_val), func in table.items():
                res.add((source_val, target_val, func))
            return res
    
    def remove_function(self, source_val: D, target: N, target_val: D) -> bool:
        """
        Removes a jump function. The source statement is implicit.
        
        Args:
            source_val: The source value
            target: The target node
            target_val: The target value
            
        Returns:
            True if the function has actually been removed. False if it was not
            there anyway.
        """
        assert source_val is not None
        assert target is not None
        assert target_val is not None
        
        with self._lock:
            # Remove from reverse lookup
            key = (target, target_val)
            source_val_to_func = self.non_empty_reverse_lookup.get(key)
            if source_val_to_func is None:
                return False
            if source_val not in source_val_to_func:
                return False
            del source_val_to_func[source_val]
            if not source_val_to_func:
                del self.non_empty_reverse_lookup[key]
            
            # Remove from forward lookup
            key = (source_val, target)
            target_val_to_func = self.non_empty_forward_lookup.get(key)
            if target_val_to_func is None:
                return False
            if target_val not in target_val_to_func:
                return False
            del target_val_to_func[target_val]
            if not target_val_to_func:
                del self.non_empty_forward_lookup[key]
            
            # Remove from lookup by target node
            table = self.non_empty_lookup_by_target_node.get(target)
            if table is None:
                return False
            key = (source_val, target_val)
            if key not in table:
                return False
            del table[key]
            if not table:
                del self.non_empty_lookup_by_target_node[target]
            
            return True
    
    def clear(self):
        """
        Removes all jump functions.
        """
        with self._lock:
            self.non_empty_forward_lookup.clear()
            self.non_empty_lookup_by_target_node.clear()
            self.non_empty_reverse_lookup.clear()


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
package heros.solver;

import heros.DontSynchronize;
import heros.EdgeFunction;
import heros.SynchronizedBy;
import heros.ThreadSafe;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;


import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;


/**
 * The IDE algorithm uses a list of jump functions. Instead of a list, we use a set of three
 * maps that are kept in sync. This allows for efficient indexing: the algorithm accesses
 * elements from the list through three different indices.
 */
@ThreadSafe
public class JumpFunctions<N,D,L> {
	
	//mapping from target node and value to a list of all source values and associated functions
	//where the list is implemented as a mapping from the source value to the function
	//we exclude empty default functions
	@SynchronizedBy("consistent lock on this")
	protected Table<N,D,Map<D,EdgeFunction<L>>> nonEmptyReverseLookup = HashBasedTable.create();
	
	//mapping from source value and target node to a list of all target values and associated functions
	//where the list is implemented as a mapping from the source value to the function
	//we exclude empty default functions 
	@SynchronizedBy("consistent lock on this")
	protected Table<D,N,Map<D,EdgeFunction<L>>> nonEmptyForwardLookup = HashBasedTable.create();

	//a mapping from target node to a list of triples consisting of source value,
	//target value and associated function; the triple is implemented by a table
	//we exclude empty default functions 
	@SynchronizedBy("consistent lock on this")
	protected Map<N,Table<D,D,EdgeFunction<L>>> nonEmptyLookupByTargetNode = new LinkedHashMap<N,Table<D,D,EdgeFunction<L>>>();

	@DontSynchronize("immutable")	
	private final EdgeFunction<L> allTop;
	
	public JumpFunctions(EdgeFunction<L> allTop) {
		this.allTop = allTop;
	}

	/**
	 * Records a jump function. The source statement is implicit.
	 * @see PathEdge
	 */
	public synchronized void addFunction(D sourceVal, N target, D targetVal, EdgeFunction<L> function) {
		assert sourceVal!=null;
		assert target!=null;
		assert targetVal!=null;
		assert function!=null;
		
		//we do not store the default function (all-top)
		if(function.equalTo(allTop)) return;
		
		Map<D,EdgeFunction<L>> sourceValToFunc = nonEmptyReverseLookup.get(target, targetVal);
		if(sourceValToFunc==null) {
			sourceValToFunc = new LinkedHashMap<D,EdgeFunction<L>>();
			nonEmptyReverseLookup.put(target,targetVal,sourceValToFunc);
		}
		sourceValToFunc.put(sourceVal, function);
		
		Map<D, EdgeFunction<L>> targetValToFunc = nonEmptyForwardLookup.get(sourceVal, target);
		if(targetValToFunc==null) {
			targetValToFunc = new LinkedHashMap<D,EdgeFunction<L>>();
			nonEmptyForwardLookup.put(sourceVal,target,targetValToFunc);
		}
		targetValToFunc.put(targetVal, function);

		Table<D,D,EdgeFunction<L>> table = nonEmptyLookupByTargetNode.get(target);
		if(table==null) {
			table = HashBasedTable.create();
			nonEmptyLookupByTargetNode.put(target,table);
		}
		table.put(sourceVal, targetVal, function);
	}
	
	/**
     * Returns, for a given target statement and value all associated
     * source values, and for each the associated edge function.
     * The return value is a mapping from source value to function.
	 */
	public synchronized Map<D,EdgeFunction<L>> reverseLookup(N target, D targetVal) {
		assert target!=null;
		assert targetVal!=null;
		Map<D,EdgeFunction<L>> res = nonEmptyReverseLookup.get(target,targetVal);
		if(res==null) return Collections.emptyMap();
		return res;
	}
	
	/**
	 * Returns, for a given source value and target statement all
	 * associated target values, and for each the associated edge function. 
     * The return value is a mapping from target value to function.
	 */
	public synchronized Map<D,EdgeFunction<L>> forwardLookup(D sourceVal, N target) {
		assert sourceVal!=null;
		assert target!=null;
		Map<D, EdgeFunction<L>> res = nonEmptyForwardLookup.get(sourceVal, target);
		if(res==null) return Collections.emptyMap();
		return res;
	}
	
	/**
	 * Returns for a given target statement all jump function records with this target.
	 * The return value is a set of records of the form (sourceVal,targetVal,edgeFunction).
	 */
	public synchronized Set<Cell<D,D,EdgeFunction<L>>> lookupByTarget(N target) {
		assert target!=null;
		Table<D, D, EdgeFunction<L>> table = nonEmptyLookupByTargetNode.get(target);
		if(table==null) return Collections.emptySet();
		Set<Cell<D, D, EdgeFunction<L>>> res = table.cellSet();
		if(res==null) return Collections.emptySet();
		return res;
	}
	
	/**
	 * Removes a jump function. The source statement is implicit.
	 * @see PathEdge
	 * @return True if the function has actually been removed. False if it was not
	 * there anyway.
	 */
	public synchronized boolean removeFunction(D sourceVal, N target, D targetVal) {
		assert sourceVal!=null;
		assert target!=null;
		assert targetVal!=null;
		
		Map<D,EdgeFunction<L>> sourceValToFunc = nonEmptyReverseLookup.get(target, targetVal);
		if (sourceValToFunc == null)
			return false;
		if (sourceValToFunc.remove(sourceVal) == null)
			return false;
		if (sourceValToFunc.isEmpty())
			nonEmptyReverseLookup.remove(targetVal, targetVal);
		
		Map<D, EdgeFunction<L>> targetValToFunc = nonEmptyForwardLookup.get(sourceVal, target);
		if (targetValToFunc == null)
			return false;
		if (targetValToFunc.remove(targetVal) == null)
			return false;
		if (targetValToFunc.isEmpty())
			nonEmptyForwardLookup.remove(sourceVal, target);

		Table<D,D,EdgeFunction<L>> table = nonEmptyLookupByTargetNode.get(target);
		if (table == null)
			return false;
		if (table.remove(sourceVal, targetVal) == null)
			return false;
		if (table.isEmpty())
			nonEmptyLookupByTargetNode.remove(target);
		
		return true;
	}

	/**
	 * Removes all jump functions
	 */
	public synchronized void clear() {
		this.nonEmptyForwardLookup.clear();
		this.nonEmptyLookupByTargetNode.clear();
		this.nonEmptyReverseLookup.clear();
	}

}
\n"""