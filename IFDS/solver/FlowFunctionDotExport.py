"""
Copyright (c) 2015 Eric Bodden.
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    John Toman - initial API and implementation
"""

from typing import TypeVar, Generic, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
from .IDESolver import IDESolver
from Core.FlowFunction import FlowFunction
from Core.FlowFunction import FlowFunction
import os

# Type variables
N = TypeVar('N')  # The type of nodes in the interprocedural control-flow graph
D = TypeVar('D')  # The type of data-flow facts to be computed by the tabulation problem
M = TypeVar('M')  # The type of objects used to represent methods
I = TypeVar('I')  # The type of inter-procedural control-flow graph being used


class Pair(Generic[N, D]):
    """
    A simple pair class to hold two values together.
    """
    
    def __init__(self, first: N, second: D):
        self.first = first
        self.second = second
    
    def __hash__(self):
        return hash((self.first, self.second))
    
    def __eq__(self, other):
        if not isinstance(other, Pair):
            return False
        return self.first == other.first and self.second == other.second
    
    def __repr__(self):
        return f"Pair({self.first}, {self.second})"


class Numberer(Generic[D]):
    """
    A class to assign unique numbers to objects.
    """
    
    def __init__(self):
        self.counter = 1
        self.map: Dict[D, int] = {}
    
    def add(self, obj: D):
        """
        Add an object and assign it a number if not already present.
        
        Args:
            obj: The object to add
        """
        if obj in self.map:
            return
        self.map[obj] = self.counter
        self.counter += 1
    
    def get(self, obj: D) -> int:
        """
        Get the number assigned to an object.
        
        Args:
            obj: The object to look up
            
        Returns:
            The number assigned to this object
            
        Raises:
            ValueError: If obj is None or not found in the map
        """
        if obj is None:
            raise ValueError("Null key")
        if obj not in self.map:
            raise ValueError(f"Failed to find number for: {obj}")
        return self.map[obj]


class UnitFactTracker(Generic[N, D, M]):
    """
    Tracks units (statements) and facts for visualization purposes.
    """
    
    def __init__(self):
        self.fact_numbers: Numberer[Pair[N, D]] = Numberer()
        self.unit_numbers: Numberer[N] = Numberer()
        self.facts_for_unit: Dict[N, Set[D]] = defaultdict(set)
        self.method_to_unit: Dict[M, Set[N]] = defaultdict(set)
        self.stub_methods: Dict[M, Set[N]] = defaultdict(set)
    
    def register_fact_at_unit(self, unit: N, fact: D):
        """
        Register a fact at a specific unit.
        
        Args:
            unit: The unit (statement) where the fact holds
            fact: The data-flow fact
        """
        self.facts_for_unit[unit].add(fact)
        self.fact_numbers.add(Pair(unit, fact))
    
    def register_unit(self, method: M, unit: N):
        """
        Register a unit within a method.
        
        Args:
            method: The method containing the unit
            unit: The unit to register
        """
        self.unit_numbers.add(unit)
        self.method_to_unit[method].add(unit)
    
    def register_stub_unit(self, method: M, unit: N):
        """
        Register a stub unit (for filtered methods).
        
        Args:
            method: The method containing the stub unit
            unit: The stub unit to register
        """
        assert method not in self.method_to_unit
        self.unit_numbers.add(unit)
        self.stub_methods[method].add(unit)
    
    def get_unit_label(self, unit: N) -> str:
        """
        Get the DOT label for a unit.
        
        Args:
            unit: The unit
            
        Returns:
            A string label like "u123"
        """
        return f"u{self.unit_numbers.get(unit)}"
    
    def get_fact_label(self, unit: N, fact: D) -> str:
        """
        Get the DOT label for a fact at a specific unit.
        
        Args:
            unit: The unit where the fact holds
            fact: The fact
            
        Returns:
            A string label like "f456"
        """
        return f"f{self.fact_numbers.get(Pair(unit, fact))}"
    
    def get_edge_point(self, unit: N, fact: D) -> str:
        """
        Get the DOT edge endpoint for a fact at a unit.
        
        Args:
            unit: The unit
            fact: The fact
            
        Returns:
            A string like "u123:f456" representing the connection point
        """
        return f"{self.get_unit_label(unit)}:{self.get_fact_label(unit, fact)}"


class ItemPrinter(Generic[N, D, M]):
    """
    Interface for printing nodes, facts, and methods.
    Implementations should provide string representations for visualization.
    """
    
    def print_node(self, node: N, method: M) -> str:
        """
        Print a node (statement) in the context of a method.
        
        Args:
            node: The node to print
            method: The method containing the node
            
        Returns:
            String representation of the node
        """
        return str(node)
    
    def print_fact(self, fact: D) -> str:
        """
        Print a data-flow fact.
        
        Args:
            fact: The fact to print
            
        Returns:
            String representation of the fact
        """
        return str(fact)
    
    def print_method(self, method: M) -> str:
        """
        Print a method.
        
        Args:
            method: The method to print
            
        Returns:
            String representation of the method
        """
        return str(method)


class FlowFunctionDotExport(Generic[N, D, M, I]):
    """
    A class to dump the results of flow functions to a dot file for visualization.
    
    This class can be used for both IDE and IFDS problems that have implemented the
    SolverDebugConfiguration and overridden recordEdges() to return true.
    
    Type Parameters:
        N: The type of nodes in the interprocedural control-flow graph. Typically Unit.
        D: The type of data-flow facts to be computed by the tabulation problem.
        M: The type of objects used to represent methods. Typically SootMethod.
        I: The type of inter-procedural control-flow graph being used.
    """
    
    def __init__(self, solver: 'IDESolver[N, D, M, Any, I]',
                 printer: ItemPrinter[N, D, M],
                 method_whitelist: Optional[Set[M]] = None):
        """
        Constructor.
        
        Args:
            solver: The solver instance to dump.
            printer: The printer object to use to create the string representations of
                    the nodes, facts, and methods in the exploded super-graph.
            method_whitelist: A set of methods of type M for which the full graphs should be printed.
                            Flow functions for which both unit endpoints are not contained in a method
                            in method_whitelist are not printed. Callee/caller edges into/out of the
                            methods in the set are still printed.
        """
        self.solver = solver
        self.printer = printer
        self.method_whitelist = method_whitelist
    
    @staticmethod
    def _get_or_make_set(map_dict: Dict[Any, Set[Any]], key: Any) -> Set[Any]:
        """
        Get or create a set in a dictionary.
        
        Args:
            map_dict: The dictionary
            key: The key to look up
            
        Returns:
            The set associated with the key
        """
        if key not in map_dict:
            map_dict[key] = set()
        return map_dict[key]
    
    def _escape_label_string(self, to_escape: str) -> str:
        """
        Escape special characters for DOT format.
        
        Args:
            to_escape: The string to escape
            
        Returns:
            The escaped string
        """
        return (to_escape
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("<", "\\<")
                .replace(">", "\\>"))
    
    def _number_edges(self, edge_set: Dict[Tuple[N, N], Dict[D, Set[D]]],
                     utf: UnitFactTracker[N, D, M]):
        """
        Number all edges in the edge set for DOT output.
        
        Args:
            edge_set: The set of edges to number (represented as nested dictionaries)
            utf: The unit fact tracker to update
        """
        for (source_unit, dest_unit), flow_map in edge_set.items():
            dest_method = self.solver.icfg.get_method_of(dest_unit)
            source_method = self.solver.icfg.get_method_of(source_unit)
            
            if self._is_method_filtered(source_method) and self._is_method_filtered(dest_method):
                continue
            
            if self._is_method_filtered(dest_method):
                utf.register_stub_unit(dest_method, dest_unit)
            else:
                utf.register_unit(dest_method, dest_unit)
            
            if self._is_method_filtered(source_method):
                utf.register_stub_unit(source_method, source_unit)
            else:
                utf.register_unit(source_method, source_unit)
            
            for source_fact, dest_facts in flow_map.items():
                utf.register_fact_at_unit(source_unit, source_fact)
                for dest_fact in dest_facts:
                    utf.register_fact_at_unit(dest_unit, dest_fact)
    
    def _is_method_filtered(self, method: M) -> bool:
        """
        Check if a method is filtered (not in whitelist).
        
        Args:
            method: The method to check
            
        Returns:
            True if the method is filtered, False otherwise
        """
        return self.method_whitelist is not None and method not in self.method_whitelist
    
    def _is_node_filtered(self, node: N) -> bool:
        """
        Check if a node is filtered (its method is not in whitelist).
        
        Args:
            node: The node to check
            
        Returns:
            True if the node is filtered, False otherwise
        """
        return self._is_method_filtered(self.solver.icfg.get_method_of(node))
    
    def _print_method_units(self, units: Set[N], method: M,
                           file_handle, utf: UnitFactTracker[N, D, M]):
        """
        Print units (statements) within a method to the DOT file.
        
        Args:
            units: The units to print
            method: The method containing the units
            file_handle: The file to write to
            utf: The unit fact tracker
        """
        for method_unit in units:
            facts = utf.facts_for_unit.get(method_unit)
            unit_text = self._escape_label_string(self.printer.print_node(method_unit, method))
            
            file_handle.write(f'{utf.get_unit_label(method_unit)} [shape=record,label="{unit_text} ')
            
            if facts:  # NOTE: if the '0' fact is removed for some reason this will be None
                for fact in facts:
                    escaped_fact = self._escape_label_string(self.printer.print_fact(fact))
                    file_handle.write(f'| <{utf.get_fact_label(method_unit, fact)}> {escaped_fact}')
            
            file_handle.write('"];\n')
    
    def dump_dot_file(self, file_name: str):
        """
        Write a graph representation of the flow functions computed by the solver
        to the file indicated by file_name.
        
        Note: This method should only be called after the solver passed to this object's
        constructor has had its solve() method called.
        
        Args:
            file_name: The output file to which to write the dot representation.
            
        Raises:
            IOError: If writing to the file fails
        """
        try:
            with open(file_name, 'w') as pf:
                utf = UnitFactTracker[N, D, M]()
                
                # Number all edges
                self._number_edges(self.solver.computed_intra_p_edges, utf)
                self._number_edges(self.solver.computed_inter_p_edges, utf)
                
                # Start DOT graph
                pf.write("digraph ifds {\nnode[shape=record];\n")
                
                # Print method subgraphs
                method_counter = 0
                for method, units in utf.method_to_unit.items():
                    pf.write(f"subgraph cluster{method_counter} {{\n")
                    method_counter += 1
                    
                    self._print_method_units(units, method, pf, utf)
                    
                    # Print intraprocedural edges
                    for method_unit in units:
                        if method_unit in self.solver.computed_intra_p_edges:
                            flows = self.solver.computed_intra_p_edges[method_unit]
                            for dest_unit, flow_map in flows.items():
                                for source_fact, dest_facts in flow_map.items():
                                    for dest_fact in dest_facts:
                                        edge = (f"{utf.get_edge_point(method_unit, source_fact)} -> "
                                               f"{utf.get_edge_point(dest_unit, dest_fact)}")
                                        pf.write(f"{edge};\n")
                    
                    escaped_method = self._escape_label_string(self.printer.print_method(method))
                    pf.write(f'label="{escaped_method}";\n')
                    pf.write("}\n")
                
                # Print stub methods
                for method, units in utf.stub_methods.items():
                    pf.write(f"subgraph cluster{method_counter} {{\n")
                    method_counter += 1
                    
                    self._print_method_units(units, method, pf, utf)
                    
                    escaped_method = self._escape_label_string(self.printer.print_method(method))
                    pf.write(f'label="[STUB] {escaped_method}";\n')
                    pf.write("graph[style=dotted];\n")
                    pf.write("}\n")
                
                # Print interprocedural edges
                for (source_node, dest_node), flow_map in self.solver.computed_inter_p_edges.items():
                    if self._is_node_filtered(source_node) and self._is_node_filtered(dest_node):
                        continue
                    
                    for source_fact, dest_facts in flow_map.items():
                        for dest_fact in dest_facts:
                            pf.write(f"{utf.get_edge_point(source_node, source_fact)} -> ")
                            pf.write(f"{utf.get_edge_point(dest_node, dest_fact)}")
                            pf.write(" [style=dotted];\n")
                
                pf.write("}\n")
                
        except IOError as e:
            raise RuntimeError("Writing dot output failed") from e


# Example usage and helper functions
def export_solver_to_dot(solver: 'IDESolver', output_file: str,
                        printer: Optional[ItemPrinter] = None,
                        method_whitelist: Optional[Set] = None):
    """
    Convenience function to export a solver's results to a DOT file.
    
    Args:
        solver: The IDE solver instance
        output_file: Path to the output DOT file
        printer: Optional custom printer (uses default if None)
        method_whitelist: Optional set of methods to focus on
    """
    if printer is None:
        printer = ItemPrinter()
    
    exporter = FlowFunctionDotExport(solver, printer, method_whitelist)
    exporter.dump_dot_file(output_file)


"""\n===== ORIGINAL JAVA FOR REFERENCE =====\n/*******************************************************************************
 * Copyright (c) 2015 Eric Bodden.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     John Toman - initial API and implementation 
 ******************************************************************************/
package heros.solver;

import heros.InterproceduralCFG;
import heros.ItemPrinter;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;

/**
 * A class to dump the results of flow functions to a dot file for visualization.
 * 
 * This class can be used for both IDE and IFDS problems that have implemented the
 * {@link SolverDebugConfiguration} and overridden {@link SolverDebugConfiguration#recordEdges()}
 * to return true.
 * 
 * @param <N> The type of nodes in the interprocedural control-flow graph. Typically {@link Unit}.
 * @param <D> The type of data-flow facts to be computed by the tabulation problem.
 * @param <M> The type of objects used to represent methods. Typically {@link SootMethod}.
 * @param <I> The type of inter-procedural control-flow graph being used.
 */
public class FlowFunctionDotExport<N,D,M,I extends InterproceduralCFG<N, M>> {
	private static class Numberer<D> {
		long counter = 1;
		Map<D, Long> map = new HashMap<D, Long>();
		
		public void add(D o) {
			if(map.containsKey(o)) {
				return;
			}
			map.put(o, counter++);
		}
		public long get(D o) {
			if(o == null) {
				throw new IllegalArgumentException("Null key");
			}
			if(!map.containsKey(o)) {
				throw new IllegalArgumentException("Failed to find number for: " + o);
			}
			return map.get(o);
			
		}
	}
	private final IDESolver<N, D, M, ?, I> solver;
	private final ItemPrinter<? super N, ? super D, ? super M> printer;
	private final Set<M> methodWhitelist;
	
	/**
	 * Constructor.
	 * @param solver The solver instance to dump.
	 * @param printer The printer object to use to create the string representations of
	 * the nodes, facts, and methods in the exploded super-graph.
	 */
	public FlowFunctionDotExport(IDESolver<N, D, M, ?, I> solver, ItemPrinter<? super N, ? super D, ? super M> printer) {
		this(solver, printer, null);
	}
	
	/**
	 * Constructor.
	 * @param solver The solver instance to dump.
	 * @param printer The printer object to use to create the string representations of
	 * the nodes, facts, and methods in the exploded super-graph.
	 * @param methodWhitelist A set of methods of type M for which the full graphs should be printed.
	 * Flow functions for which both unit endpoints are not contained in a method in methodWhitelist are not printed.
	 * Callee/caller edges into/out of the methods in the set are still printed.  
	 */
	public FlowFunctionDotExport(IDESolver<N, D, M, ?, I> solver, ItemPrinter<? super N, ? super D, ? super M> printer, Set<M> methodWhitelist) {
		this.solver = solver;
		this.printer = printer;
		this.methodWhitelist = methodWhitelist;
	}
	
	private static <K,U> Set<U> getOrMakeSet(Map<K,Set<U>> map, K key) {
		if(map.containsKey(key)) {
			return map.get(key);
		}
		HashSet<U> toRet = new HashSet<U>();
		map.put(key, toRet);
		return toRet;
	}
	
	private String escapeLabelString(String toEscape) { 
		return toEscape.replace("\\", "\\\\")
				.replace("\"", "\\\"")
				.replace("<", "\\<")
				.replace(">", "\\>");
	}
	
	private class UnitFactTracker {
		private Numberer<Pair<N, D>> factNumbers = new Numberer<Pair<N, D>>();
		private Numberer<N> unitNumbers = new Numberer<N>();
		private Map<N, Set<D>> factsForUnit = new HashMap<N, Set<D>>();
		private Map<M, Set<N>> methodToUnit = new HashMap<M, Set<N>>();
		
		private Map<M, Set<N>> stubMethods = new HashMap<M, Set<N>>();
		
		public void registerFactAtUnit(N unit, D fact) {
			getOrMakeSet(factsForUnit, unit).add(fact);
			factNumbers.add(new Pair<N, D>(unit, fact));
		}

		public void registerUnit(M method, N unit) {
			unitNumbers.add(unit);
			getOrMakeSet(methodToUnit, method).add(unit);
		}
		
		public void registerStubUnit(M method, N unit) {
			assert !methodToUnit.containsKey(method);
			unitNumbers.add(unit);
			getOrMakeSet(stubMethods, method).add(unit);
		}
		
		public String getUnitLabel(N unit) {
			return "u" + unitNumbers.get(unit);
		}
		
		public String getFactLabel(N unit, D fact) {
			return "f" + factNumbers.get(new Pair<N, D>(unit, fact));
		}
		
		public String getEdgePoint(N unit, D fact) {
			return this.getUnitLabel(unit) + ":" + this.getFactLabel(unit, fact);
		}
	}
	
	private void numberEdges(Table<N, N, Map<D, Set<D>>> edgeSet, UnitFactTracker utf) {
		for(Cell<N,N,Map<D,Set<D>>> c : edgeSet.cellSet()) {
			N sourceUnit = c.getRowKey();
			N destUnit = c.getColumnKey();
			M destMethod = solver.icfg.getMethodOf(destUnit);
			M sourceMethod = solver.icfg.getMethodOf(sourceUnit);
			if(isMethodFiltered(sourceMethod) && isMethodFiltered(destMethod)) {
				continue;
			}
			if(isMethodFiltered(destMethod)) {
				utf.registerStubUnit(destMethod, destUnit);
			} else {
				utf.registerUnit(destMethod, destUnit);
			}
			if(isMethodFiltered(sourceMethod)) {
				utf.registerStubUnit(sourceMethod, sourceUnit);
			} else {
				utf.registerUnit(sourceMethod, sourceUnit);
			}
			for(Map.Entry<D, Set<D>> entry : c.getValue().entrySet()) {
				utf.registerFactAtUnit(sourceUnit, entry.getKey());
				for(D destFact : entry.getValue()) {
					utf.registerFactAtUnit(destUnit, destFact);
				}
			}
		}
	}

	private boolean isMethodFiltered(M method) {
		return methodWhitelist != null && !methodWhitelist.contains(method);
	}
	
	private boolean isNodeFiltered(N node) {
		return isMethodFiltered(solver.icfg.getMethodOf(node));
	}
	
	private void printMethodUnits(Set<N> units, M method, PrintStream pf, UnitFactTracker utf) {
		for(N methodUnit : units) {
			Set<D> loc = utf.factsForUnit.get(methodUnit);
			String unitText = escapeLabelString(printer.printNode(methodUnit, method));
			pf.print(utf.getUnitLabel(methodUnit) + " [shape=record,label=\""+ unitText + " ");
			if(loc != null){//NOTE: if the '0' fact is removed for some reason this will be null
				for(D hl : loc) {
					pf.print("| <" + utf.getFactLabel(methodUnit, hl) + "> " + escapeLabelString(printer.printFact(hl)));
				}
			}
			pf.println("\"];");
		}
	}
	
	/**
	 * Write a graph representation of the flow functions computed by the solver
	 * to the file indicated by fileName.
	 * 
	 * <b>Note:</b> This method should only be called after 
	 * the solver passed to this object's constructor has had its {@link IDESolver#solve()}
	 * method called.  
	 * @param fileName The output file to which to write the dot representation.
	 */
	public void dumpDotFile(String fileName) {
		File f = new File(fileName);
		PrintStream pf = null;
		try {
			pf = new PrintStream(f);
			UnitFactTracker utf = new UnitFactTracker();
			
			numberEdges(solver.computedIntraPEdges, utf);
			numberEdges(solver.computedInterPEdges, utf);
			
			pf.println("digraph ifds {" +
					"node[shape=record];"
			);
			int methodCounter = 0;
			for(Map.Entry<M, Set<N>> kv : utf.methodToUnit.entrySet()) {
				Set<N> intraProc = kv.getValue();
				pf.println("subgraph cluster" + methodCounter + " {");
				methodCounter++;
				printMethodUnits(intraProc, kv.getKey(), pf, utf);
				for(N methodUnit : intraProc) {
					Map<N, Map<D, Set<D>>> flows = solver.computedIntraPEdges.row(methodUnit);
					for(Map.Entry<N, Map<D, Set<D>>> kv2 : flows.entrySet()) {
						N destUnit = kv2.getKey();
						for(Map.Entry<D, Set<D>> pointFlow : kv2.getValue().entrySet()) {
							for(D destFact : pointFlow.getValue()) {
								String edge = utf.getEdgePoint(methodUnit, pointFlow.getKey()) + " -> " + utf.getEdgePoint(destUnit, destFact);
								pf.print(edge);
								pf.println(";");
							}
						}
					}
				}
				pf.println("label=\"" + escapeLabelString(printer.printMethod(kv.getKey())) + "\";");
				pf.println("}");
			}
			for(Map.Entry<M, Set<N>> kv : utf.stubMethods.entrySet()) {
				pf.println("subgraph cluster" + methodCounter++ + " {");
				printMethodUnits(kv.getValue(), kv.getKey(), pf, utf);
				pf.println("label=\"" + escapeLabelString("[STUB] " + printer.printMethod(kv.getKey())) + "\";");
				pf.println("graph[style=dotted];");
				pf.println("}");
			}
			for(Cell<N, N, Map<D, Set<D>>> c : solver.computedInterPEdges.cellSet()) {
				if(isNodeFiltered(c.getRowKey()) && isNodeFiltered(c.getColumnKey())) {
					continue;
				}
				for(Map.Entry<D, Set<D>> kv : c.getValue().entrySet()) {
					for(D dFact : kv.getValue()) {
						pf.print(utf.getEdgePoint(c.getRowKey(), kv.getKey()));
						pf.print(" -> ");
						pf.print(utf.getEdgePoint(c.getColumnKey(), dFact));
						pf.println(" [style=dotted];");
					}
				}
			}
			pf.println("}");
		} catch (FileNotFoundException e) {	
			throw new RuntimeException("Writing dot output failed", e); 
		} finally {
			if(pf != null) {
				pf.close();
			}
		}
	}
}
\n"""