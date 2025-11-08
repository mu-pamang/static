"""
Copyright (c) 2012 Eric Bodden.
Copyright (c) 2013 Tata Consultancy Services & Ecole Polytechnique de Montreal
All rights reserved. This program and the accompanying materials
are made available under the terms of the GNU Lesser Public License v2.1
which accompanies this distribution, and is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
    Marc-Andre Laverdiere-Papineau - Fixed race condition
    John Toman - Adds edge recording
"""

from typing import TypeVar, Generic, Dict, Set, Optional, Callable, Collection, Tuple, Any
from collections import defaultdict
from threading import Lock
from Core.MeetLattice import MeetLattice
from Core.EdgeFunction import EdgeFunction
from Core.FlowFunction import FlowFunction
from Core.IDETabulationProblem import IDETabulationProblem
from Core.PathEdge import PathEdge
from .JumpFunctions import JumpFunctions
from .CountingThreadPoolExecutor import CountingThreadPoolExecutor
import logging
import time

# Type variables
N = TypeVar('N')  # The type of nodes in the interprocedural control-flow graph
D = TypeVar('D')  # The type of data-flow facts to be computed by the tabulation problem
M = TypeVar('M')  # The type of objects used to represent methods
V = TypeVar('V')  # The type of values to be computed along flow edges
I = TypeVar('I')  # The type of inter-procedural control-flow graph being used

logger = logging.getLogger(__name__)

class BinaryDomain:
    """Java의 IFDSSolver.BinaryDomain enum 역할 (TOP: Taint O, BOTTOM: Taint X)"""
    TOP = 'TOP'
    BOTTOM = 'BOTTOM'

class PathEdge(Generic[N, D]):
    """Represents an edge in the exploded super graph."""
    
    def __init__(self, fact_at_source: D, target: N, fact_at_target: D):
        self._fact_at_source = fact_at_source
        self._target = target
        self._fact_at_target = fact_at_target
    
    def fact_at_source(self) -> D:
        return self._fact_at_source
    
    def get_target(self) -> N:
        return self._target
    
    def fact_at_target(self) -> D:
        return self._fact_at_target
    
    def __hash__(self):
        return hash((self._fact_at_source, self._target, self._fact_at_target))
    
    def __eq__(self, other):
        if not isinstance(other, PathEdge):
            return False
        return (self._fact_at_source == other._fact_at_source and
                self._target == other._target and
                self._fact_at_target == other._fact_at_target)


class JumpFunctions(Generic[N, D, V]):
    """Manages jump functions in the exploded super graph."""
    
    def __init__(self, all_top: 'EdgeFunction[V]'):
        self.all_top = all_top
        self._lock = Lock()
        # sourceVal -> target -> targetVal -> EdgeFunction
        self._forward_lookup: Dict[D, Dict[N, Dict[D, 'EdgeFunction[V]']]] = defaultdict(
            lambda: defaultdict(dict)
        )
        # target -> targetVal -> sourceVal -> EdgeFunction
        self._reverse_lookup: Dict[N, Dict[D, Dict[D, 'EdgeFunction[V]']]] = defaultdict(
            lambda: defaultdict(dict)
        )
    
    def add_function(self, source_val: D, target: N, target_val: D, 
                    func: 'EdgeFunction[V]'):
        """Add a jump function."""
        with self._lock:
            self._forward_lookup[source_val][target][target_val] = func
            self._reverse_lookup[target][target_val][source_val] = func
    
    def forward_lookup(self, source_val: D, target: N) -> Dict[D, 'EdgeFunction[V]']:
        """Look up functions by source value and target."""
        with self._lock:
            return dict(self._forward_lookup[source_val][target])
    
    def reverse_lookup(self, target: N, target_val: D) -> Dict[D, 'EdgeFunction[V]']:
        """Look up functions by target and target value."""
        with self._lock:
            return dict(self._reverse_lookup[target][target_val])
    
    def lookup_by_target(self, target: N) -> Set[Tuple[D, D, 'EdgeFunction[V]']]:
        """Look up all functions targeting a specific node."""
        with self._lock:
            result = set()
            for target_val, source_map in self._reverse_lookup[target].items():
                for source_val, func in source_map.items():
                    result.add((source_val, target_val, func))
            return result


class IDESolver(Generic[N, D, M, V, I]):
    """
    Solves the given IDETabulationProblem as described in the 1996 paper by Sagiv,
    Horwitz and Reps. To solve the problem, call solve(). Results can then be
    queried by using result_at() and results_at().
    
    Type Parameters:
        N: The type of nodes in the interprocedural control-flow graph.
        D: The type of data-flow facts to be computed by the tabulation problem.
        M: The type of objects used to represent methods.
        V: The type of values to be computed along flow edges.
        I: The type of inter-procedural control-flow graph being used.
    """
    
    DEBUG = logger.isEnabledFor(logging.DEBUG)
    
    def __init__(self, tabulation_problem: 'IDETabulationProblem[N, D, M, V, I]'):
        """
        Creates a solver for the given problem.
        The solver must then be started by calling solve().
        """
        self.zero_value = tabulation_problem.zero_value()
        self.icfg = tabulation_problem.interprocedural_cfg()
        
        # Flow functions
        self.flow_functions = tabulation_problem.flow_functions()
        self.edge_functions = tabulation_problem.edge_functions()
        
        self.initial_seeds = tabulation_problem.initial_seeds()
        self.unbalanced_ret_sites: Set[N] = set()
        self.value_lattice = tabulation_problem.meet_lattice()
        self.all_top = tabulation_problem.all_top_function()
        self.jump_fn = JumpFunctions[N, D, V](self.all_top)
        self.follow_returns_past_seeds = tabulation_problem.follow_returns_past_seeds()
        self.num_threads = max(1, tabulation_problem.num_threads())
        self.compute_values = tabulation_problem.compute_values()
        self.record_edges = tabulation_problem.record_edges()
        
        # Edge recording tables
        self.computed_intra_p_edges: Dict[Tuple[N, N], Dict[D, Set[D]]] = {}
        self.computed_inter_p_edges: Dict[Tuple[N, N], Dict[D, Set[D]]] = {}
        self.computed_intra_p_edges_lock = Lock()
        self.computed_inter_p_edges_lock = Lock()
        
        # End summaries and incoming edges
        self.end_summary: Dict[Tuple[N, D], Dict[Tuple[N, D], 'EdgeFunction[V]']] = defaultdict(dict)
        self.incoming: Dict[Tuple[N, D], Dict[N, Set[D]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.incoming_lock = Lock()
        
        # Value computation
        self.val: Dict[Tuple[N, D], V] = {}
        self.val_lock = Lock()
        
        # Statistics
        self.flow_function_application_count = 0
        self.flow_function_construction_count = 0
        self.propagation_count = 0
        self.duration_flow_function_construction = 0
        self.duration_flow_function_application = 0
        
        # Executor
        self.executor = self.get_executor()
    
    def solve(self):
        """Runs the solver on the configured problem. This can take some time."""
        self.submit_initial_seeds()
        self.await_completion_compute_values_and_shutdown()
    
    def submit_initial_seeds(self):
        """
        Schedules the processing of initial seeds, initiating the analysis.
        """
        from .EdgeIdentity import EdgeIdentity

        
        for start_point, seeds in self.initial_seeds.items():
            for val in seeds:
                self.propagate(self.zero_value, start_point, val, 
                             EdgeIdentity.v(), None, False)
            self.jump_fn.add_function(self.zero_value, start_point, 
                                     self.zero_value, EdgeIdentity.v())
    
    def await_completion_compute_values_and_shutdown(self):
        """
        Awaits the completion of the exploded super graph. When complete, computes
        result values, shuts down the executor and returns.
        """
        before = time.time()
        self.run_executor_and_await_completion()
        self.duration_flow_function_construction = time.time() - before
        
        if self.compute_values:
            before = time.time()
            self.compute_values_internal()
            self.duration_flow_function_application = time.time() - before
        
        if self.DEBUG:
            self.print_stats()
        
        self.executor.shutdown()
        self.run_executor_and_await_completion()
    
    def run_executor_and_await_completion(self):
        """Runs execution, re-throwing exceptions that might be thrown during its execution."""
        try:
            self.executor.await_completion()
        except InterruptedError as e:
            logger.error("Interrupted during execution", exc_info=True)
        
        exception = self.executor.get_exception()
        if exception is not None:
            raise RuntimeError("There were exceptions during IDE analysis. Exiting.") from exception
    
    def schedule_edge_processing(self, edge: PathEdge[N, D]):
        """Dispatch the processing of a given edge."""
        if self.executor._shutdown:
            return
        self.executor.submit(self.path_edge_processing_task, edge)
        self.propagation_count += 1
    
    def schedule_value_processing(self, n_and_d: Tuple[N, D]):
        """Dispatch the processing of a given value."""
        if self.executor._shutdown:
            return
        self.executor.submit(self.value_propagation_task, n_and_d)
    
    def schedule_value_computation_task(self, values: list, num: int):
        """Dispatch the computation of a given value."""
        if self.executor._shutdown:
            return
        self.executor.submit(self.value_computation_task, values, num)
    
    def save_edges(self, source_node: N, sink_stmt: N, source_val: D, 
                  dest_vals: Set[D], inter_p: bool):
        """Save edges for visualization/debugging."""
        if not self.record_edges:
            return
        
        lock = self.computed_inter_p_edges_lock if inter_p else self.computed_intra_p_edges_lock
        tgt_map = self.computed_inter_p_edges if inter_p else self.computed_intra_p_edges
        
        with lock:
            key = (source_node, sink_stmt)
            if key not in tgt_map:
                tgt_map[key] = {}
            tgt_map[key][source_val] = set(dest_vals)
    
    def process_call(self, edge: PathEdge[N, D]):
        """
        Lines 13-20 of the algorithm; processing a call site in the caller's context.
        """
        from .EdgeIdentity import EdgeIdentity
        
        d1 = edge.fact_at_source()
        n = edge.get_target()
        d2 = edge.fact_at_target()
        f = self.jump_function(edge)
        
        logger.debug(f"Processing call to {n}")
        
        return_site_ns = self.icfg.get_return_sites_of_call_at(n)
        callees = self.icfg.get_callees_of_call_at(n)
        
        for s_called_proc_n in callees:
            function = self.flow_functions.get_call_flow_function(n, s_called_proc_n)
            self.flow_function_construction_count += 1
            res = self.compute_call_flow_function(function, d1, d2)
            
            start_points_of = self.icfg.get_start_points_of(s_called_proc_n)
            for s_p in start_points_of:
                self.save_edges(n, s_p, d2, res, True)
                
                for d3 in res:
                    self.propagate(d3, s_p, d3, EdgeIdentity.v(), n, False)
                    
                    with self.incoming_lock:
                        self.add_incoming(s_p, d3, n, d2)
                        end_summ = set(self.end_summary.get((s_p, d3), {}).items())
                    
                    for (e_p, d4), f_callee_summary in end_summ:
                        for ret_site_n in return_site_ns:
                            ret_function = self.flow_functions.get_return_flow_function(
                                n, s_called_proc_n, e_p, ret_site_n
                            )
                            self.flow_function_construction_count += 1
                            returned_facts = self.compute_return_flow_function(
                                ret_function, d3, d4, n, {d2}
                            )
                            self.save_edges(e_p, ret_site_n, d4, returned_facts, True)
                            
                            for d5 in returned_facts:
                                f4 = self.edge_functions.get_call_edge_function(
                                    n, d2, s_called_proc_n, d3
                                )
                                f5 = self.edge_functions.get_return_edge_function(
                                    n, s_called_proc_n, e_p, d4, ret_site_n, d5
                                )
                                f_prime = f4.compose_with(f_callee_summary).compose_with(f5)
                                d5_restored_ctx = self.restore_context_on_returned_fact(n, d2, d5)
                                self.propagate(d1, ret_site_n, d5_restored_ctx, 
                                             f.compose_with(f_prime), n, False)
        
        # Process call-to-return flows
        for return_site_n in return_site_ns:
            call_to_return_flow_function = self.flow_functions.get_call_to_return_flow_function(
                n, return_site_n
            )
            self.flow_function_construction_count += 1
            return_facts = self.compute_call_to_return_flow_function(
                call_to_return_flow_function, d1, d2
            )
            self.save_edges(n, return_site_n, d2, return_facts, False)
            
            for d3 in return_facts:
                edge_fn_e = self.edge_functions.get_call_to_return_edge_function(
                    n, d2, return_site_n, d3
                )
                self.propagate(d1, return_site_n, d3, f.compose_with(edge_fn_e), n, False)
    
    def compute_call_flow_function(self, call_flow_function: 'FlowFunction[D]', 
                                  d1: D, d2: D) -> Set[D]:
        """Computes the call flow function for the given call-site abstraction."""
        return call_flow_function.compute_targets(d2)
    
    def compute_call_to_return_flow_function(self, call_to_return_flow_function: 'FlowFunction[D]',
                                            d1: D, d2: D) -> Set[D]:
        """Computes the call-to-return flow function."""
        return call_to_return_flow_function.compute_targets(d2)
    
    def process_exit(self, edge: PathEdge[N, D]):
        """
        Lines 21-32 of the algorithm.
        Stores callee-side summaries.
        """
        n = edge.get_target()
        f = self.jump_function(edge)
        method_that_needs_summary = self.icfg.get_method_of(n)
        
        d1 = edge.fact_at_source()
        d2 = edge.fact_at_target()
        
        start_points_of = self.icfg.get_start_points_of(method_that_needs_summary)
        inc = {}
        
        for s_p in start_points_of:
            with self.incoming_lock:
                self.add_end_summary(s_p, d1, n, d2, f)
                for entry_key, entry_val in self.incoming_helper(d1, s_p).items():
                    inc[entry_key] = set(entry_val)
        
        for c, d4_set in inc.items():
            for ret_site_c in self.icfg.get_return_sites_of_call_at(c):
                ret_function = self.flow_functions.get_return_flow_function(
                    c, method_that_needs_summary, n, ret_site_c
                )
                self.flow_function_construction_count += 1
                
                for d4 in d4_set:
                    targets = self.compute_return_flow_function(
                        ret_function, d1, d2, c, d4_set
                    )
                    self.save_edges(n, ret_site_c, d2, targets, True)
                    
                    for d5 in targets:
                        f4 = self.edge_functions.get_call_edge_function(
                            c, d4, self.icfg.get_method_of(n), d1
                        )
                        f5 = self.edge_functions.get_return_edge_function(
                            c, self.icfg.get_method_of(n), n, d2, ret_site_c, d5
                        )
                        f_prime = f4.compose_with(f).compose_with(f5)
                        
                        for d3, f3 in self.jump_fn.reverse_lookup(c, d4).items():
                            if not f3.equal_to(self.all_top):
                                d5_restored_ctx = self.restore_context_on_returned_fact(c, d4, d5)
                                self.propagate(d3, ret_site_c, d5_restored_ctx,
                                             f3.compose_with(f_prime), c, False)
        
        # Handle unbalanced returns
        if self.follow_returns_past_seeds and not inc and d1 == self.zero_value:
            callers = self.icfg.get_callers_of(method_that_needs_summary)
            for c in callers:
                for ret_site_c in self.icfg.get_return_sites_of_call_at(c):
                    ret_function = self.flow_functions.get_return_flow_function(
                        c, method_that_needs_summary, n, ret_site_c
                    )
                    self.flow_function_construction_count += 1
                    targets = self.compute_return_flow_function(
                        ret_function, d1, d2, c, {self.zero_value}
                    )
                    self.save_edges(n, ret_site_c, d2, targets, True)
                    
                    for d5 in targets:
                        f5 = self.edge_functions.get_return_edge_function(
                            c, self.icfg.get_method_of(n), n, d2, ret_site_c, d5
                        )
                        self.propagate_unbalanced_return_flow(
                            ret_site_c, d5, f.compose_with(f5), c
                        )
                        self.unbalanced_ret_sites.add(ret_site_c)
            
            if not callers:
                ret_function = self.flow_functions.get_return_flow_function(
                    None, method_that_needs_summary, n, None
                )
                self.flow_function_construction_count += 1
                ret_function.compute_targets(d2)
    
    def propagate_unbalanced_return_flow(self, ret_site_c: N, target_val: D,
                                        edge_function: 'EdgeFunction[V]', 
                                        related_call_site: N):
        """Propagate unbalanced return flow."""
        self.propagate(self.zero_value, ret_site_c, target_val, 
                      edge_function, related_call_site, True)
    
    def restore_context_on_returned_fact(self, call_site: N, d4: D, d5: D) -> D:
        """
        Transfer knowledge from the calling edge to the returning edge.
        """
        if hasattr(d5, 'set_calling_context'):
            d5.set_calling_context(d4)
        return d5
    
    def compute_return_flow_function(self, ret_function: 'FlowFunction[D]',
                                    d1: D, d2: D, call_site: N, 
                                    caller_side_ds: Set[D]) -> Set[D]:
        """Computes the return flow function."""
        return ret_function.compute_targets(d2)
    
    def process_normal_flow(self, edge: PathEdge[N, D]):
        """
        Lines 33-37 of the algorithm.
        Simply propagate normal, intra-procedural flows.
        """
        d1 = edge.fact_at_source()
        n = edge.get_target()
        d2 = edge.fact_at_target()
        
        f = self.jump_function(edge)
        for m in self.icfg.get_succs_of(n):
            flow_function = self.flow_functions.get_normal_flow_function(n, m)
            self.flow_function_construction_count += 1
            res = self.compute_normal_flow_function(flow_function, d1, d2)
            self.save_edges(n, m, d2, res, False)
            
            for d3 in res:
                f_prime = f.compose_with(
                    self.edge_functions.get_normal_edge_function(n, d2, m, d3)
                )
                self.propagate(d1, m, d3, f_prime, None, False)
    
    def compute_normal_flow_function(self, flow_function: 'FlowFunction[D]',
                                    d1: D, d2: D) -> Set[D]:
        """Computes the normal flow function."""
        return flow_function.compute_targets(d2)
    
    def propagate(self, source_val: D, target: N, target_val: D,
                 f: 'EdgeFunction[V]', related_call_site: Optional[N],
                 is_unbalanced_return: bool):
        """
        Propagates the flow further down the exploded super graph.
        """
        jump_fn_e = self.jump_fn.reverse_lookup(target, target_val).get(source_val)
        if jump_fn_e is None:
            jump_fn_e = self.all_top
        
        f_prime = jump_fn_e.meet_with(f)
        new_function = not f_prime.equal_to(jump_fn_e)
        
        if new_function:
            self.jump_fn.add_function(source_val, target, target_val, f_prime)
            edge = PathEdge(source_val, target, target_val)
            self.schedule_edge_processing(edge)
            
            if target_val != self.zero_value:
                logger.debug(f"{self.get_debug_name()} - EDGE: <{self.icfg.get_method_of(target)},{source_val}> -> <{target},{target_val}> - {f_prime}")
    
    def compute_values_internal(self):
        """Computes the final values for edge functions."""
        logger.debug("Computing the final values for the edge functions")
        
        all_seeds = dict(self.initial_seeds)
        for unbalanced_ret_site in self.unbalanced_ret_sites:
            if unbalanced_ret_site not in all_seeds:
                all_seeds[unbalanced_ret_site] = set()
            all_seeds[unbalanced_ret_site].add(self.zero_value)
        
        for start_point, vals in all_seeds.items():
            for val in vals:
                self.set_val(start_point, val, self.value_lattice.bottom_element())
                self.schedule_value_processing((start_point, val))
        
        logger.debug("Computed the final values of the edge functions")
        
        try:
            self.executor.await_completion()
        except InterruptedError as e:
            logger.error("Interrupted", exc_info=True)
        
        # Phase II(ii)
        all_non_call_start_nodes = list(self.icfg.all_non_call_start_nodes())
        for t in range(self.num_threads):
            self.schedule_value_computation_task(all_non_call_start_nodes, t)
        
        try:
            self.executor.await_completion()
        except InterruptedError as e:
            logger.error("Interrupted", exc_info=True)
    
    def jump_function(self, edge: PathEdge[N, D]) -> 'EdgeFunction[V]':
        """Get the jump function for an edge."""
        function = self.jump_fn.forward_lookup(
            edge.fact_at_source(), edge.get_target()
        ).get(edge.fact_at_target())
        if function is None:
            return self.all_top
        return function
    
    def add_end_summary(self, s_p: N, d1: D, e_p: N, d2: D, f: 'EdgeFunction[V]'):
        """Add an end summary."""
        key = (s_p, d1)
        self.end_summary[key][(e_p, d2)] = f
    
    def incoming_helper(self, d1: D, s_p: N) -> Dict[N, Set[D]]:
        """Get incoming edges."""
        return self.incoming.get((s_p, d1), {})
    
    def add_incoming(self, s_p: N, d3: D, n: N, d2: D):
        """Add an incoming edge."""
        key = (s_p, d3)
        self.incoming[key][n].add(d2)
    
    def val(self, n: N, d: D) -> V:
        """Get the value at a node."""
        with self.val_lock:
            result = self.val.get((n, d))
        if result is None:
            return self.value_lattice.top_element()
        return result
    
    def set_val(self, n: N, d: D, l: V):
        """Set the value at a node."""
        with self.val_lock:
            if l == self.value_lattice.top_element():
                self.val.pop((n, d), None)
            else:
                self.val[(n, d)] = l
        logger.debug(f"VALUE: {self.icfg.get_method_of(n)} {n} {d} {l}")
    
    def result_at(self, stmt: N, value: D) -> Optional[V]:
        """Returns the V-type result for the given value at the given statement."""
        return self.val.get((stmt, value))
    
    def results_at(self, stmt: N) -> Dict[D, V]:
        """Returns the resulting environment for the given statement."""
        result = {}
        for (n, d), v in self.val.items():
            if n == stmt and d != self.zero_value:
                result[d] = v
        return result
    
    def get_executor(self) -> 'CountingThreadPoolExecutor':
        """Factory method for this solver's thread-pool executor."""
        from .CountingThreadPoolExecutor import CountingThreadPoolExecutor
        from queue import Queue
        return CountingThreadPoolExecutor(
            self.num_threads, self.num_threads * 2, 30, 'seconds', Queue()
        )
    
    def get_debug_name(self) -> str:
        """Returns a String used to identify the output of this solver in debug mode."""
        return ""
    
    def print_stats(self):
        """Print statistics about the solver execution."""
        logger.info(f"Flow function construction count: {self.flow_function_construction_count}")
        logger.info(f"Flow function application count: {self.flow_function_application_count}")
        logger.info(f"Propagation count: {self.propagation_count}")
    
    # Task methods
    def path_edge_processing_task(self, edge: PathEdge[N, D]):
        """Process a path edge."""
        if self.icfg.is_call_stmt(edge.get_target()):
            self.process_call(edge)
        else:
            if self.icfg.is_exit_stmt(edge.get_target()):
                self.process_exit(edge)
            if self.icfg.get_succs_of(edge.get_target()):
                self.process_normal_flow(edge)
    
    def value_propagation_task(self, n_and_d: Tuple[N, D]):
        """Process value propagation."""
        n, d = n_and_d
        if (self.icfg.is_start_point(n) or 
            n in self.initial_seeds or 
            n in self.unbalanced_ret_sites):
            self.propagate_value_at_start(n_and_d, n)
        if self.icfg.is_call_stmt(n):
            self.propagate_value_at_call(n_and_d, n)
    
    def value_computation_task(self, values: list, num: int):
        """Compute values for a section of nodes."""
        section_size = len(values) // self.num_threads + self.num_threads
        start = section_size * num
        end = min(section_size * (num + 1), len(values))
        
        for i in range(start, end):
            n = values[i]
            for s_p in self.icfg.get_start_points_of(self.icfg.get_method_of(n)):
                lookup_by_target = self.jump_fn.lookup_by_target(n)
                for d_prime, d, f_prime in lookup_by_target:
                    with self.val_lock:
                        current_val = self.val(n, d)
                        new_val = f_prime.compute_target(self.val(s_p, d_prime))
                        self.set_val(n, d, self.value_lattice.meet(current_val, new_val))
                    self.flow_function_application_count += 1
    
    def propagate_value_at_start(self, n_and_d: Tuple[N, D], n: N):
        """Propagate value at start point."""
        _, d = n_and_d
        p = self.icfg.get_method_of(n)
        for c in self.icfg.get_calls_from_within(p):
            entries = self.jump_fn.forward_lookup(d, c)
            for d_prime, f_prime in entries.items():
                s_p = n
                self.propagate_value(c, d_prime, f_prime.compute_target(self.val(s_p, d)))
                self.flow_function_application_count += 1
    
    def propagate_value_at_call(self, n_and_d: Tuple[N, D], n: N):
        """Propagate value at call site."""
        _, d = n_and_d
        for q in self.icfg.get_callees_of_call_at(n):
            call_flow_function = self.flow_functions.get_call_flow_function(n, q)
            self.flow_function_construction_count += 1
            for d_prime in call_flow_function.compute_targets(d):
                edge_fn = self.edge_functions.get_call_edge_function(n, d, q, d_prime)
                for start_point in self.icfg.get_start_points_of(q):
                    self.propagate_value(start_point, d_prime, 
                                       edge_fn.compute_target(self.val(n, d)))
                    self.flow_function_application_count += 1
    
    def propagate_value(self, n: N, d: D, v: V):
        """Propagate a value."""
        with self.val_lock:
            val_n = self.val(n, d)
            v_prime = self.value_lattice.meet(val_n, v)
            if v_prime != val_n:
                self.set_val(n, d, v_prime)
                self.schedule_value_processing((n, d))


"""\n===== ORIGINAL JAVA FOR REFERENCE =====\n/*******************************************************************************
 * Copyright (c) 2012 Eric Bodden.
 * Copyright (c) 2013 Tata Consultancy Services & Ecole Polytechnique de Montreal
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Eric Bodden - initial API and implementation
 *     Marc-Andre Laverdiere-Papineau - Fixed race condition
 *     John Toman - Adds edge recording
 ******************************************************************************/
package heros.solver;


import heros.DontSynchronize;
import heros.EdgeFunction;
import heros.EdgeFunctionCache;
import heros.EdgeFunctions;
import heros.FlowFunction;
import heros.FlowFunctionCache;
import heros.FlowFunctions;
import heros.IDETabulationProblem;
import heros.InterproceduralCFG;
import heros.MeetLattice;
import heros.SynchronizedBy;
import heros.ZeroedFlowFunctions;
import heros.edgefunc.EdgeIdentity;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Predicate;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Maps;
import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;


/**
 * Solves the given {@link IDETabulationProblem} as described in the 1996 paper by Sagiv,
 * Horwitz and Reps. To solve the problem, call {@link #solve()}. Results can then be
 * queried by using {@link #resultAt(Object, Object)} and {@link #resultsAt(Object)}.
 * 
 * Note that this solver and its data structures internally use mostly {@link java.util.LinkedHashSet}s
 * instead of normal HashSet to fix the iteration order as much as possible. This
 * is to produce, as much as possible, reproducible benchmarking results. We have found
 * that the iteration order can matter a lot in terms of speed.
 *
 * @param <N> The type of nodes in the interprocedural control-flow graph. 
 * @param <D> The type of data-flow facts to be computed by the tabulation problem.
 * @param <M> The type of objects used to represent methods.
 * @param <V> The type of values to be computed along flow edges.
 * @param <I> The type of inter-procedural control-flow graph being used.
 */
public class IDESolver<N,D,M,V,I extends InterproceduralCFG<N, M>> {
	
	public static CacheBuilder<Object, Object> DEFAULT_CACHE_BUILDER = CacheBuilder.newBuilder().concurrencyLevel(Runtime.getRuntime().availableProcessors()).initialCapacity(10000).softValues();
	
    protected static final Logger logger = LoggerFactory.getLogger(IDESolver.class);

    @SynchronizedBy("consistent lock on field")
    protected Table<N,N,Map<D,Set<D>>> computedIntraPEdges = HashBasedTable.create();
    
    @SynchronizedBy("consistent lock on field")
    protected Table<N,N,Map<D,Set<D>>> computedInterPEdges = HashBasedTable.create();

    //enable with -Dorg.slf4j.simpleLogger.defaultLogLevel=trace
    public static boolean DEBUG = logger.isDebugEnabled();

	protected CountingThreadPoolExecutor executor;
	
	@DontSynchronize("only used by single thread")
	protected int numThreads;
	
	@SynchronizedBy("thread safe data structure, consistent locking when used")
	protected final JumpFunctions<N,D,V> jumpFn;
	
	@SynchronizedBy("thread safe data structure, only modified internally")
	protected final I icfg;
	
	//stores summaries that were queried before they were computed
	//see CC 2010 paper by Naeem, Lhotak and Rodriguez
	@SynchronizedBy("consistent lock on 'incoming'")
	protected final Table<N,D,Table<N,D,EdgeFunction<V>>> endSummary = HashBasedTable.create();

	//edges going along calls
	//see CC 2010 paper by Naeem, Lhotak and Rodriguez
	@SynchronizedBy("consistent lock on field")
	protected final Table<N,D,Map<N,Set<D>>> incoming = HashBasedTable.create();

	//stores the return sites (inside callers) to which we have unbalanced returns
	//if followReturnPastSeeds is enabled
	@SynchronizedBy("use of ConcurrentHashMap")
	protected final Set<N> unbalancedRetSites;

	@DontSynchronize("stateless")
	protected final FlowFunctions<N, D, M> flowFunctions;

	@DontSynchronize("stateless")
	protected final EdgeFunctions<N,D,M,V> edgeFunctions;

	@DontSynchronize("only used by single thread")
	protected final Map<N,Set<D>> initialSeeds;

	@DontSynchronize("stateless")
	protected final MeetLattice<V> valueLattice;
	
	@DontSynchronize("stateless")
	protected final EdgeFunction<V> allTop;

	@SynchronizedBy("consistent lock on field")
	protected final Table<N,D,V> val = HashBasedTable.create();	
	
	@DontSynchronize("benign races")
	public long flowFunctionApplicationCount;

	@DontSynchronize("benign races")
	public long flowFunctionConstructionCount;
	
	@DontSynchronize("benign races")
	public long propagationCount;
	
	@DontSynchronize("benign races")
	public long durationFlowFunctionConstruction;
	
	@DontSynchronize("benign races")
	public long durationFlowFunctionApplication;

	@DontSynchronize("stateless")
	protected final D zeroValue;
	
	@DontSynchronize("readOnly")
	protected final FlowFunctionCache<N,D,M> ffCache; 

	@DontSynchronize("readOnly")
	protected final EdgeFunctionCache<N,D,M,V> efCache;

	@DontSynchronize("readOnly")
	protected final boolean followReturnsPastSeeds;

	@DontSynchronize("readOnly")
	protected final boolean computeValues;

	private boolean recordEdges;

	/**
	 * Creates a solver for the given problem, which caches flow functions and edge functions.
	 * The solver must then be started by calling {@link #solve()}.
	 */
	public IDESolver(IDETabulationProblem<N,D,M,V,I> tabulationProblem) {
		this(tabulationProblem, DEFAULT_CACHE_BUILDER, DEFAULT_CACHE_BUILDER);
	}

	/**
	 * Creates a solver for the given problem, constructing caches with the given {@link CacheBuilder}. The solver must then be started by calling
	 * {@link #solve()}.
	 * @param flowFunctionCacheBuilder A valid {@link CacheBuilder} or <code>null</code> if no caching is to be used for flow functions.
	 * @param edgeFunctionCacheBuilder A valid {@link CacheBuilder} or <code>null</code> if no caching is to be used for edge functions.
	 */
	public IDESolver(IDETabulationProblem<N,D,M,V,I> tabulationProblem, @SuppressWarnings("rawtypes") CacheBuilder flowFunctionCacheBuilder, @SuppressWarnings("rawtypes") CacheBuilder edgeFunctionCacheBuilder) {
		if(logger.isDebugEnabled()) {
			if(flowFunctionCacheBuilder != null)
				flowFunctionCacheBuilder = flowFunctionCacheBuilder.recordStats();
			if(edgeFunctionCacheBuilder != null)
				edgeFunctionCacheBuilder = edgeFunctionCacheBuilder.recordStats();
		}
		this.zeroValue = tabulationProblem.zeroValue();
		this.icfg = tabulationProblem.interproceduralCFG();		
		FlowFunctions<N, D, M> flowFunctions = tabulationProblem.autoAddZero() ?
				new ZeroedFlowFunctions<N,D,M>(tabulationProblem.flowFunctions(), tabulationProblem.zeroValue()) : tabulationProblem.flowFunctions(); 
		EdgeFunctions<N, D, M, V> edgeFunctions = tabulationProblem.edgeFunctions();
		if(flowFunctionCacheBuilder!=null) {
			ffCache = new FlowFunctionCache<N,D,M>(flowFunctions, flowFunctionCacheBuilder);
			flowFunctions = ffCache;
		} else {
			ffCache = null;
		}
		if(edgeFunctionCacheBuilder!=null) {
			efCache = new EdgeFunctionCache<N,D,M,V>(edgeFunctions, edgeFunctionCacheBuilder);
			edgeFunctions = efCache;
		} else {
			efCache = null;
		}
		this.flowFunctions = flowFunctions;
		this.edgeFunctions = edgeFunctions;
		this.initialSeeds = tabulationProblem.initialSeeds();
		this.unbalancedRetSites = Collections.synchronizedSet(new LinkedHashSet<N>());
		this.valueLattice = tabulationProblem.meetLattice();
		this.allTop = tabulationProblem.allTopFunction();
		this.jumpFn = new JumpFunctions<N,D,V>(allTop);
		this.followReturnsPastSeeds = tabulationProblem.followReturnsPastSeeds();
		this.numThreads = Math.max(1,tabulationProblem.numThreads());
		this.computeValues = tabulationProblem.computeValues();
		this.executor = getExecutor();
		this.recordEdges = tabulationProblem.recordEdges();
	}

	/**
	 * Runs the solver on the configured problem. This can take some time.
	 */
	public void solve() {		
		submitInitialSeeds();
		awaitCompletionComputeValuesAndShutdown();
	}

	/**
	 * Schedules the processing of initial seeds, initiating the analysis.
	 * Clients should only call this methods if performing synchronization on
	 * their own. Normally, {@link #solve()} should be called instead.
	 */
	protected void submitInitialSeeds() {
		for(Entry<N, Set<D>> seed: initialSeeds.entrySet()) {
			N startPoint = seed.getKey();
			for(D val: seed.getValue()) {
				propagate(zeroValue, startPoint, val, EdgeIdentity.<V>v(), null, false);
			}
			jumpFn.addFunction(zeroValue, startPoint, zeroValue, EdgeIdentity.<V>v());
		}
	}

	/**
	 * Awaits the completion of the exploded super graph. When complete, computes result values,
	 * shuts down the executor and returns.
	 */
	protected void awaitCompletionComputeValuesAndShutdown() {
		{
			final long before = System.currentTimeMillis();
			//run executor and await termination of tasks
			runExecutorAndAwaitCompletion();
			durationFlowFunctionConstruction = System.currentTimeMillis() - before;
		}
		if(computeValues) {
			final long before = System.currentTimeMillis();
			computeValues();
			durationFlowFunctionApplication = System.currentTimeMillis() - before;
		}
		if(logger.isDebugEnabled())
			printStats();
		
		//ask executor to shut down;
		//this will cause new submissions to the executor to be rejected,
		//but at this point all tasks should have completed anyway
		executor.shutdown();
		//similarly here: we await termination, but this should happen instantaneously,
		//as all tasks should have completed
		runExecutorAndAwaitCompletion();
	}

	/**
	 * Runs execution, re-throwing exceptions that might be thrown during its execution.
	 */
	private void runExecutorAndAwaitCompletion() {
		try {
			executor.awaitCompletion();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		Throwable exception = executor.getException();
		if(exception!=null) {
			throw new RuntimeException("There were exceptions during IDE analysis. Exiting.",exception);
		}
	}

    /**
     * Dispatch the processing of a given edge. It may be executed in a different thread.
     * @param edge the edge to process
     */
    protected void scheduleEdgeProcessing(PathEdge<N,D> edge){
    	// If the executor has been killed, there is little point
    	// in submitting new tasks
    	if (executor.isTerminating())
    		return;
    	executor.execute(new PathEdgeProcessingTask(edge));
    	propagationCount++;
    }
	
    /**
     * Dispatch the processing of a given value. It may be executed in a different thread.
     * @param vpt
     */
    private void scheduleValueProcessing(ValuePropagationTask vpt){
    	// If the executor has been killed, there is little point
    	// in submitting new tasks
    	if (executor.isTerminating())
    		return;
    	executor.execute(vpt);
    }
  
    /**
     * Dispatch the computation of a given value. It may be executed in a different thread.
     * @param task
     */
	private void scheduleValueComputationTask(ValueComputationTask task) {
    	// If the executor has been killed, there is little point
    	// in submitting new tasks
    	if (executor.isTerminating())
    		return;
		executor.execute(task);
	}
	
	private void saveEdges(N sourceNode, N sinkStmt, D sourceVal, Set<D> destVals, boolean interP) {
		if(!this.recordEdges) {
			return;
		}
		Table<N, N, Map<D, Set<D>>> tgtMap = interP ? computedInterPEdges : computedIntraPEdges;
		synchronized (tgtMap) {
			Map<D,Set<D>> map = tgtMap.get(sourceNode, sinkStmt);
			if(map == null) {
				map = new LinkedHashMap<D, Set<D>>();
				tgtMap.put(sourceNode, sinkStmt, map);
			}
			map.put(sourceVal, new LinkedHashSet<D>(destVals));
		}
	}
	
	/**
	 * Lines 13-20 of the algorithm; processing a call site in the caller's context.
	 * 
	 * For each possible callee, registers incoming call edges.
	 * Also propagates call-to-return flows and summarized callee flows within the caller. 
	 * 
	 * @param edge an edge whose target node resembles a method call
	 */
	private void processCall(PathEdge<N,D> edge) {
		final D d1 = edge.factAtSource();
		final N n = edge.getTarget(); // a call node; line 14...

        logger.trace("Processing call to {}", n);

		final D d2 = edge.factAtTarget();
		EdgeFunction<V> f = jumpFunction(edge);
		Collection<N> returnSiteNs = icfg.getReturnSitesOfCallAt(n);
		
		//for each possible callee
		Collection<M> callees = icfg.getCalleesOfCallAt(n);
		for(M sCalledProcN: callees) { //still line 14
			
			//compute the call-flow function
			FlowFunction<D> function = flowFunctions.getCallFlowFunction(n, sCalledProcN);
			flowFunctionConstructionCount++;
			Set<D> res = computeCallFlowFunction(function, d1, d2);
			//for each callee's start point(s)
			Collection<N> startPointsOf = icfg.getStartPointsOf(sCalledProcN);
			for(N sP: startPointsOf) {
				saveEdges(n, sP, d2, res, true);
				//for each result node of the call-flow function
				for(D d3: res) {
					//create initial self-loop
					propagate(d3, sP, d3, EdgeIdentity.<V>v(), n, false); //line 15
	
					//register the fact that <sp,d3> has an incoming edge from <n,d2>
					Set<Cell<N, D, EdgeFunction<V>>> endSumm;
					synchronized (incoming) {
						//line 15.1 of Naeem/Lhotak/Rodriguez
						addIncoming(sP,d3,n,d2);
						//line 15.2, copy to avoid concurrent modification exceptions by other threads
						endSumm = new LinkedHashSet<Table.Cell<N,D,EdgeFunction<V>>>(endSummary(sP, d3));
					}
					
					//still line 15.2 of Naeem/Lhotak/Rodriguez
					//for each already-queried exit value <eP,d4> reachable from <sP,d3>,
					//create new caller-side jump functions to the return sites
					//because we have observed a potentially new incoming edge into <sP,d3>
					for(Cell<N, D, EdgeFunction<V>> entry: endSumm) {
						N eP = entry.getRowKey();
						D d4 = entry.getColumnKey();
						EdgeFunction<V> fCalleeSummary = entry.getValue();
						//for each return site
						for(N retSiteN: returnSiteNs) {
							//compute return-flow function
							FlowFunction<D> retFunction = flowFunctions.getReturnFlowFunction(n, sCalledProcN, eP, retSiteN);
							flowFunctionConstructionCount++;
							Set<D> returnedFacts = computeReturnFlowFunction(retFunction, d3, d4, n, Collections.singleton(d2));
							saveEdges(eP, retSiteN, d4, returnedFacts, true);
							//for each target value of the function
							for(D d5: returnedFacts) {
								//update the caller-side summary function
								EdgeFunction<V> f4 = edgeFunctions.getCallEdgeFunction(n, d2, sCalledProcN, d3);
								EdgeFunction<V> f5 = edgeFunctions.getReturnEdgeFunction(n, sCalledProcN, eP, d4, retSiteN, d5);
								EdgeFunction<V> fPrime = f4.composeWith(fCalleeSummary).composeWith(f5);					
								D d5_restoredCtx = restoreContextOnReturnedFact(n, d2, d5);
								propagate(d1, retSiteN, d5_restoredCtx, f.composeWith(fPrime), n, false);
							}
						}
					}
				}		
			}
		}
		//line 17-19 of Naeem/Lhotak/Rodriguez		
		//process intra-procedural flows along call-to-return flow functions
		for (N returnSiteN : returnSiteNs) {
			FlowFunction<D> callToReturnFlowFunction = flowFunctions.getCallToReturnFlowFunction(n, returnSiteN);
			flowFunctionConstructionCount++;
			Set<D> returnFacts = computeCallToReturnFlowFunction(callToReturnFlowFunction, d1, d2);
			saveEdges(n, returnSiteN, d2, returnFacts, false);
			for(D d3: returnFacts) {
				EdgeFunction<V> edgeFnE = edgeFunctions.getCallToReturnEdgeFunction(n, d2, returnSiteN, d3);
				propagate(d1, returnSiteN, d3, f.composeWith(edgeFnE), n, false);
			}
		}
	}

	/**
	 * Computes the call flow function for the given call-site abstraction
	 * @param callFlowFunction The call flow function to compute
	 * @param d1 The abstraction at the current method's start node.
	 * @param d2 The abstraction at the call site
	 * @return The set of caller-side abstractions at the callee's start node
	 */
	protected Set<D> computeCallFlowFunction
			(FlowFunction<D> callFlowFunction, D d1, D d2) {
		return callFlowFunction.computeTargets(d2);
	}

	/**
	 * Computes the call-to-return flow function for the given call-site
	 * abstraction
	 * @param callToReturnFlowFunction The call-to-return flow function to
	 * compute
	 * @param d1 The abstraction at the current method's start node.
	 * @param d2 The abstraction at the call site
	 * @return The set of caller-side abstractions at the return site
	 */
	protected Set<D> computeCallToReturnFlowFunction
			(FlowFunction<D> callToReturnFlowFunction, D d1, D d2) {
		return callToReturnFlowFunction.computeTargets(d2);
	}
	
	/**
	 * Lines 21-32 of the algorithm.
	 * 
	 * Stores callee-side summaries.
	 * Also, at the side of the caller, propagates intra-procedural flows to return sites
	 * using those newly computed summaries.
	 * 
	 * @param edge an edge whose target node resembles a method exits
	 */
	protected void processExit(PathEdge<N,D> edge) {
		final N n = edge.getTarget(); // an exit node; line 21...
		EdgeFunction<V> f = jumpFunction(edge);
		M methodThatNeedsSummary = icfg.getMethodOf(n);
		
		final D d1 = edge.factAtSource();
		final D d2 = edge.factAtTarget();
		
		//for each of the method's start points, determine incoming calls
		Collection<N> startPointsOf = icfg.getStartPointsOf(methodThatNeedsSummary);
		Map<N,Set<D>> inc = new LinkedHashMap<N,Set<D>>();
		for(N sP: startPointsOf) {
			//line 21.1 of Naeem/Lhotak/Rodriguez
			
			//register end-summary
			synchronized (incoming) {
				addEndSummary(sP, d1, n, d2, f);
				//copy to avoid concurrent modification exceptions by other threads
				for (Entry<N, Set<D>> entry : incoming(d1, sP).entrySet())
					inc.put(entry.getKey(), new LinkedHashSet<D>(entry.getValue()));
			}
		}
		
		//for each incoming call edge already processed
		//(see processCall(..))
		for (Entry<N,Set<D>> entry: inc.entrySet()) {
			//line 22
			N c = entry.getKey();
			//for each return site
			for(N retSiteC: icfg.getReturnSitesOfCallAt(c)) {
				//compute return-flow function
				FlowFunction<D> retFunction = flowFunctions.getReturnFlowFunction(c, methodThatNeedsSummary,n,retSiteC);
				flowFunctionConstructionCount++;
				//for each incoming-call value
				for(D d4: entry.getValue()) {
					Set<D> targets = computeReturnFlowFunction(retFunction, d1, d2, c, entry.getValue());
					saveEdges(n, retSiteC, d2, targets, true);
					//for each target value at the return site
					//line 23
					for(D d5: targets) {
						//compute composed function
						EdgeFunction<V> f4 = edgeFunctions.getCallEdgeFunction(c, d4, icfg.getMethodOf(n), d1);
						EdgeFunction<V> f5 = edgeFunctions.getReturnEdgeFunction(c, icfg.getMethodOf(n), n, d2, retSiteC, d5);
						EdgeFunction<V> fPrime = f4.composeWith(f).composeWith(f5);
						//for each jump function coming into the call, propagate to return site using the composed function
						synchronized (jumpFn) { // some other thread might change jumpFn on the way
							for(Map.Entry<D,EdgeFunction<V>> valAndFunc: jumpFn.reverseLookup(c,d4).entrySet()) {
								EdgeFunction<V> f3 = valAndFunc.getValue();
								if(!f3.equalTo(allTop)) {
									D d3 = valAndFunc.getKey();
									D d5_restoredCtx = restoreContextOnReturnedFact(c, d4, d5);
									propagate(d3, retSiteC, d5_restoredCtx, f3.composeWith(fPrime), c, false);
								}
							}
						}
					}
				}
			}
		}
		
		//handling for unbalanced problems where we return out of a method with a fact for which we have no incoming flow
		//note: we propagate that way only values that originate from ZERO, as conditionally generated values should only
		//be propagated into callers that have an incoming edge for this condition
		if(followReturnsPastSeeds && inc.isEmpty() && d1.equals(zeroValue)) {
			// only propagate up if we 
				Collection<N> callers = icfg.getCallersOf(methodThatNeedsSummary);
				for(N c: callers) {
					for(N retSiteC: icfg.getReturnSitesOfCallAt(c)) {
						FlowFunction<D> retFunction = flowFunctions.getReturnFlowFunction(c, methodThatNeedsSummary,n,retSiteC);
						flowFunctionConstructionCount++;
						Set<D> targets = computeReturnFlowFunction(retFunction, d1, d2, c, Collections.singleton(zeroValue));
						saveEdges(n, retSiteC, d2, targets, true);
						for(D d5: targets) {
							EdgeFunction<V> f5 = edgeFunctions.getReturnEdgeFunction(c, icfg.getMethodOf(n), n, d2, retSiteC, d5);
							propagateUnbalancedReturnFlow(retSiteC, d5, f.composeWith(f5), c);
							//register for value processing (2nd IDE phase)
							unbalancedRetSites.add(retSiteC);
						}
					}
				}
				//in cases where there are no callers, the return statement would normally not be processed at all;
				//this might be undesirable if the flow function has a side effect such as registering a taint;
				//instead we thus call the return flow function will a null caller
				if(callers.isEmpty()) {
					FlowFunction<D> retFunction = flowFunctions.getReturnFlowFunction(null, methodThatNeedsSummary,n,null);
					flowFunctionConstructionCount++;
					retFunction.computeTargets(d2);
				}
			}
		}
	
	protected void propagateUnbalancedReturnFlow(N retSiteC, D targetVal, EdgeFunction<V> edgeFunction, N relatedCallSite) {		
		propagate(zeroValue, retSiteC, targetVal, edgeFunction, relatedCallSite, true);
	}

	/**
	 * This method will be called for each incoming edge and can be used to
	 * transfer knowledge from the calling edge to the returning edge, without
	 * affecting the summary edges at the callee.
	 * @param callSite 
	 * 
	 * @param d4
	 *            Fact stored with the incoming edge, i.e., present at the
	 *            caller side
	 * @param d5
	 *            Fact that originally should be propagated to the caller.
	 * @return Fact that will be propagated to the caller.
	 */
	@SuppressWarnings("unchecked")
	protected D restoreContextOnReturnedFact(N callSite, D d4, D d5) {
		if (d5 instanceof LinkedNode) {
			((LinkedNode<D>) d5).setCallingContext(d4);
		}
		if(d5 instanceof JoinHandlingNode) {
			((JoinHandlingNode<D>) d5).setCallingContext(d4);
		}			
		return d5;
	}
	
	/**
	 * Computes the return flow function for the given set of caller-side
	 * abstractions.
	 * @param retFunction The return flow function to compute
	 * @param d1 The abstraction at the beginning of the callee
	 * @param d2 The abstraction at the exit node in the callee
	 * @param callSite The call site
	 * @param callerSideDs The abstractions at the call site
	 * @return The set of caller-side abstractions at the return site
	 */
	protected Set<D> computeReturnFlowFunction
			(FlowFunction<D> retFunction, D d1, D d2, N callSite, Set<D> callerSideDs) {
		return retFunction.computeTargets(d2);
	}

	/**
	 * Lines 33-37 of the algorithm.
	 * Simply propagate normal, intra-procedural flows.
	 * @param edge
	 */
	private void processNormalFlow(PathEdge<N,D> edge) {
		final D d1 = edge.factAtSource();
		final N n = edge.getTarget(); 
		final D d2 = edge.factAtTarget();
		
		EdgeFunction<V> f = jumpFunction(edge);
		for (N m : icfg.getSuccsOf(n)) {
			FlowFunction<D> flowFunction = flowFunctions.getNormalFlowFunction(n,m);
			flowFunctionConstructionCount++;
			Set<D> res = computeNormalFlowFunction(flowFunction, d1, d2);
			saveEdges(n, m, d2, res, false);
			for (D d3 : res) {
				EdgeFunction<V> fprime = f.composeWith(edgeFunctions.getNormalEdgeFunction(n, d2, m, d3));
				propagate(d1, m, d3, fprime, null, false); 
			}
		}
	}
	
	/**
	 * Computes the normal flow function for the given set of start and end
	 * abstractions-
	 * @param flowFunction The normal flow function to compute
	 * @param d1 The abstraction at the method's start node
	 * @param d1 The abstraction at the current node
	 * @return The set of abstractions at the successor node
	 */
	protected Set<D> computeNormalFlowFunction
			(FlowFunction<D> flowFunction, D d1, D d2) {
		return flowFunction.computeTargets(d2);
	}

	/**
	 * Propagates the flow further down the exploded super graph, merging any edge function that might
	 * already have been computed for targetVal at target. 
	 * @param sourceVal the source value of the propagated summary edge
	 * @param target the target statement
	 * @param targetVal the target value at the target statement
	 * @param f the new edge function computed from (s0,sourceVal) to (target,targetVal) 
	 * @param relatedCallSite for call and return flows the related call statement, <code>null</code> otherwise
	 *        (this value is not used within this implementation but may be useful for subclasses of {@link IDESolver}) 
	 * @param isUnbalancedReturn <code>true</code> if this edge is propagating an unbalanced return
	 *        (this value is not used within this implementation but may be useful for subclasses of {@link IDESolver}) 
	 */
	protected void propagate(D sourceVal, N target, D targetVal, EdgeFunction<V> f,
		/* deliberately exposed to clients */ N relatedCallSite,
		/* deliberately exposed to clients */ boolean isUnbalancedReturn) {
		EdgeFunction<V> jumpFnE;
		EdgeFunction<V> fPrime;
		boolean newFunction;
		synchronized (jumpFn) {
			jumpFnE = jumpFn.reverseLookup(target, targetVal).get(sourceVal);
			if(jumpFnE==null) jumpFnE = allTop; //JumpFn is initialized to all-top (see line [2] in SRH96 paper)
			fPrime = jumpFnE.meetWith(f);
			newFunction = !fPrime.equalTo(jumpFnE);
			if(newFunction) {
				jumpFn.addFunction(sourceVal, target, targetVal, fPrime);
			}
		}

		if(newFunction) {
			PathEdge<N,D> edge = new PathEdge<N,D>(sourceVal, target, targetVal);
			scheduleEdgeProcessing(edge);

            if(targetVal!=zeroValue) {
                logger.trace("{} - EDGE: <{},{}> -> <{},{}> - {}", getDebugName(), icfg.getMethodOf(target), sourceVal, target, targetVal, fPrime );
            }
		}
	}
	
	/**
	 * Computes the final values for edge functions.
	 */
	private void computeValues() {	
		//Phase II(i)
        logger.debug("Computing the final values for the edge functions");
        //add caller seeds to initial seeds in an unbalanced problem
        Map<N, Set<D>> allSeeds = new LinkedHashMap<N, Set<D>>(initialSeeds);
        for(N unbalancedRetSite: unbalancedRetSites) {
        	Set<D> seeds = allSeeds.get(unbalancedRetSite);
        	if(seeds==null) {
        		seeds = new LinkedHashSet<D>();
        		allSeeds.put(unbalancedRetSite, seeds);
        	}
        	seeds.add(zeroValue);
        }
		//do processing
		for(Entry<N, Set<D>> seed: allSeeds.entrySet()) {
			N startPoint = seed.getKey();
			for(D val: seed.getValue()) {
				setVal(startPoint, val, valueLattice.bottomElement());
				Pair<N, D> superGraphNode = new Pair<N,D>(startPoint, val); 
				scheduleValueProcessing(new ValuePropagationTask(superGraphNode));
			}
		}
		logger.debug("Computed the final values of the edge functions");
		//await termination of tasks
		try {
			executor.awaitCompletion();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		//Phase II(ii)
		//we create an array of all nodes and then dispatch fractions of this array to multiple threads
		Set<N> allNonCallStartNodes = icfg.allNonCallStartNodes();
		@SuppressWarnings("unchecked")
		N[] nonCallStartNodesArray = (N[]) new Object[allNonCallStartNodes.size()];
		int i=0;
		for (N n : allNonCallStartNodes) {
			nonCallStartNodesArray[i] = n;
			i++;
		}
		//No need to keep track of the number of tasks scheduled here, since we call shutdown
		for(int t=0;t<numThreads; t++) {
			ValueComputationTask task = new ValueComputationTask(nonCallStartNodesArray, t);
			scheduleValueComputationTask(task);
		}
		//await termination of tasks
		try {
			executor.awaitCompletion();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	private void propagateValueAtStart(Pair<N, D> nAndD, N n) {
		D d = nAndD.getO2();		
		M p = icfg.getMethodOf(n);
		for(N c: icfg.getCallsFromWithin(p)) {					
			Set<Entry<D, EdgeFunction<V>>> entries; 
			synchronized (jumpFn) {
				entries = jumpFn.forwardLookup(d,c).entrySet();
				for(Map.Entry<D,EdgeFunction<V>> dPAndFP: entries) {
					D dPrime = dPAndFP.getKey();
					EdgeFunction<V> fPrime = dPAndFP.getValue();
					N sP = n;
					propagateValue(c,dPrime,fPrime.computeTarget(val(sP,d)));
					flowFunctionApplicationCount++;
				}
			}
		}
	}
	
	private void propagateValueAtCall(Pair<N, D> nAndD, N n) {
		D d = nAndD.getO2();
		for(M q: icfg.getCalleesOfCallAt(n)) {
			FlowFunction<D> callFlowFunction = flowFunctions.getCallFlowFunction(n, q);
			flowFunctionConstructionCount++;
			for(D dPrime: callFlowFunction.computeTargets(d)) {
				EdgeFunction<V> edgeFn = edgeFunctions.getCallEdgeFunction(n, d, q, dPrime);
				for(N startPoint: icfg.getStartPointsOf(q)) {
					propagateValue(startPoint,dPrime, edgeFn.computeTarget(val(n,d)));
					flowFunctionApplicationCount++;
				}
			}
		}
	}
	
	protected V meetValueAt(N unit, D fact, V curr, V newVal) {
		return valueLattice.meet(curr, newVal);
	}
	
	private void propagateValue(N nHashN, D nHashD, V v) {
		synchronized (val) {
			V valNHash = val(nHashN, nHashD);
			V vPrime = meetValueAt(nHashN, nHashD, valNHash,v);
			if(!vPrime.equals(valNHash)) {
				setVal(nHashN, nHashD, vPrime);
				scheduleValueProcessing(new ValuePropagationTask(new Pair<N,D>(nHashN,nHashD)));
			}
		}
	}

	private V val(N nHashN, D nHashD){ 
		V l;
		synchronized (val) {
			l = val.get(nHashN, nHashD);
		}
		if(l==null) return valueLattice.topElement(); //implicitly initialized to top; see line [1] of Fig. 7 in SRH96 paper
		else return l;
	}
	
	private void setVal(N nHashN, D nHashD,V l){
		// TOP is the implicit default value which we do not need to store.
		synchronized (val) {
			if (l == valueLattice.topElement())     // do not store top values
				val.remove(nHashN, nHashD);
			else
				val.put(nHashN, nHashD,l);
		}
        logger.debug("VALUE: {} {} {} {}", icfg.getMethodOf(nHashN), nHashN, nHashD, l);
	}

	private EdgeFunction<V> jumpFunction(PathEdge<N,D> edge) {
		synchronized (jumpFn) {
			EdgeFunction<V> function = jumpFn.forwardLookup(edge.factAtSource(), edge.getTarget()).get(edge.factAtTarget());
			if(function==null) return allTop; //JumpFn initialized to all-top, see line [2] in SRH96 paper
			return function;
		}
	}

	protected Set<Cell<N, D, EdgeFunction<V>>> endSummary(N sP, D d3) {
		Table<N, D, EdgeFunction<V>> map = endSummary.get(sP, d3);
		if(map==null) return Collections.emptySet();
		return map.cellSet();
	}

	private void addEndSummary(N sP, D d1, N eP, D d2, EdgeFunction<V> f) {
		Table<N, D, EdgeFunction<V>> summaries = endSummary.get(sP, d1);
		if(summaries==null) {
			summaries = HashBasedTable.create();
			endSummary.put(sP, d1, summaries);
		}
		//note: at this point we don't need to join with a potential previous f
		//because f is a jump function, which is already properly joined
		//within propagate(..)
		summaries.put(eP,d2,f);
	}	
	
	protected Map<N, Set<D>> incoming(D d1, N sP) {
		synchronized (incoming) {
			Map<N, Set<D>> map = incoming.get(sP, d1);
			if(map==null) return Collections.emptyMap();
			return map;
		}
	}
	
	protected void addIncoming(N sP, D d3, N n, D d2) {
		synchronized (incoming) {
			Map<N, Set<D>> summaries = incoming.get(sP, d3);
			if(summaries==null) {
				summaries = new LinkedHashMap<N, Set<D>>();
				incoming.put(sP, d3, summaries);
			}
			Set<D> set = summaries.get(n);
			if(set==null) {
				set = new LinkedHashSet<D>();
				summaries.put(n,set);
			}
			set.add(d2);
		}
	}	
	
	/**
	 * Returns the V-type result for the given value at the given statement.
	 * TOP values are never returned.
	 */
	public V resultAt(N stmt, D value) {
		//no need to synchronize here as all threads are known to have terminated
		return val.get(stmt, value);
	}
	
	/**
	 * Returns the resulting environment for the given statement.
	 * The artificial zero value is automatically stripped. TOP values are
	 * never returned.
	 */
	public Map<D,V> resultsAt(N stmt) {
		//filter out the artificial zero-value
		//no need to synchronize here as all threads are known to have terminated
		return Maps.filterKeys(val.row(stmt), new com.google.common.base.Predicate<D>() {

			public boolean apply(D val) {
				return val!=zeroValue;
			}
		});
	}
	
	/**
	 * Factory method for this solver's thread-pool executor.
	 */
	protected CountingThreadPoolExecutor getExecutor() {
		return new CountingThreadPoolExecutor(this.numThreads, Integer.MAX_VALUE, 30, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
	}
	
	/**
	 * Returns a String used to identify the output of this solver in debug mode.
	 * Subclasses can overwrite this string to distinguish the output from different solvers.
	 */
	protected String getDebugName() {
		return "";
	}

	public void printStats() {
		if(logger.isDebugEnabled()) {
			if(ffCache!=null)
				ffCache.printStats();
			if(efCache!=null)
				efCache.printStats();
		} else {
			logger.info("No statistics were collected, as DEBUG is disabled.");
		}
	}
	
	private class PathEdgeProcessingTask implements Runnable {
		private final PathEdge<N,D> edge;

		public PathEdgeProcessingTask(PathEdge<N,D> edge) {
			this.edge = edge;
		}

		public void run() {
			if(icfg.isCallStmt(edge.getTarget())) {
				processCall(edge);
			} else {
				//note that some statements, such as "throw" may be
				//both an exit statement and a "normal" statement
				if(icfg.isExitStmt(edge.getTarget())) {
					processExit(edge);
				}
				if(!icfg.getSuccsOf(edge.getTarget()).isEmpty()) {
					processNormalFlow(edge);
				}
			}
		}
	}
	
	private class ValuePropagationTask implements Runnable {
		private final Pair<N, D> nAndD;

		public ValuePropagationTask(Pair<N,D> nAndD) {
			this.nAndD = nAndD;
		}

		public void run() {
			N n = nAndD.getO1();
			if(icfg.isStartPoint(n) ||
				initialSeeds.containsKey(n) ||			//our initial seeds are not necessarily method-start points but here they should be treated as such
				unbalancedRetSites.contains(n)) { 		//the same also for unbalanced return sites in an unbalanced problem
				propagateValueAtStart(nAndD, n);
			}
			if(icfg.isCallStmt(n)) {
				propagateValueAtCall(nAndD, n);
			}
		}
	}
	
	private class ValueComputationTask implements Runnable {
		private final N[] values;
		final int num;

		public ValueComputationTask(N[] values, int num) {
			this.values = values;
			this.num = num;
		}

		public void run() {
			int sectionSize = (int) Math.floor(values.length / numThreads) + numThreads;
			for(int i = sectionSize * num; i < Math.min(sectionSize * (num+1),values.length); i++) {
				N n = values[i];
				for(N sP: icfg.getStartPointsOf(icfg.getMethodOf(n))) {					
					Set<Cell<D, D, EdgeFunction<V>>> lookupByTarget;
					lookupByTarget = jumpFn.lookupByTarget(n);
					for(Cell<D, D, EdgeFunction<V>> sourceValTargetValAndFunction : lookupByTarget) {
						D dPrime = sourceValTargetValAndFunction.getRowKey();
						D d = sourceValTargetValAndFunction.getColumnKey();
						EdgeFunction<V> fPrime = sourceValTargetValAndFunction.getValue();
						synchronized (val) {
							setVal(n,d,valueLattice.meet(val(n,d),fPrime.computeTarget(val(sP,dPrime))));
						}
						flowFunctionApplicationCount++;
					}
				}
			}
		}
	}

}
\n"""