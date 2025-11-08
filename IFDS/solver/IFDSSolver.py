import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, namedtuple
from Core.IFDSTabulationProblem import IFDSTabulationProblem
from .IDESolver import IDESolver
from .EdgeIdentity import EdgeIdentity
from typing import TypeVar, Generic, Dict, Set, Callable, Any, Optional, Collection

# --- Type Variables (Similar to Java Generics) ---
N = TypeVar('N')  # Nodes in CFG
D = TypeVar('D')  # Data-flow facts
M = TypeVar('M')  # Methods
V = TypeVar('V')  # Values
I = TypeVar('I')  # InterproceduralCFG

# --- Named Tuples for Data Structures ---
PathEdge = namedtuple("PathEdge", ["source_val", "target_node", "target_val"])
Pair = namedtuple("Pair", ["o1", "o2"]) # Helper for (N, D)

class BinaryDomain: 
    """IFDS의 V 타입 정의"""
    TOP = 'TOP'
    BOTTOM = 'BOTTOM'


# --- Simplified Stubs for Core Heros Interfaces ---
class MeetLattice:
    def bottom_element(self): return "BOTTOM"
    def top_element(self): return "TOP"
    def meet(self, v1, v2): return v1 # Simplified: just return v1

class EdgeFunction:
    def compose_with(self, other: 'EdgeFunction') -> 'EdgeFunction': return self
    def meet_with(self, other: 'EdgeFunction') -> 'EdgeFunction': return self
    def compute_target(self, source_val: V) -> V: return source_val
    def equal_to(self, other: 'EdgeFunction') -> bool: return self is other

class EdgeIdentity(EdgeFunction):
    pass

class FlowFunction:
    def compute_targets(self, source_val: D) -> Set[D]: return {source_val}

class InterproceduralCFG(Generic[N, M]):
    # ... placeholder methods ...
    def is_call_stmt(self, n: N) -> bool: return False
    def is_exit_stmt(self, n: N) -> bool: return False
    def get_succs_of(self, n: N) -> Collection[N]: return set()
    def get_callees_of_call_at(self, n: N) -> Collection[M]: return set()
    def get_start_points_of(self, m: M) -> Collection[N]: return set()
    def get_return_sites_of_call_at(self, n: N) -> Collection[N]: return set()
    def get_method_of(self, n: N) -> M: return "METHOD"

# --- Main Solver Classes ---

class JumpFunctions(Generic[N, D, V]):
    """
    Simplified equivalent of Java's JumpFunctions Table, storing D -> (D -> EdgeFunction)
    Structure: {N: {D_source: {D_target: EdgeFunction}}}
    """
    def __init__(self, all_top: EdgeFunction[V]):
        self.jump_map: Dict[N, Dict[D, Dict[D, EdgeFunction[V]]]] = defaultdict(lambda: defaultdict(dict))
        self.all_top = all_top
        self.lock = threading.Lock()

    def add_function(self, d_source: D, n_target: N, d_target: D, f: EdgeFunction[V]):
        with self.lock:
            self.jump_map[n_target][d_source][d_target] = f

    def forward_lookup(self, d_source: D, n_target: N) -> Dict[D, EdgeFunction[V]]:
        """Lookup by source D and target N. Returns {D_target: EdgeFunction}"""
        with self.lock:
            # We need to restructure the data on lookup for efficiency if needed
            # For simplicity, let's assume we iterate over all sources for a given N
            results = {}
            for d_s, targets in self.jump_map[n_target].items():
                 if d_s == d_source:
                    results.update(targets)
            return results

    def reverse_lookup(self, n_target: N, d_target: D) -> Dict[D, EdgeFunction[V]]:
        """Lookup by target N and target D. Returns {D_source: EdgeFunction}"""
        with self.lock:
            results = {}
            for d_source, target_funcs in self.jump_map[n_target].items():
                if d_target in target_funcs:
                    results[d_source] = target_funcs[d_target]
            return results


class IDESolver(Generic[N, D, M, V, I]):
    
    def __init__(self, problem: Any): # Replace Any with IDETabulationProblem
        # Configuration and Components
        self.zero_value: D = problem.zeroValue()
        self.icfg: I = problem.interproceduralCFG()
        self.flow_functions = problem.flowFunctions()
        self.edge_functions = problem.edgeFunctions()
        self.initial_seeds: Dict[N, Set[D]] = problem.initialSeeds()
        self.value_lattice: MeetLattice[V] = problem.meetLattice()
        self.all_top: EdgeFunction[V] = problem.allTopFunction()
        self.num_threads: int = max(1, problem.numThreads())
        self.compute_values: bool = problem.computeValues()
        self.follow_returns_past_seeds: bool = problem.followReturnsPastSeeds()

        # Shared/Synchronized Data Structures
        self.jump_fn: JumpFunctions[N, D, V] = JumpFunctions(self.all_top)
        self.val: Dict[N, Dict[D, V]] = defaultdict(dict) # Store final values
        self.incoming: Dict[N, Dict[D, Dict[N, Set[D]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.end_summary: Dict[N, Dict[D, Dict[N, Dict[D, EdgeFunction[V]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.unbalanced_ret_sites: Set[N] = set()
        
        # Locks for shared data
        self.val_lock = threading.Lock()
        self.incoming_lock = threading.Lock()
        self.end_summary_lock = threading.Lock()
        
        # Executor
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=self.num_threads)
        self.active_tasks = threading.Semaphore(0) # Acts like a counter for active tasks
        self.lock = threading.Lock() # Lock for active_tasks counter

        # Stats (simplified)
        self.propagation_count: int = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # --- Helper Methods for Concurrency ---
    
    def increment_active_tasks(self):
        with self.lock:
            self.active_tasks.release()

    def decrement_active_tasks(self):
        with self.lock:
            # We don't actually need to acquire here, release/acquire in await_completion() handles it
            pass

    def run_executor_and_await_completion(self):
        """Awaits until the counter for active tasks is zero."""
        # This is a simplification of CountingThreadPoolExecutor.awaitCompletion()
        # In a real implementation, you'd track pending futures.
        self.logger.info("Awaiting task completion...")
        while True:
            # Try to acquire all released semaphores. If successful, all tasks finished.
            if self.active_tasks.acquire(blocking=False):
                # If we acquired a semaphore, there are still active tasks or we just finished one
                self.active_tasks.release() # Put it back for the next task to consume
                threading.Event().wait(0.1) # Sleep briefly to avoid busy-waiting
            else:
                # If we fail to acquire, the counter is zero (or less than 1, which shouldn't happen)
                break
        
    def schedule_task(self, task: Callable, *args, **kwargs):
        """Submits a task to the executor and increments the active task counter."""
        if self.executor._shutdown:
             return
        
        self.increment_active_tasks()
        future = self.executor.submit(task, *args, **kwargs)
        future.add_done_callback(lambda f: self.active_tasks.acquire())
        self.propagation_count += 1
        
    # --- Main Logic ---

    def solve(self):
        """Runs the solver."""
        self.submit_initial_seeds()
        self.await_completion_compute_values_and_shutdown()

    def submit_initial_seeds(self):
        """Submits initial seeds (based on propagate)."""
        for start_point, vals in self.initial_seeds.items():
            for val in vals:
                # Initial edge from zero_value at method start point
                self.propagate(self.zero_value, start_point, val, EdgeIdentity(), None, False)
                self.jump_fn.add_function(self.zero_value, start_point, self.zero_value, EdgeIdentity())

    def await_completion_compute_values_and_shutdown(self):
        """Awaits task completion, computes values, and shuts down."""
        self.run_executor_and_await_completion()
        
        if self.compute_values:
            self.compute_values_phase()
        
        self.logger.info(f"Total Propagations: {self.propagation_count}")
        self.executor.shutdown()
        
    def propagate(self, source_val: D, target: N, target_val: D, f: EdgeFunction[V], related_call_site: Optional[N], is_unbalanced_return: bool):
        """Updates jump function and schedules new edge processing if function changes (Lines 38-51)."""
        
        jump_fn_e = self.jump_fn.reverse_lookup(target, target_val).get(source_val, self.all_top)
        f_prime = jump_fn_e.meet_with(f)
        new_function = not f_prime.equal_to(jump_fn_e)
        
        if new_function:
            self.jump_fn.add_function(source_val, target, target_val, f_prime)
            edge = PathEdge(source_val, target, target_val)
            self.schedule_task(self.process_edge, edge)
            
            self.logger.debug(f"EDGE: <{self.icfg.get_method_of(target)},{source_val}> -> <{target},{target_val}> - {f_prime}")

    def process_edge(self, edge: PathEdge[N, D]):
        """Decides how to process an edge based on the target node type (call, exit, normal)."""
        n = edge.target_node
        if self.icfg.is_call_stmt(n):
            self.process_call(edge)
        else:
            if self.icfg.is_exit_stmt(n):
                self.process_exit(edge)
            
            for m in self.icfg.get_succs_of(n):
                self.process_normal_flow(edge)
                break # Simplified: only process normal flow once per node

    def process_call(self, edge: PathEdge[N, D]):
        """Handles call edges (Lines 13-20)."""
        d1 = edge.source_val
        n = edge.target_node
        d2 = edge.target_val
        f = self.jump_fn.forward_lookup(d1, n).get(d2, self.all_top)
        
        return_site_ns = self.icfg.get_return_sites_of_call_at(n)
        callees = self.icfg.get_callees_of_call_at(n)
        
        for s_called_proc_n in callees:
            call_func = self.flow_functions.getCallFlowFunction(n, s_called_proc_n)
            res = call_func.compute_targets(d2)
            
            for s_p in self.icfg.get_start_points_of(s_called_proc_n):
                for d3 in res:
                    # Line 15: Initial self-loop
                    self.propagate(d3, s_p, d3, EdgeIdentity(), n, False)
                    
                    # Line 15.1: Register incoming call
                    with self.incoming_lock:
                        self.incoming[s_p][d3][n].add(d2)
                        
                    # Line 15.2: For each already-queried exit value (summary)
                    for e_p, d4, f_callee_summary in self.get_end_summary_cells(s_p, d3):
                        for ret_site_n in return_site_ns:
                            # Propagate using summary
                            ret_func = self.flow_functions.getReturnFlowFunction(n, s_called_proc_n, e_p, ret_site_n)
                            returned_facts = ret_func.compute_targets(d4)
                            
                            for d5 in returned_facts:
                                # Compute composed function
                                f4 = self.edge_functions.getCallEdgeFunction(n, d2, s_called_proc_n, d3)
                                f5 = self.edge_functions.getReturnEdgeFunction(n, s_called_proc_n, e_p, d4, ret_site_n, d5)
                                f_prime = f4.compose_with(f_callee_summary).compose_with(f5)
                                d5_restored_ctx = self.restore_context_on_returned_fact(n, d2, d5)
                                self.propagate(d1, ret_site_n, d5_restored_ctx, f.compose_with(f_prime), n, False)

        # Lines 17-19: Call-to-Return flows
        for return_site_n in return_site_ns:
            ctr_func = self.flow_functions.getCallToReturnFlowFunction(n, return_site_n)
            return_facts = ctr_func.compute_targets(d2)
            for d3 in return_facts:
                edge_fn_e = self.edge_functions.getCallToReturnEdgeFunction(n, d2, return_site_n, d3)
                self.propagate(d1, return_site_n, d3, f.compose_with(edge_fn_e), n, False)

    def process_exit(self, edge: PathEdge[N, D]):
        """Handles exit edges (Lines 21-32)."""
        n = edge.target_node
        f = self.jump_fn.forward_lookup(edge.source_val, n).get(edge.target_val, self.all_top)
        method_that_needs_summary = self.icfg.getMethodOf(n)
        d1 = edge.source_val
        d2 = edge.target_val
        
        start_points = self.icfg.get_start_points_of(method_that_needs_summary)
        inc = defaultdict(set)
        
        for s_p in start_points:
            # Line 21.1: Register end-summary
            with self.end_summary_lock:
                self.add_end_summary(s_p, d1, n, d2, f)
            
            # Copy incoming to avoid concurrent modification
            with self.incoming_lock:
                if s_p in self.incoming and d1 in self.incoming[s_p]:
                    for c, d_set in self.incoming[s_p][d1].items():
                        inc[c].update(d_set)

        # Lines 22-24: For each incoming call edge already processed
        for c, d_four_set in inc.items():
            for ret_site_c in self.icfg.get_return_sites_of_call_at(c):
                ret_func = self.flow_functions.getReturnFlowFunction(c, method_that_needs_summary, n, ret_site_c)
                
                for d4 in d_four_set:
                    targets = ret_func.compute_targets(d2)
                    
                    for d5 in targets:
                        f4 = self.edge_functions.getCallEdgeFunction(c, d4, self.icfg.get_method_of(n), d1)
                        f5 = self.edge_functions.getReturnEdgeFunction(c, self.icfg.get_method_of(n), n, d2, ret_site_c, d5)
                        f_prime = f4.compose_with(f).compose_with(f5)
                        
                        # Propagate to return site using composed function
                        with self.jump_fn.lock:
                             for d3, f3 in self.jump_fn.reverse_lookup(c, d4).items():
                                if not f3.equal_to(self.all_top):
                                    d5_restored_ctx = self.restore_context_on_returned_fact(c, d4, d5)
                                    self.propagate(d3, ret_site_c, d5_restored_ctx, f3.compose_with(f_prime), c, False)
                                    
        # Unbalanced returns (if enabled)
        if self.follow_returns_past_seeds and not inc and d1 == self.zero_value:
             # Simplified: just add ret site for value computation phase
             self.unbalanced_ret_sites.add(n)

    def process_normal_flow(self, edge: PathEdge[N, D]):
        """Handles normal intra-procedural flows (Lines 33-37)."""
        d1 = edge.source_val
        n = edge.target_node
        d2 = edge.target_val
        f = self.jump_fn.forward_lookup(d1, n).get(d2, self.all_top)

        for m in self.icfg.get_succs_of(n):
            flow_func = self.flow_functions.getNormalFlowFunction(n, m)
            res = flow_func.compute_targets(d2)
            
            for d3 in res:
                f_prime = f.compose_with(self.edge_functions.getNormalEdgeFunction(n, d2, m, d3))
                self.propagate(d1, m, d3, f_prime, None, False)

    def compute_values_phase(self):
        """Phase II(i) and II(ii) - Compute final V values."""
        
        # Phase II(i): Value propagation initialization
        all_seeds = self.initial_seeds.copy()
        for unbalanced_ret_site in self.unbalanced_ret_sites:
             all_seeds.setdefault(unbalanced_ret_site, set()).add(self.zero_value)
             
        for start_point, vals in all_seeds.items():
            for val in vals:
                self.set_val(start_point, val, self.value_lattice.bottom_element())
                self.schedule_task(self.propagate_value_at_start_or_call, start_point, val)

        self.run_executor_and_await_completion()
        
        # Phase II(ii): Value computation (using the final jump functions)
        # Simplified: one big task instead of multi-threaded array processing
        self.schedule_task(self.final_value_computation)
        self.run_executor_and_await_completion()

    def propagate_value_at_start_or_call(self, n: N, d: D):
        """Processes a propagated value from a supergraph node (N, D)."""
        if self.icfg.is_start_point(n) or n in self.initial_seeds or n in self.unbalanced_ret_sites:
             # Propagate from start point to calls within method
             p = self.icfg.get_method_of(n)
             for c in self.icfg.get_calls_from_within(p):
                 for d_prime, f_prime in self.jump_fn.forward_lookup(d, c).items():
                     s_p = n # start_point is n
                     new_val = f_prime.compute_target(self.get_val(s_p, d))
                     self.propagate_value(c, d_prime, new_val)
        
        if self.icfg.is_call_stmt(n):
             # Propagate from call site to callee's start point
             for q in self.icfg.get_callees_of_call_at(n):
                 call_flow_func = self.flow_functions.getCallFlowFunction(n, q)
                 for d_prime in call_flow_func.compute_targets(d):
                     edge_fn = self.edge_functions.getCallEdgeFunction(n, d, q, d_prime)
                     for start_point in self.icfg.get_start_points_of(q):
                         new_val = edge_fn.compute_target(self.get_val(n, d))
                         self.propagate_value(start_point, d_prime, new_val)

    def propagate_value(self, n_hash_n: N, n_hash_d: D, v: V):
        """Meets the new value and schedules propagation if it changed."""
        with self.val_lock:
            val_n_hash = self.get_val(n_hash_n, n_hash_d)
            v_prime = self.value_lattice.meet(val_n_hash, v)
            if v_prime != val_n_hash:
                self.set_val(n_hash_n, n_hash_d, v_prime)
                self.schedule_task(self.propagate_value_at_start_or_call, n_hash_n, n_hash_d)

    def final_value_computation(self):
        """Phase II(ii) core computation (simplified loop over all computed jump functions)."""
        # Note: This is a highly simplified version of the multithreaded array loop
        with self.jump_fn.lock:
            for n in self.jump_fn.jump_map:
                for d_prime, target_map in self.jump_fn.jump_map[n].items():
                     if self.icfg.is_start_point(n):
                         s_p = n
                         for d, f_prime in target_map.items():
                            with self.val_lock:
                                current_val = self.get_val(n, d)
                                # Meet(current, f'(val(sP, d'))
                                new_val = self.value_lattice.meet(current_val, f_prime.compute_target(self.get_val(s_p, d_prime)))
                                self.set_val(n, d, new_val)
                                
    def get_val(self, n: N, d: D) -> V:
        """Helper to safely get value, defaulting to TOP."""
        # Note: assumes lock is held if needed outside this method
        return self.val[n].get(d, self.value_lattice.top_element())

    def set_val(self, n: N, d: D, l: V):
        """Helper to safely set value, ignoring TOP."""
        # Note: assumes lock is held if needed outside this method
        if l != self.value_lattice.top_element():
            self.val[n][d] = l
        elif d in self.val[n]:
            del self.val[n][d]

    def get_end_summary_cells(self, s_p: N, d3: D):
        """Returns end summary cells (N_exit, D_exit, EdgeFunction)."""
        with self.end_summary_lock:
            # Structure: Dict[N, Dict[D, Dict[N, Dict[D, EdgeFunction]]]]
            if s_p in self.end_summary and d3 in self.end_summary[s_p]:
                for n_exit, d_exit_map in self.end_summary[s_p][d3].items():
                    for d_exit, func in d_exit_map.items():
                        yield n_exit, d_exit, func
        
    def add_end_summary(self, s_p: N, d1: D, e_p: N, d2: D, f: EdgeFunction[V]):
        """Adds a new end summary entry."""
        # Structure: Dict[N, Dict[D, Dict[N, Dict[D, EdgeFunction]]]]
        with self.end_summary_lock:
            self.end_summary[s_p].setdefault(d1, {}).setdefault(e_p, {})[d2] = f

    def restore_context_on_returned_fact(self, call_site: N, d4: D, d5: D) -> D:
        # Placeholder for restoring context, which is common in IDE problems
        return d5

    # --- Public Results API ---
    
    def result_at(self, stmt: N, value: D) -> Optional[V]:
        """Returns the final V-type result for the given fact at the given statement."""
        # No lock needed after solve() completes
        if value == self.zero_value:
             return None
        return self.val.get(stmt, {}).get(value, None)
    
    def results_at(self, stmt: N) -> Dict[D, V]:
        """Returns the resulting environment for the given statement."""
        # No lock needed after solve() completes
        return {d: v for d, v in self.val.get(stmt, {}).items() if d != self.zero_value}

# Note: The above is a structural and conceptual translation. 
# A full, runnable version would require implementing all stubbed classes 
# (IDETabulationProblem, FlowFunctions, EdgeFunctions, InterproceduralCFG, etc.).
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

import static heros.solver.IFDSSolver.BinaryDomain.BOTTOM;
import static heros.solver.IFDSSolver.BinaryDomain.TOP;
import heros.EdgeFunction;
import heros.EdgeFunctions;
import heros.FlowFunctions;
import heros.IDETabulationProblem;
import heros.IFDSTabulationProblem;
import heros.InterproceduralCFG;
import heros.MeetLattice;
import heros.edgefunc.AllBottom;
import heros.edgefunc.AllTop;
import heros.edgefunc.EdgeIdentity;

import java.util.Map;
import java.util.Set;

/**
 * A solver for an {@link IFDSTabulationProblem}. This solver in effect uses the {@link IDESolver}
 * to solve the problem, as any IFDS problem can be intepreted as a special case of an IDE problem.
 * See Section 5.4.1 of the SRH96 paper. In effect, the IFDS problem is solved by solving an IDE
 * problem in which the environments (D to N mappings) represent the set's characteristic function.
 * 
 * @param <N> The type of nodes in the interprocedural control-flow graph. Typically {@link Unit}.
 * @param <D> The type of data-flow facts to be computed by the tabulation problem.
 * @param <M> The type of objects used to represent methods. Typically {@link SootMethod}.
 * @param <I> The type of inter-procedural control-flow graph being used.
 * @see IFDSTabulationProblem
 */
public class IFDSSolver<N,D,M,I extends InterproceduralCFG<N, M>> extends IDESolver<N,D,M,IFDSSolver.BinaryDomain,I> {

	protected static enum BinaryDomain { TOP,BOTTOM } 
	
	private final static EdgeFunction<BinaryDomain> ALL_BOTTOM = new AllBottom<BinaryDomain>(BOTTOM);
	
	/**
	 * Creates a solver for the given problem. The solver must then be started by calling
	 * {@link #solve()}.
	 */
	public IFDSSolver(final IFDSTabulationProblem<N,D,M,I> ifdsProblem) {
		super(createIDETabulationProblem(ifdsProblem));
	}

	static <N, D, M, I extends InterproceduralCFG<N, M>> IDETabulationProblem<N, D, M, BinaryDomain, I> createIDETabulationProblem(
			final IFDSTabulationProblem<N, D, M, I> ifdsProblem) {
		return new IDETabulationProblem<N,D,M,BinaryDomain,I>() {

			public FlowFunctions<N,D,M> flowFunctions() {
				return ifdsProblem.flowFunctions();
			}

			public I interproceduralCFG() {
				return ifdsProblem.interproceduralCFG();
			}

			public Map<N,Set<D>> initialSeeds() {
				return ifdsProblem.initialSeeds();
			}

			public D zeroValue() {
				return ifdsProblem.zeroValue();
			}

			public EdgeFunctions<N,D,M,BinaryDomain> edgeFunctions() {
				return new IFDSEdgeFunctions();
			}

			public MeetLattice<BinaryDomain> meetLattice() {
				return new MeetLattice<BinaryDomain>() {

					public BinaryDomain topElement() {
						return BinaryDomain.TOP;
					}

					public BinaryDomain bottomElement() {
						return BinaryDomain.BOTTOM;
					}

					public BinaryDomain meet(BinaryDomain left, BinaryDomain right) {
						if(left==TOP && right==TOP) {
							return TOP;
						} else {
							return BOTTOM;
						}
					}
				};
			}

			@Override
			public EdgeFunction<BinaryDomain> allTopFunction() {
				return new AllTop<BinaryDomain>(TOP);
			}
			
			@Override
			public boolean followReturnsPastSeeds() {
				return ifdsProblem.followReturnsPastSeeds();
			}
			
			@Override
			public boolean autoAddZero() {
				return ifdsProblem.autoAddZero();
			}
			
			@Override
			public int numThreads() {
				return ifdsProblem.numThreads();
			}
			
			@Override
			public boolean computeValues() {
				return ifdsProblem.computeValues();
			}
			
			class IFDSEdgeFunctions implements EdgeFunctions<N,D,M,BinaryDomain> {
		
				public EdgeFunction<BinaryDomain> getNormalEdgeFunction(N src,D srcNode,N tgt,D tgtNode) {
					if(srcNode==ifdsProblem.zeroValue()) return ALL_BOTTOM;
					return EdgeIdentity.v(); 
				}
		
				public EdgeFunction<BinaryDomain> getCallEdgeFunction(N callStmt,D srcNode,M destinationMethod,D destNode) {
					if(srcNode==ifdsProblem.zeroValue()) return ALL_BOTTOM;
					return EdgeIdentity.v(); 
				}
		
				public EdgeFunction<BinaryDomain> getReturnEdgeFunction(N callSite, M calleeMethod,N exitStmt,D exitNode,N returnSite,D retNode) {
					if(exitNode==ifdsProblem.zeroValue()) return ALL_BOTTOM;
					return EdgeIdentity.v(); 
				}
		
				public EdgeFunction<BinaryDomain> getCallToReturnEdgeFunction(N callStmt,D callNode,N returnSite,D returnSideNode) {
					if(callNode==ifdsProblem.zeroValue()) return ALL_BOTTOM;
					return EdgeIdentity.v(); 
				}
			}
			
			@Override
			public boolean recordEdges() {
				return ifdsProblem.recordEdges();
			}

			};
	}
	
	/**
	 * Returns the set of facts that hold at the given statement.
	 */
	public Set<D> ifdsResultsAt(N statement) {
		return resultsAt(statement).keySet();
	}

}
\n"""