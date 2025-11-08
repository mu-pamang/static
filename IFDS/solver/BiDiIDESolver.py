"""
Copyright (c) 2012 Eric Bodden.
All rights reserved. This program and the accompanying materials
is available at
http://www.gnu.org/licenses/old-licenses/gpl-2.0.html

Contributors:
    Eric Bodden - initial API and implementation
"""

import time # CountingThreadPoolExecutor 인스턴스화 시 필요 (Java TimeUnit 대체)
import sys # For Python 3.9+ compatibility fix if needed
import collections # For Java's Collections equivalent

from typing import TypeVar, Generic, Set, Dict, Tuple, Optional, Collection, Any
from dataclasses import dataclass
from collections import defaultdict
from Core.IDETabulationProblem import EdgeFunctions, IDETabulationProblem
from .CountingThreadPoolExecutor import CountingThreadPoolExecutor # <-- 수정: 상대 경로 임포트
from Core.EdgeFunction import EdgeFunction
from Core.FlowFunctions import FlowFunctions, FlowFunction # <-- 수정: FlowFunction 임포트 추가
from threading import Lock
from .IDESolver import IDESolver # IDE Solver 본체를 가져옵니다.
import logging


# Type variables
N = TypeVar('N')  # Node type
D = TypeVar('D')  # Data-flow abstraction type
M = TypeVar('M')  # Method type
V = TypeVar('V')  # Value type
I = TypeVar('I')  # InterproceduralCFG type

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeakKey(Generic[N]):
    """
    Data structure used to identify which edges can be unpaused by a SingleDirectionSolver.
    Each SingleDirectionSolver stores its leaks using this structure.
    """
    source_stmt: N
    related_call_site: N

    def __hash__(self):
        return hash((self.source_stmt, self.related_call_site))

    def __eq__(self, other):
        if not isinstance(other, LeakKey):
            return False
        return (self.source_stmt == other.source_stmt and 
                self.related_call_site == other.related_call_site)


class PausedEdge(Generic[N, V]):
    """Represents a paused edge in the analysis."""
    
    def __init__(self, ret_site_c: N, target_val: 'AbstractionWithSourceStmt', 
                 edge_function: 'EdgeFunction[V]', related_call_site: N):
        self.ret_site_c = ret_site_c
        self.target_val = target_val
        self.edge_function = edge_function
        self.related_call_site = related_call_site


class AbstractionWithSourceStmt(Generic[D, N]):
    """
    Augmented abstraction propagated by the SingleDirectionSolver.
    Associates with the abstraction the source statement from which this fact originated.
    """
    
    def __init__(self, abstraction: D, source: Optional[N]):
        self.abstraction = abstraction
        self.source = source
    
    def get_abstraction(self) -> D:
        return self.abstraction
    
    def get_source_stmt(self) -> Optional[N]:
        return self.source
    
    def __str__(self):
        if self.source is not None:
            return f"{self.abstraction}-@-{self.source}"
        else:
            return str(self.abstraction)
    
    def __hash__(self):
        return hash((self.abstraction, self.source))
    
    def __eq__(self, other):
        if not isinstance(other, AbstractionWithSourceStmt):
            return False
        return (self.abstraction == other.abstraction and 
                self.source == other.source)

# ====================================================================
# SingleDirectionSolver (IDESolver를 상속하여 BiDi 로직 구현)
# ====================================================================

class SingleDirectionSolver(IDESolver[N, AbstractionWithSourceStmt, M, V, I], Generic[N, D, M, V, I]):
    """
    Modified IFDS solver that is capable of pausing and unpausing return-flow edges.
    """
    
    def __init__(self, ifds_problem: 'IDETabulationProblem', 
                debug_name: str, parent: 'BiDiIDESolver'):
        
        # IDESolver의 __init__ 호출 (ZeroValue, ICFG, FlowFunctions 등 초기화)
        super().__init__(ifds_problem)
        
        self.problem = ifds_problem
        self.debug_name = debug_name
        self.parent = parent
        self.other_solver: Optional['SingleDirectionSolver'] = None # <-- 타입 힌트 수정
        
        self.leaked_sources: Set[LeakKey[N]] = set()
        self.leaked_sources_lock = Lock()
        
        self.paused_path_edges: Dict[LeakKey[N], Set[PausedEdge]] = defaultdict(set)
        self.paused_edges_lock = Lock()
    
    # --- BiDi 로직 오버라이드 ---
    
    def propagate_unbalanced_return_flow(self, ret_site_c: N, 
                                        target_val: AbstractionWithSourceStmt,
                                        edge_function: 'EdgeFunction[V]', 
                                        related_call_site: N):
        """Handle unbalanced return flow."""
        source_stmt = target_val.get_source_stmt()
        leak_key = LeakKey(source_stmt, related_call_site)
        
        with self.leaked_sources_lock:
            self.leaked_sources.add(leak_key)
        
        if self.other_solver.has_leaked(leak_key):
            # If the other solver has leaked already, unpause its edges and continue
            self.other_solver.unpause_path_edges_for_source(leak_key)
            
            # IDESolver의 일반 propagate 로직을 사용 (is_unbalanced_return=True)
            super().propagate(self.zero_value, ret_site_c, target_val, edge_function, related_call_site, True)
            
        else:
            # Otherwise pause this solver's edge and don't continue
            edge = PausedEdge(ret_site_c, target_val, edge_function, related_call_site)
            
            with self.paused_edges_lock:
                self.paused_path_edges[leak_key].add(edge)
            
            # Check again if other solver has leaked in the meantime
            if self.other_solver.has_leaked(leak_key):
                with self.paused_edges_lock:
                    if edge in self.paused_path_edges[leak_key]:
                        self.paused_path_edges[leak_key].remove(edge)
                        super().propagate(self.zero_value, ret_site_c, target_val, edge_function, related_call_site, True)
            else:
                logger.debug(f" ++ PAUSE {self.debug_name}: {edge}")
    
    def propagate(self, source_val: AbstractionWithSourceStmt, target: N,
                 target_val: AbstractionWithSourceStmt, f: 'EdgeFunction[V]',
                 related_call_site: N, is_unbalanced_return: bool):
        """Propagate data-flow facts."""
        
        if is_unbalanced_return:
            # Attach target statement as new "source" statement to track
            target_val = AbstractionWithSourceStmt(
                target_val.get_abstraction(), related_call_site
            )
        
        # 부모 클래스 IDESolver의 propagate 호출
        super().propagate(source_val, target, target_val, f, 
                            related_call_site, is_unbalanced_return)

    def restore_context_on_returned_fact(self, call_site: N, 
                                        d4: AbstractionWithSourceStmt,
                                        d5: AbstractionWithSourceStmt) -> AbstractionWithSourceStmt:
        """Restore context when returning from a method call."""
        return AbstractionWithSourceStmt(d5.get_abstraction(), d4.get_source_stmt())
    
    def has_leaked(self, leak_key: LeakKey[N]) -> bool:
        """Check if this solver has tried to leak an edge from the given source."""
        with self.leaked_sources_lock:
            return leak_key in self.leaked_sources
    
    def unpause_path_edges_for_source(self, leak_key: LeakKey[N]):
        """Unpause all edges associated with the given source statement."""
        with self.paused_edges_lock:
            if leak_key in self.paused_path_edges:
                edges = list(self.paused_path_edges[leak_key])
                for edge in edges:
                    if edge in self.paused_path_edges[leak_key]:
                        self.paused_path_edges[leak_key].remove(edge)
                        logger.debug(f"-- UNPAUSE {self.debug_name}: {edge}")
                        
                        super().propagate(self.zero_value, edge.ret_site_c, edge.target_val, 
                                          edge.edge_function, edge.related_call_site, True)
    
    def get_executor(self) -> 'CountingThreadPoolExecutor':
        """Get shared executor."""
        # IDESolver의 get_executor를 오버라이드하여 공유 Executor 반환
        return self.parent.shared_executor # 부모 BiDiIDESolver의 shared_executor를 사용
    
    def get_debug_name(self) -> str:
        return self.debug_name
    
    def submit_initial_seeds(self):
        # BiDiIDESolver가 submitInitialSeeds()를 호출할 때 사용
        for start_point, seeds in self.initial_seeds.items():
            for val in seeds:
                super().propagate(self.zero_value, start_point, val, 
                                 self.parent.all_top, None, False)
            self.jump_fn.add_function(self.zero_value, start_point, 
                                     self.zero_value, self.parent.all_top)
    
    def solve(self):
        # BiDiIDESolver의 solve()에서 호출될 때, 이 메서드는 실제 IDESolver의 solve 로직을 실행합니다.
        super().submit_initial_seeds()
        super().await_completion_compute_values_and_shutdown()


class BiDiIDESolver(Generic[N, D, M, V, I]):
    """
    Special IFDS solver that solves the analysis problem inside out, i.e., from further 
    down the call stack to further up the call stack. This can be useful for taint 
    analysis problems that track flows in two directions.
    
    The solver is instantiated with two analyses, one computed forward and one backward.
    Both analysis problems must be unbalanced, i.e., must return True for 
    followReturnsPastSeeds().
    """
    
    def __init__(self, forward_problem: 'IDETabulationProblem[N, D, M, V, I]', 
                 backward_problem: 'IDETabulationProblem[N, D, M, V, I]'):
        if (not forward_problem.follow_returns_past_seeds() or 
            not backward_problem.follow_returns_past_seeds()):
            raise ValueError(
                "This solver is only meant for bottom-up problems, "
                "so followReturnsPastSeeds() should return True."
            )
        
        self.forward_problem = self.AugmentedTabulationProblem(forward_problem)
        self.backward_problem = self.AugmentedTabulationProblem(backward_problem)
        
        max_threads = max(1, forward_problem.num_threads())
        
        # --- 수정: CountingThreadPoolExecutor 인스턴스 생성 및 초기화 ---
        self.shared_executor = CountingThreadPoolExecutor(
            core_pool_size=max_threads, 
            maximum_pool_size=max_threads * 2, 
            keep_alive_time=30, 
            time_unit='seconds', 
            work_queue=None
        )
        # -----------------------------------------------------------------
        
        self.fw_solver: Optional[SingleDirectionSolver] = None # <-- 타입 힌트 수정
        self.bw_solver: Optional[SingleDirectionSolver] = None # <-- 타입 힌트 수정
        
    
    def solve(self):
        """Execute the bidirectional analysis."""
        self.fw_solver = self.create_single_direction_solver(self.forward_problem, "FW")
        self.bw_solver = self.create_single_direction_solver(self.backward_problem, "BW")
        
        self.fw_solver.other_solver = self.bw_solver
        self.bw_solver.other_solver = self.fw_solver
        
        # Start the backward solver
        self.bw_solver.submit_initial_seeds()
        
        # Start the forward solver and block until both solvers have completed
        self.fw_solver.solve()
    
    def create_single_direction_solver(self, problem: 'IDETabulationProblem', 
                                      debug_name: str) -> SingleDirectionSolver: # <-- 타입 힌트 수정
        """Creates a solver to be used for each single analysis direction."""
        return SingleDirectionSolver(problem, debug_name, self)
    
    class AugmentedTabulationProblem(Generic[N, D, M, V, I]):
        """
        Tabulation problem that propagates augmented abstractions where the normal 
        problem would propagate normal abstractions.
        """
        
        def __init__(self, delegate: 'IDETabulationProblem[N, D, M, V, I]'):
            self.delegate = delegate
            self.original_functions = self.delegate.flow_functions()
            self.ZERO = AbstractionWithSourceStmt(delegate.zero_value(), None)
        
        def flow_functions(self) -> 'FlowFunctions':
            """Return augmented flow functions."""
            parent = self
            
            class AugmentedFlowFunctions:
                def get_normal_flow_function(self, curr: N, succ: N):
                    def flow_func(source: AbstractionWithSourceStmt) -> Set[AbstractionWithSourceStmt]:
                        return self._copy_over_source_stmts(
                            source, 
                            parent.original_functions.get_normal_flow_function(curr, succ)
                        )
                    return flow_func
                
                def get_call_flow_function(self, call_stmt: N, destination_method: M):
                    def flow_func(source: AbstractionWithSourceStmt) -> Set[AbstractionWithSourceStmt]:
                        orig_targets = parent.original_functions.get_call_flow_function(
                            call_stmt, destination_method
                        ).compute_targets(source.get_abstraction())
                        
                        return {AbstractionWithSourceStmt(d, None) for d in orig_targets}
                    return flow_func
                
                def get_return_flow_function(self, call_site: N, callee_method: M, 
                                           exit_stmt: N, return_site: N):
                    def flow_func(source: AbstractionWithSourceStmt) -> Set[AbstractionWithSourceStmt]:
                        return self._copy_over_source_stmts(
                            source,
                            parent.original_functions.get_return_flow_function(
                                call_site, callee_method, exit_stmt, return_site
                            )
                        )
                    return flow_func
                
                def get_call_to_return_flow_function(self, call_site: N, return_site: N):
                    def flow_func(source: AbstractionWithSourceStmt) -> Set[AbstractionWithSourceStmt]:
                        return self._copy_over_source_stmts(
                            source,
                            parent.original_functions.get_call_to_return_flow_function(
                                call_site, return_site
                            )
                        )
                    return flow_func
                
                def _copy_over_source_stmts(self, source: AbstractionWithSourceStmt,
                                          original_function: FlowFunction) -> Set[AbstractionWithSourceStmt]: # 타입 힌트 수정
                    original_abstraction = source.get_abstraction()
                    orig_targets = original_function.compute_targets(original_abstraction)
                    
                    res = set()
                    for d in orig_targets:
                         res.add(AbstractionWithSourceStmt(d, source.get_source_stmt()))
                    return res
            
            return AugmentedFlowFunctions()
        
        # Delegate methods
        def follow_returns_past_seeds(self) -> bool:
            return self.delegate.follow_returns_past_seeds()
        
        def auto_add_zero(self) -> bool:
            return self.delegate.auto_add_zero()
        
        def num_threads(self) -> int:
            return self.delegate.num_threads()
        
        def compute_values(self) -> bool:
            return self.delegate.compute_values()
        
        def interprocedural_cfg(self) -> I:
            return self.delegate.interprocedural_cfg()
        
        def initial_seeds(self) -> Dict[N, Set[AbstractionWithSourceStmt]]:
            """Attach the original seed statement to the abstraction."""
            original_seeds = self.delegate.initial_seeds()
            result = {}
            
            for stmt, seeds in original_seeds.items():
                res_set = {
                    AbstractionWithSourceStmt(d, stmt) for d in seeds
                }
                result[stmt] = res_set
            
            return result
        
        def zero_value(self) -> AbstractionWithSourceStmt:
            return self.ZERO
        
        def edge_functions(self) -> 'EdgeFunctions':
            """Return augmented edge functions."""
            parent = self
            
            class AugmentedEdgeFunctions:
                def get_normal_edge_function(self, curr: N, curr_node: AbstractionWithSourceStmt,
                                           succ: N, succ_node: AbstractionWithSourceStmt):
                    return parent.delegate.edge_functions().get_normal_edge_function(
                        curr, curr_node.get_abstraction(), 
                        succ, succ_node.get_abstraction()
                    )
                
                def get_call_edge_function(self, call_stmt: N, src_node: AbstractionWithSourceStmt,
                                         destination_method: M, dest_node: AbstractionWithSourceStmt):
                    return parent.delegate.edge_functions().get_call_edge_function(
                        call_stmt, src_node.get_abstraction(),
                        destination_method, dest_node.get_abstraction()
                    )
                
                def get_return_edge_function(self, call_site: N, callee_method: M, 
                                           exit_stmt: N, exit_node: AbstractionWithSourceStmt,
                                           return_site: N, ret_node: AbstractionWithSourceStmt):
                    return parent.delegate.edge_functions().get_return_edge_function(
                        call_site, callee_method, exit_stmt, exit_node.get_abstraction(),
                        return_site, ret_node.get_abstraction()
                    )
                
                def get_call_to_return_edge_function(self, call_site: N, 
                                                   call_node: AbstractionWithSourceStmt,
                                                   return_site: N, 
                                                   return_side_node: AbstractionWithSourceStmt):
                    return parent.delegate.edge_functions().get_call_to_return_edge_function(
                        call_site, call_node.get_abstraction(),
                        return_site, return_side_node.get_abstraction()
                    )
            
            return AugmentedEdgeFunctions()
        
        def meet_lattice(self):
            return self.delegate.meet_lattice()
        
        def all_top_function(self):
            return self.delegate.all_top_function()
        
        def record_edges(self) -> bool:
            return self.delegate.record_edges()


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

import heros.EdgeFunction;
import heros.EdgeFunctions;
import heros.FlowFunction;
import heros.FlowFunctions;
import heros.IDETabulationProblem;
import heros.IFDSTabulationProblem;
import heros.InterproceduralCFG;
import heros.MeetLattice;
import heros.solver.IFDSSolver.BinaryDomain;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import com.google.common.collect.Maps;

/**
 * This is a special IFDS solver that solves the analysis problem inside out, i.e., from further down the call stack to
 * further up the call stack. This can be useful, for instance, for taint analysis problems that track flows in two directions.
 * 
 * The solver is instantiated with two analyses, one to be computed forward and one to be computed backward. Both analysis problems
 * must be unbalanced, i.e., must return <code>true</code> for {@link IFDSTabulationProblem#followReturnsPastSeeds()}.
 * The solver then executes both analyses in lockstep, i.e., when one of the analyses reaches an unbalanced return edge (signified
 * by a ZERO source value) then the solver pauses this analysis until the other analysis reaches the same unbalanced return (if ever).
 * The result is that the analyses will never diverge, i.e., will ultimately always only propagate into contexts in which both their
 * computed paths are realizable at the same time.
 * 
 * This solver requires data-flow abstractions that implement the {@link LinkedNode} interface such that data-flow values can be linked to form
 * reportable paths.  
 *
 * @param <N> see {@link IFDSSolver}
 * @param <D> A data-flow abstraction that must implement the {@link LinkedNode} interface such that data-flow values can be linked to form
 * 				reportable paths.
 * @param <M> see {@link IFDSSolver}
 * @param <I> see {@link IFDSSolver}
 */
public class BiDiIDESolver<N, D, M, V, I extends InterproceduralCFG<N, M>> {

	private final IDETabulationProblem<N, AbstractionWithSourceStmt, M,V, I> forwardProblem;
	private final IDETabulationProblem<N, AbstractionWithSourceStmt, M,V, I> backwardProblem;
	private final CountingThreadPoolExecutor sharedExecutor;
	protected SingleDirectionSolver fwSolver;
	protected SingleDirectionSolver bwSolver;

	/**
	 * Instantiates a {@link BiDiIDESolver} with the associated forward and backward problem.
	 */
	public BiDiIDESolver(IDETabulationProblem<N,D,M,V,I> forwardProblem, IDETabulationProblem<N,D,M,V,I> backwardProblem) {
		if(!forwardProblem.followReturnsPastSeeds() || !backwardProblem.followReturnsPastSeeds()) {
			throw new IllegalArgumentException("This solver is only meant for bottom-up problems, so followReturnsPastSeeds() should return true."); 
		}
		this.forwardProblem = new AugmentedTabulationProblem(forwardProblem);
		this.backwardProblem = new AugmentedTabulationProblem(backwardProblem);
		this.sharedExecutor = new CountingThreadPoolExecutor(Math.max(1,forwardProblem.numThreads()), Integer.MAX_VALUE, 30, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
	}
	
	public void solve() {		
		fwSolver = createSingleDirectionSolver(forwardProblem, "FW");
		bwSolver = createSingleDirectionSolver(backwardProblem, "BW");
		fwSolver.otherSolver = bwSolver;
		bwSolver.otherSolver = fwSolver;
		
		//start the bw solver
		bwSolver.submitInitialSeeds();
		
		//start the fw solver and block until both solvers have completed
		//(note that they both share the same executor, see below)
		//note to self: the order of the two should not matter
		fwSolver.solve();
	}
	
	/**
	 * Creates a solver to be used for each single analysis direction.
	 */
	protected SingleDirectionSolver createSingleDirectionSolver(IDETabulationProblem<N, AbstractionWithSourceStmt, M,V, I> problem, String debugName) {
		return new SingleDirectionSolver(problem, debugName);
	}
	
	private class PausedEdge {
		private N retSiteC;
		private AbstractionWithSourceStmt targetVal;
		private EdgeFunction<V> edgeFunction;
		private N relatedCallSite;
		
		public PausedEdge(N retSiteC, AbstractionWithSourceStmt targetVal, EdgeFunction<V> edgeFunction, N relatedCallSite) {
			this.retSiteC = retSiteC;
			this.targetVal = targetVal;
			this.edgeFunction = edgeFunction;
			this.relatedCallSite = relatedCallSite;
		}
	}

	/**
	 *  Data structure used to identify which edges can be unpaused by a {@link SingleDirectionSolver}. Each {@link SingleDirectionSolver} stores 
	 *  its leaks using this structure. A leak always requires a flow from some {@link #sourceStmt} (this is either the statement used as initial seed
	 *  or a call site of an unbalanced return) to a return site. This return site is always different for the forward and backward solvers,
	 *  but, the related call site of these return sites must be the same, if two entangled flows exist. 
	 *  Moreover, this structure represents the pair of such a {@link #sourceStmt} and the {@link #relatedCallSite}.
	 *
	 */
	private static class LeakKey<N> {
		private N sourceStmt;
		private N relatedCallSite;
		
		public LeakKey(N sourceStmt, N relatedCallSite) {
			this.sourceStmt = sourceStmt;
			this.relatedCallSite = relatedCallSite;
		}
		
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((relatedCallSite == null) ? 0 : relatedCallSite.hashCode());
			result = prime * result + ((sourceStmt == null) ? 0 : sourceStmt.hashCode());
			return result;
		}
		
		@SuppressWarnings("rawtypes")
		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (!(obj instanceof LeakKey))
				return false;
			LeakKey other = (LeakKey) obj;
			if (relatedCallSite == null) {
				if (other.relatedCallSite != null)
					return false;
			} else if (!relatedCallSite.equals(other.relatedCallSite))
				return false;
			if (sourceStmt == null) {
				if (other.sourceStmt != null)
					return false;
			} else if (!sourceStmt.equals(other.sourceStmt))
				return false;
			return true;
		}
	}
	
	/**
	 * This is a modified IFDS solver that is capable of pausing and unpausing return-flow edges.
	 */
	protected class SingleDirectionSolver extends IDESolver<N,AbstractionWithSourceStmt,M,V,I> {
		private final String debugName;
		private SingleDirectionSolver otherSolver;
		private Set<LeakKey<N>> leakedSources = Collections.newSetFromMap(Maps.<LeakKey<N>, Boolean>newConcurrentMap());
		private ConcurrentMap<LeakKey<N>,Set<PausedEdge>> pausedPathEdges =
				Maps.newConcurrentMap();

		public SingleDirectionSolver(IDETabulationProblem<N, AbstractionWithSourceStmt, M,V, I> ifdsProblem, String debugName) {
			super(ifdsProblem);
			this.debugName = debugName;
		}
		
		@Override
		protected void propagateUnbalancedReturnFlow(N retSiteC, AbstractionWithSourceStmt targetVal,
				EdgeFunction<V> edgeFunction, N relatedCallSite) {
			//if an edge is originating from ZERO then to us this signifies an unbalanced return edge
			N sourceStmt = targetVal.getSourceStmt();
			//we mark the fact that this solver would like to "leak" this edge to the caller
			LeakKey<N> leakKey = new LeakKey<N>(sourceStmt, relatedCallSite);
			leakedSources.add(leakKey);
			if(otherSolver.hasLeaked(leakKey)) {
				//if the other solver has leaked already then unpause its edges and continue
				otherSolver.unpausePathEdgesForSource(leakKey);
				super.propagateUnbalancedReturnFlow(retSiteC, targetVal, edgeFunction, relatedCallSite);
			} else {
				//otherwise we pause this solver's edge and don't continue
				Set<PausedEdge> newPausedEdges = 
						Collections.newSetFromMap(Maps.<PausedEdge, Boolean>newConcurrentMap()); 
				Set<PausedEdge> existingPausedEdges = pausedPathEdges.putIfAbsent(leakKey, newPausedEdges);
				if(existingPausedEdges==null)
					existingPausedEdges=newPausedEdges;
				
				PausedEdge edge = new PausedEdge(retSiteC, targetVal, edgeFunction, relatedCallSite);
				existingPausedEdges.add(edge);
				
				//if the other solver has leaked in the meantime, we have to make sure that the paused edge is unpaused
				if(otherSolver.hasLeaked(leakKey) && existingPausedEdges.remove(edge)) {
					super.propagateUnbalancedReturnFlow(retSiteC, targetVal, edgeFunction, relatedCallSite);
				}
						
                logger.debug(" ++ PAUSE {}: {}", debugName, edge);
			}
		}
		
		@Override
		protected void propagate(AbstractionWithSourceStmt sourceVal, N target, AbstractionWithSourceStmt targetVal, EdgeFunction<V> f, N relatedCallSite, boolean isUnbalancedReturn) {
			//the follwing branch will be taken only on an unbalanced return
			if(isUnbalancedReturn) {
				assert sourceVal.getSourceStmt()==null : "source value should have no statement attached";
				
				//attach target statement as new "source" statement to track
				targetVal = new AbstractionWithSourceStmt(targetVal.getAbstraction(), relatedCallSite);
				
				super.propagate(sourceVal, target, targetVal, f, relatedCallSite, isUnbalancedReturn);
			} else { 
				super.propagate(sourceVal, target, targetVal, f, relatedCallSite, isUnbalancedReturn);
			}
		}
		
		@Override
		protected AbstractionWithSourceStmt restoreContextOnReturnedFact(N callSite, AbstractionWithSourceStmt d4, AbstractionWithSourceStmt d5) {
			return new AbstractionWithSourceStmt(d5.getAbstraction(), d4.getSourceStmt());
		}
		
		/**
		 * Returns <code>true</code> if this solver has tried to leak an edge originating from the given source
		 * to its caller.
		 */
		private boolean hasLeaked(LeakKey<N> leakKey) {
			return leakedSources.contains(leakKey);
		}
		
		/**
		 * Unpauses all edges associated with the given source statement.
		 */
		private void unpausePathEdgesForSource(LeakKey<N> leakKey) {
			Set<PausedEdge> pausedEdges = pausedPathEdges.get(leakKey);
			if(pausedEdges!=null) {
				for(PausedEdge edge: pausedEdges) {
					if(pausedEdges.remove(edge)) {
						if(DEBUG)
							logger.debug("-- UNPAUSE {}: {}",debugName, edge);
						super.propagateUnbalancedReturnFlow(edge.retSiteC, edge.targetVal, edge.edgeFunction, edge.relatedCallSite);
					}
				}
			}
		}
		
		/* we share the same executor; this will cause the call to solve() above to block
		 * until both solvers have finished
		 */ 
		protected CountingThreadPoolExecutor getExecutor() {
			return sharedExecutor;
		}
		
		protected String getDebugName() {
			return debugName;
		}
	}

	/**
	 * This is an augmented abstraction propagated by the {@link SingleDirectionSolver}. It associates with the
	 * abstraction the source statement from which this fact originated. 
	 */
	public class AbstractionWithSourceStmt {

		protected final D abstraction;
		protected final N source;
		
		private AbstractionWithSourceStmt(D abstraction, N source) {
			this.abstraction = abstraction;
			this.source = source;
		}

		public D getAbstraction() {
			return abstraction;
		}
		
		public N getSourceStmt() {
			return source;
		}	
		
		@Override
		public String toString() {
			if(source!=null)
				return ""+abstraction+"-@-"+source+"";
			else
				return abstraction.toString();
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((abstraction == null) ? 0 : abstraction.hashCode());
			result = prime * result + ((source == null) ? 0 : source.hashCode());
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
			@SuppressWarnings("unchecked")
			AbstractionWithSourceStmt other = (AbstractionWithSourceStmt) obj;
			if (abstraction == null) {
				if (other.abstraction != null)
					return false;
			} else if (!abstraction.equals(other.abstraction))
				return false;
			if (source == null) {
				if (other.source != null)
					return false;
			} else if (!source.equals(other.source))
				return false;
			return true;
		}
	}
	
	/**
	 * This tabulation problem simply propagates augmented abstractions where the normal problem would propagate normal abstractions.
	 */
	private class AugmentedTabulationProblem implements IDETabulationProblem<N, AbstractionWithSourceStmt,M,V,I> {

		private final IDETabulationProblem<N,D,M,V,I> delegate;
		private final AbstractionWithSourceStmt ZERO;
		private final FlowFunctions<N, D, M> originalFunctions;
		
		public AugmentedTabulationProblem(IDETabulationProblem<N, D, M,V, I> delegate) {
			this.delegate = delegate;
			originalFunctions = this.delegate.flowFunctions();
			ZERO = new AbstractionWithSourceStmt(delegate.zeroValue(), null);
		}

		@Override
		public FlowFunctions<N, AbstractionWithSourceStmt, M> flowFunctions() {
			return new FlowFunctions<N, AbstractionWithSourceStmt, M>() {

				@Override
				public FlowFunction<AbstractionWithSourceStmt> getNormalFlowFunction(final N curr, final N succ) {
					return new FlowFunction<AbstractionWithSourceStmt>() {
						@Override
						public Set<AbstractionWithSourceStmt> computeTargets(AbstractionWithSourceStmt source) {
							return copyOverSourceStmts(source, originalFunctions.getNormalFlowFunction(curr, succ));
						}
					};
				}

				@Override
				public FlowFunction<AbstractionWithSourceStmt> getCallFlowFunction(final N callStmt, final M destinationMethod) {
					return new FlowFunction<AbstractionWithSourceStmt>() {
						@Override
						public Set<AbstractionWithSourceStmt> computeTargets(AbstractionWithSourceStmt source) {
							Set<D> origTargets = originalFunctions.getCallFlowFunction(callStmt, destinationMethod).computeTargets(
									source.getAbstraction());

							Set<AbstractionWithSourceStmt> res = new HashSet<AbstractionWithSourceStmt>();
							for (D d : origTargets) {
								res.add(new AbstractionWithSourceStmt(d, null));
							}
							return res;
						}
					};
				}

				@Override
				public FlowFunction<AbstractionWithSourceStmt> getReturnFlowFunction(final N callSite, final M calleeMethod, final N exitStmt, final N returnSite) {
					return new FlowFunction<AbstractionWithSourceStmt>() {
						@Override
						public Set<AbstractionWithSourceStmt> computeTargets(AbstractionWithSourceStmt source) {
							return copyOverSourceStmts(source, originalFunctions.getReturnFlowFunction(callSite, calleeMethod, exitStmt, returnSite));
						}
					};
				}

				@Override
				public FlowFunction<AbstractionWithSourceStmt> getCallToReturnFlowFunction(final N callSite, final N returnSite) {
					return new FlowFunction<AbstractionWithSourceStmt>() {
						@Override
						public Set<AbstractionWithSourceStmt> computeTargets(AbstractionWithSourceStmt source) {
							return copyOverSourceStmts(source, originalFunctions.getCallToReturnFlowFunction(callSite, returnSite));
						}
					};
				}
				
				private Set<AbstractionWithSourceStmt> copyOverSourceStmts(AbstractionWithSourceStmt source, FlowFunction<D> originalFunction) {
					D originalAbstraction = source.getAbstraction();
					Set<D> origTargets = originalFunction.computeTargets(originalAbstraction);
					
					Set<AbstractionWithSourceStmt> res = new HashSet<AbstractionWithSourceStmt>();
					for(D d: origTargets) {
						res.add(new AbstractionWithSourceStmt(d,source.getSourceStmt()));
					}
					return res;
				}
			};
		}
		
		//delegate methods follow

		public boolean followReturnsPastSeeds() {
			return delegate.followReturnsPastSeeds();
		}

		public boolean autoAddZero() {
			return delegate.autoAddZero();
		}

		public int numThreads() {
			return delegate.numThreads();
		}

		public boolean computeValues() {
			return delegate.computeValues();
		}

		public I interproceduralCFG() {
			return delegate.interproceduralCFG();
		}

		/* attaches the original seed statement to the abstraction
		 */
		public Map<N,Set<AbstractionWithSourceStmt>> initialSeeds() {
			Map<N, Set<D>> originalSeeds = delegate.initialSeeds();
			Map<N,Set<AbstractionWithSourceStmt>> res = new HashMap<N, Set<AbstractionWithSourceStmt>>();
			for(Entry<N, Set<D>> entry: originalSeeds.entrySet()) {
				N stmt = entry.getKey();
				Set<D> seeds = entry.getValue();
				Set<AbstractionWithSourceStmt> resSet = new HashSet<AbstractionWithSourceStmt>();
				for (D d : seeds) {
					//attach source stmt to abstraction
					resSet.add(new AbstractionWithSourceStmt(d, stmt));
				}
				res.put(stmt, resSet);
			}			
			return res;
		}

		public AbstractionWithSourceStmt zeroValue() {
			return ZERO;
		}

		@Override
		public EdgeFunctions<N, AbstractionWithSourceStmt, M, V> edgeFunctions() {
			return new EdgeFunctions<N, AbstractionWithSourceStmt, M, V>() {

				@Override
				public EdgeFunction<V> getNormalEdgeFunction(N curr, AbstractionWithSourceStmt currNode, N succ,
						AbstractionWithSourceStmt succNode) {
					return delegate.edgeFunctions().getNormalEdgeFunction(curr, currNode.getAbstraction(), succ, succNode.getAbstraction());
				}

				@Override
				public EdgeFunction<V> getCallEdgeFunction(N callStmt,AbstractionWithSourceStmt srcNode,
						M destinationMethod, AbstractionWithSourceStmt destNode) {
					return delegate.edgeFunctions().getCallEdgeFunction(callStmt, srcNode.getAbstraction(), destinationMethod, destNode.getAbstraction());
				}

				@Override
				public EdgeFunction<V> getReturnEdgeFunction(N callSite, M calleeMethod, N exitStmt,
						AbstractionWithSourceStmt exitNode, N returnSite,
						AbstractionWithSourceStmt retNode) {
					return delegate.edgeFunctions().getReturnEdgeFunction(callSite, calleeMethod, exitStmt, exitNode.getAbstraction(), returnSite, retNode.getAbstraction());
				}

				@Override
				public EdgeFunction<V> getCallToReturnEdgeFunction(N callSite, AbstractionWithSourceStmt callNode,
						N returnSite, AbstractionWithSourceStmt returnSideNode) {
					return delegate.edgeFunctions().getCallToReturnEdgeFunction(callSite, callNode.getAbstraction(), returnSite, returnSideNode.getAbstraction());
				}
			};
		}

		@Override
		public MeetLattice<V> meetLattice() {
			return delegate.meetLattice();
		}

		@Override
		public EdgeFunction<V> allTopFunction() {
			return delegate.allTopFunction();
		}

		@Override
		public boolean recordEdges() {
			return delegate.recordEdges();
		}

	}
	
}
\n"""