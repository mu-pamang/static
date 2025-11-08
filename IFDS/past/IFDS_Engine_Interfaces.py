# IFDS_Engine_Interfaces.py
# 역할: IFDS Solver의 입력으로 사용될 Flow Functions, Edge Functions, Meet Lattice의 인터페이스 정의
# 사용된 Java 파일: FlowFunction.java, FlowFunctions.java, EdgeFunction.java, EdgeFunctions.java, MeetLattice.java

from typing import TypeVar, Generic, Set, Dict, List, Collection, Optional

# 타입 변수 정의 (이전 단계와 동일)
N = TypeVar('N')  # Node (Smali Instruction)
D = TypeVar('D')  # Data-flow Fact (TaintFact)
M = TypeVar('M')  # Method (Androguard MethodAnalysis object)
V = TypeVar('V')  # Value (IFDS에서는 BinaryDomain.TOP/BOTTOM)


# ====================================================================
# 1. Flow Function (Taint 전파 규칙)
# ====================================================================

class FlowFunction(Generic[D]):
    """
    Java의 FlowFunction.java 인터페이스: D-type 값이 어떻게 전파되는지 계산
    Smali 명령어 하나하나에 대한 Taint 전파 로직(Gen/Kill)을 여기에 구현
    """
    def compute_targets(self, source: D) -> Set[D]:
        """Returns the target values reachable from the source."""
        raise NotImplementedError


class FlowFunctions(Generic[N, D, M]):
    """
    Java의 FlowFunctions.java 인터페이스: 4가지 종류의 Flow Function을 생성하는 팩토리
    """
    def get_normal_flow_function(self, curr: N, succ: N) -> FlowFunction[D]:
        """일반 명령(Normal Statement)의 흐름 계산"""
        raise NotImplementedError

    def get_call_flow_function(self, call_stmt: N, destination_method: M) -> FlowFunction[D]:
        """호출 명령(Call Statement)의 흐름 계산 (caller -> callee)"""
        raise NotImplementedError

    def get_return_flow_function(self, call_site: Optional[N], callee_method: M, exit_stmt: N, return_site: Optional[N]) -> FlowFunction[D]:
        """메서드 종료(Return/Exit Statement)의 흐름 계산 (callee -> caller)"""
        raise NotImplementedError

    def get_call_to_return_flow_function(self, call_site: N, return_site: N) -> FlowFunction[D]:
        """호출 지점부터 반환 지점까지의 흐름 계산 (Call-to-Return Edge)"""
        raise NotImplementedError


# ====================================================================
# 2. Meet Lattice (V 값 병합 규칙)
# ====================================================================

class MeetLattice(Generic[V]):
    """
    Java의 MeetLattice.java 인터페이스: 값(V)을 병합하는 격자 구조 정의
    IFDS에서는 BinaryDomain.TOP/BOTTOM에 대한 AND 연산으로 구현됨
    """
    def top_element(self) -> V:
        """Returns the unique top element of this lattice."""
        raise NotImplementedError

    def bottom_element(self) -> V:
        """Returns the unique bottom element of this lattice."""
        raise NotImplementedError

    def meet(self, left: V, right: V) -> V:
        """Computes the meet (병합) of left and right."""
        raise NotImplementedError


# ====================================================================
# 3. Edge Function (V 값 변환 규칙)
# ====================================================================

class EdgeFunction(Generic[V]):
    """
    Java의 EdgeFunction.java 인터페이스: V-type 값이 엣지를 따라 어떻게 변하는지 계산
    """
    def compute_target(self, source: V) -> V:
        """Computes the value resulting from applying this function to source."""
        raise NotImplementedError

    def compose_with(self, second_function: 'EdgeFunction[V]') -> 'EdgeFunction[V]':
        """Composes this function with the secondFunction (순차 적용)."""
        raise NotImplementedError

    def meet_with(self, other_function: 'EdgeFunction[V]') -> 'EdgeFunction[V]':
        """Returns a function that represents the meet (병합) of this function with otherFunction."""
        raise NotImplementedError

    def equal_to(self, other: 'EdgeFunction[V]') -> bool:
        """Returns true is this function represents exactly the same mapping as other."""
        raise NotImplementedError


class EdgeFunctions(Generic[N, D, M, V]):
    """
    Java의 EdgeFunctions.java 인터페이스: 4가지 종류의 Edge Function을 생성하는 팩토리
    """
    def get_normal_edge_function(self, curr: N, curr_node: D, succ: N, succ_node: D) -> EdgeFunction[V]:
        """일반 엣지 함수"""
        raise NotImplementedError

    def get_call_edge_function(self, call_stmt: N, src_node: D, destination_method: M, dest_node: D) -> EdgeFunction[V]:
        """호출 엣지 함수"""
        raise NotImplementedError

    def get_return_edge_function(self, call_site: Optional[N], callee_method: M, exit_stmt: N, exit_node: D, return_site: Optional[N], ret_node: D) -> EdgeFunction[V]:
        """반환 엣지 함수"""
        raise NotImplementedError

    def get_call_to_return_edge_function(self, call_site: N, call_node: D, return_site: N, return_side_node: D) -> EdgeFunction[V]:
        """Call-to-Return 엣지 함수"""
        raise NotImplementedError