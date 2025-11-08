# IFDS_Base_Classes.py - Java 인터페이스를 Python 클래스로 완전하게 포팅
# IFDSTabulationProblem.java, InterproceduralCFG 2개의 파일을 pythond으로 변경한 코드
from typing import TypeVar, Generic, Dict, Set, List, Collection, Optional, Tuple
# ====================================================================
# 타입 변수 정의: Java Generic 역할 (재사용성을 위한 정의)
# ====================================================================
N = TypeVar('N')  # Node (Smali Instruction/Unit)
D = TypeVar('D')  # Data-flow Fact (TaintFact)
M = TypeVar('M')  # Method (Androguard MethodAnalysis object)
I = TypeVar('I')  # InterproceduralCFG implementation
V = TypeVar('V')  # Value (IFDS/IDE Domain)

# SolverConfiguration은 빈 인터페이스로 가정 (나중에 필요한 설정만 추가)
class SolverConfiguration:
    """Java의 SolverConfiguration 인터페이스 역할"""
    pass

# FlowFunctions 인터페이스는 다음 단계에서 정의할 것
class FlowFunctions(Generic[N, D, M]):
    """Java의 FlowFunctions 인터페이스 역할"""
    pass

# ====================================================================
# 2. InterproceduralCFG (I) - [InterproceduralCFG.java 기능]
# ====================================================================
class InterproceduralCFG(Generic[N, M]):
    """
    IFDS Solver가 탐색할 수 있는 프로그램의 Interprocedural Control-Flow Graph 인터페이스.
    Smali 코드를 기반으로 이 클래스의 모든 메서드를 구현해야 합니다.
    """
    def get_method_of(self, n: N) -> M:
        """Returns the method containing a node."""
        raise NotImplementedError

    def get_preds_of(self, u: N) -> List[N]:
        """Returns the predecessor nodes of u (CFG)."""
        raise NotImplementedError

    def get_succs_of(self, n: N) -> List[N]:
        """Returns the successor nodes of n (CFG)."""
        raise NotImplementedError

    def get_callees_of_call_at(self, n: N) -> Collection[M]:
        """Returns all callee methods for a given call."""
        raise NotImplementedError

    def get_callers_of(self, m: M) -> Collection[N]:
        """Returns all caller statements/nodes of a given method."""
        raise NotImplementedError
        
    def get_calls_from_within(self, m: M) -> Set[N]:
        """Returns all call sites within a given method."""
        raise NotImplementedError
        
    def get_start_points_of(self, m: M) -> Collection[N]:
        """Returns all start points of a given method."""
        raise NotImplementedError

    def get_return_sites_of_call_at(self, n: N) -> Collection[N]:
        """Returns all statements to which a call could return."""
        raise NotImplementedError

    def is_call_stmt(self, stmt: N) -> bool:
        """Returns true if the given statement is a call site."""
        raise NotImplementedError

    def is_exit_stmt(self, stmt: N) -> bool:
        """Returns true if the given statement leads to a method return."""
        raise NotImplementedError
        
    def is_start_point(self, stmt: N) -> bool:
        """Returns true is this is a method's start statement."""
        raise NotImplementedError

    def all_non_call_start_nodes(self) -> Set[N]:
        """Returns the set of all nodes that are neither call nor start nodes."""
        raise NotImplementedError
        
    def is_fall_through_successor(self, stmt: N, succ: N) -> bool:
        """Returns whether succ is the fall-through successor of stmt."""
        raise NotImplementedError
        
    def is_branch_target(self, stmt: N, succ: N) -> bool:
        """Returns whether succ is a branch target of stmt."""
        raise NotImplementedError

# ====================================================================
# 3. IFDSTabulationProblem - [IFDSTabulationProblem.java 기능]
# ====================================================================
class IFDSTabulationProblem(Generic[N, D, M, I], SolverConfiguration):
    """
    IFDS Solver에 필요한 모든 설정을 제공하는 인터페이스입니다.
    이 클래스를 상속받아 Smali 분석에 맞게 구현해야 합니다.
    """

    def flow_functions(self) -> FlowFunctions[N, D, M]:
        """
        IFDS Flow Functions 구현체를 반환합니다. (규칙)
        NOTE: 반환 값은 캐시되어야 합니다.
        """
        raise NotImplementedError

    def interprocedural_cfg(self) -> I:
        """
        Interprocedural CFG 구현체(Supergraph)를 반환합니다. (그래프 구조)
        NOTE: 반환 값은 캐시되어야 합니다.
        """
        raise NotImplementedError

    def initial_seeds(self) -> Dict[N, Set[D]]:
        """
        분석을 시작할 초기 Taint Seeds (명령어: Taint Fact 집합)를 반환합니다. (Source API 지점)
        """
        raise NotImplementedError

    def zero_value(self) -> D:
        """
        오염되지 않은 상태를 나타내는 특수한 Data-Flow Fact (D)를 반환합니다. 
        이 값은 일반적인 Taint Fact 도메인의 어떤 값과도 동일해서는 안 됩니다.
        NOTE: 반환 값은 캐시되어야 합니다.
        """
        raise NotImplementedError