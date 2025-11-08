from typing import TypeVar, Generic, Set, Optional, Collection
#from Core.IFDS_Base_Classes import N, D, M # 이전 단계에서 정의된 타입 변수 import
from Core.FlowFunctions import FlowFunctions, FlowFunction
from Core.IFDSTabulationProblem import N, D, M

N = TypeVar('N')  # Node (Smali Instruction)
D = TypeVar('D')  # Data-flow Fact (TaintFact)
M = TypeVar('M')  # Method (Androguard MethodAnalysis object)

# ====================================================================
# 3. ZeroedFlowFunctions (Zero Value 처리)
# ====================================================================

class ZeroedFlowFunctions(FlowFunctions[N, D, M]):
    """
    Java의 ZeroedFlowFunctions.java 구현:
    Zero Value(오염되지 않은 상태)가 Taint Fact 집합에 미치는 영향을 처리합니다.
    Zero Value는 항상 전파되며, Taint Fact 집합에 포함됩니다.
    """
    
    def __init__(self, delegate: FlowFunctions[N, D, M], zero_value: D):
        # delegate: 실제 Taint 로직을 담당할 FlowFunctions 구현체
        self.delegate = delegate 
        self.zero_value = zero_value

    def get_normal_flow_function(self, curr: N, succ: N) -> FlowFunction[D]:
        return self.ZeroedFlowFunction(self.delegate.get_normal_flow_function(curr, succ), self.zero_value)

    def get_call_flow_function(self, call_stmt: N, destination_method: M) -> FlowFunction[D]:
        return self.ZeroedFlowFunction(self.delegate.get_call_flow_function(call_stmt, destination_method), self.zero_value)

    def get_return_flow_function(self, call_site: Optional[N], callee_method: M, exit_stmt: N, return_site: Optional[N]) -> FlowFunction[D]:
        return self.ZeroedFlowFunction(self.delegate.get_return_flow_function(call_site, callee_method, exit_stmt, return_site), self.zero_value)

    def get_call_to_return_flow_function(self, call_site: N, return_site: N) -> FlowFunction[D]:
        return self.ZeroedFlowFunction(self.delegate.get_call_to_return_flow_function(call_site, return_site), self.zero_value)
    
    class ZeroedFlowFunction(FlowFunction[D]):
        """
        주어진 FlowFunction 결과에 Zero Value를 반드시 포함시키는 래퍼 클래스
        """
        def __init__(self, delegate_func: FlowFunction[D], zero_value: D):
            self.del_func = delegate_func
            self.zero_value = zero_value
            
        def compute_targets(self, source: D) -> Set[D]:
            if source == self.zero_value:
                # Zero Value가 입력으로 들어오면, 원래 함수를 호출하고 Zero Value도 결과에 추가
                res = self.del_func.compute_targets(source)
                res.add(self.zero_value)
                return res
            else:
                # Zero Value가 아니면, 원래 함수만 호출 (IFDS의 정의)
                return self.del_func.compute_targets(source)