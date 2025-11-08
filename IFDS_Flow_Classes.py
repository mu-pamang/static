# # IFDS_Flow_Classes.py
# # 역할: IFDS의 Taint 전파 규칙(Flow Functions) 인터페이스 및 Zero Value 처리 구현
# # 사용된 Java 파일: FlowFunction.java, FlowFunctions.java, ZeroedFlowFunctions.java

# from typing import TypeVar, Generic, Set, Optional, Collection
# #from Core.IFDS_Base_Classes import N, D, M # 이전 단계에서 정의된 타입 변수 import
# from Core.IFDSTabulationProblem import N, D, M, IFDSTabulationProblem # 이전 단계에서 정의된 타입 변수 import
# from Core.InterproceduralCFG import N, D, M, InterproceduralCFG # 이전 단계에서 정의된 타입 변수 import


# # ====================================================================
# # 1. FlowFunction (Taint 전파의 최소 단위)
# # ====================================================================

# class FlowFunction(Generic[D]):
#     """
#     Java의 FlowFunction.java 인터페이스: D-type 값이 어떻게 전파되는지 계산
#     이 메서드를 구현하여 Smali 명령어 하나하나에 대한 Taint 로직을 정의합니다.
#     """
#     def compute_targets(self, source: D) -> Set[D]:
#         """
#         주어진 source Taint Fact로부터 도달 가능한 target Taint Fact 집합을 반환합니다.
#         source가 None이면, 메서드 내에서 Taint가 생성될 때(Zero Value 처리) 사용될 수 있습니다.
#         """
#         raise NotImplementedError

# # ====================================================================
# # 2. FlowFunctions (Taint 전파 규칙의 팩토리)
# # ====================================================================

# class FlowFunctions(Generic[N, D, M]):
#     """
#     Java의 FlowFunctions.java 인터페이스: 4가지 종류의 Flow Function을 생성하는 팩토리
#     이 클래스를 상속받아 Smali 전용 Taint 전파 로직을 구현해야 합니다.
#     """
#     def get_normal_flow_function(self, curr: N, succ: N) -> FlowFunction[D]:
#         """일반 명령(Normal Statement)의 흐름 계산 (Smali Normal Flow 구현 필요)"""
#         raise NotImplementedError

#     def get_call_flow_function(self, call_stmt: N, destination_method: M) -> FlowFunction[D]:
#         """호출 명령(Call Statement)의 흐름 계산 (Smali Interproc Flow 구현 필요)"""
#         raise NotImplementedError

#     def get_return_flow_function(self, call_site: Optional[N], callee_method: M, exit_stmt: N, return_site: Optional[N]) -> FlowFunction[D]:
#         """메서드 종료(Return/Exit Statement)의 흐름 계산 (Smali Interproc Flow 구현 필요)"""
#         raise NotImplementedError

#     def get_call_to_return_flow_function(self, call_site: N, return_site: N) -> FlowFunction[D]:
#         """호출 지점부터 반환 지점까지의 흐름 계산"""
#         raise NotImplementedError

# # ====================================================================
# # 3. ZeroedFlowFunctions (Zero Value 처리)
# # ====================================================================

# class ZeroedFlowFunctions(FlowFunctions[N, D, M]):
#     """
#     Java의 ZeroedFlowFunctions.java 구현:
#     Zero Value(오염되지 않은 상태)가 Taint Fact 집합에 미치는 영향을 처리합니다.
#     Zero Value는 항상 전파되며, Taint Fact 집합에 포함됩니다.
#     """
    
#     def __init__(self, delegate: FlowFunctions[N, D, M], zero_value: D):
#         # delegate: 실제 Taint 로직을 담당할 FlowFunctions 구현체
#         self.delegate = delegate 
#         self.zero_value = zero_value

#     def get_normal_flow_function(self, curr: N, succ: N) -> FlowFunction[D]:
#         return self.ZeroedFlowFunction(self.delegate.get_normal_flow_function(curr, succ), self.zero_value)

#     def get_call_flow_function(self, call_stmt: N, destination_method: M) -> FlowFunction[D]:
#         return self.ZeroedFlowFunction(self.delegate.get_call_flow_function(call_stmt, destination_method), self.zero_value)

#     def get_return_flow_function(self, call_site: Optional[N], callee_method: M, exit_stmt: N, return_site: Optional[N]) -> FlowFunction[D]:
#         return self.ZeroedFlowFunction(self.delegate.get_return_flow_function(call_site, callee_method, exit_stmt, return_site), self.zero_value)

#     def get_call_to_return_flow_function(self, call_site: N, return_site: N) -> FlowFunction[D]:
#         return self.ZeroedFlowFunction(self.delegate.get_call_to_return_flow_function(call_site, return_site), self.zero_value)
    
#     class ZeroedFlowFunction(FlowFunction[D]):
#         """
#         주어진 FlowFunction 결과에 Zero Value를 반드시 포함시키는 래퍼 클래스
#         """
#         def __init__(self, delegate_func: FlowFunction[D], zero_value: D):
#             self.del_func = delegate_func
#             self.zero_value = zero_value
            
#         def compute_targets(self, source: D) -> Set[D]:
#             if source == self.zero_value:
#                 # Zero Value가 입력으로 들어오면, 원래 함수를 호출하고 Zero Value도 결과에 추가
#                 res = self.del_func.compute_targets(source)
#                 res.add(self.zero_value)
#                 return res
#             else:
#                 # Zero Value가 아니면, 원래 함수만 호출 (IFDS의 정의)
#                 return self.del_func.compute_targets(source)