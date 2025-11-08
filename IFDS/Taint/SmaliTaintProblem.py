# SmaliTaintProblem.py
# 역할: IFDS Taint Analysis 문제 정의

from typing import Dict, Set,TypeVar
from Core.IFDSTabulationProblem import IFDSTabulationProblem
from solver.ZeroedFlowFunctions import ZeroedFlowFunctions
from solver.ZeroedFlowFunctions import ZeroedFlowFunctions, FlowFunctions
from .TaintFact import TaintFact 
from .SmaliInterproceduralCFG import SmaliInterproceduralCFG
from .SmaliTaintFlows import SmaliTaintFlowFunctions


# Type variables
N = TypeVar('N')  # Node type
D = TypeVar('D')  # Data-flow abstraction type
M = TypeVar('M')  # Method type
V = TypeVar('V')  # Value type
I = TypeVar('I')  # InterproceduralCFG type

class SmaliTaintProblem(IFDSTabulationProblem[N, TaintFact, M, SmaliInterproceduralCFG]):
    """
    Smali Taint Analysis를 위한 IFDS Problem 구현
    
    N: Androguard Instruction
    D: TaintFact
    M: Androguard MethodAnalysis
    I: SmaliInterproceduralCFG
    """
    
    def __init__(self, dx, source_apis: Set[str], sink_apis: Set[str] = None):
        """
        Args:
            dx: Androguard Analysis 객체
            source_apis: Source API 시그니처 집합 (예: {"Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;"})
            sink_apis: Sink API 시그니처 집합 (선택 사항)
        """
        self.dx = dx
        self.source_apis = source_apis
        self.sink_apis = sink_apis or set()
        
        # ICFG 생성
        self._icfg = SmaliInterproceduralCFG(dx)
        
        # Flow Functions 생성
        self._flow_functions = SmaliTaintFlowFunctions(source_apis, self._icfg)
        
        # Zero Value (오염되지 않은 상태)
        self._zero_value = TaintFact(target="ZERO", source_api="ZERO")
        
        # ZeroedFlowFunctions로 래핑 (Zero Value 자동 전파)
        self._zeroed_flow_functions = ZeroedFlowFunctions(self._flow_functions, self._zero_value)
    
    def flow_functions(self) -> FlowFunctions[N, TaintFact, M]:
        """IFDS Flow Functions 반환"""
        return self._zeroed_flow_functions
    
    def interprocedural_cfg(self) -> SmaliInterproceduralCFG:
        """Interprocedural CFG 반환"""
        return self._icfg
    
    def initial_seeds(self) -> Dict[N, Set[TaintFact]]:
        """
        Source API 호출 지점을 초기 Taint Seed로 반환
        
        Returns:
            {<invoke instruction>: {TaintFact(target="v0", source_api="getDeviceId")}}
        """
        seeds = {}
        
        # 모든 메서드 순회하여 Source API 호출 찾기
        for method_analysis in self.dx.get_methods():
            try:
                for bb in method_analysis.get_basic_blocks().get():
                    for inst in bb.get_instructions():
                        # invoke 명령어인지 확인
                        opcode = inst.get_name()
                        if not opcode.startswith('invoke-'):
                            continue
                        
                        # 메서드 시그니처 추출
                        output = inst.get_output()
                        import re
                        match = re.match(r'\{([^}]*)\}\s*,\s*(.+)', output)
                        if not match:
                            continue
                        
                        method_sig = match.group(2).strip()
                        
                        # Source API인지 확인
                        if method_sig in self.source_apis:
                            # invoke 다음 instruction (move-result-*)에서 결과 레지스터 추출
                            succs = self._icfg.get_succs_of(inst)
                            if succs:
                                next_inst = succs[0]
                                next_opcode = next_inst.get_name()
                                if next_opcode.startswith('move-result'):
                                    next_output = next_inst.get_output()
                                    reg_match = re.match(r'^([vp]\d+)', next_output)
                                    if reg_match:
                                        result_reg = reg_match.group(1)
                                        
                                        # Seed 생성: move-result-* instruction에 Taint 주입
                                        taint_fact = TaintFact(target=result_reg, source_api=method_sig)
                                        seeds[next_inst] = {self._zero_value, taint_fact}
                                        
                                        print(f"[Seed] Found source: {method_sig} → {result_reg} at {next_inst}")
            
            except Exception as e:
                print(f"Warning: Failed to process method {method_analysis}: {e}")
                continue
        
        return seeds
    
    def zero_value(self) -> TaintFact:
        """Zero Value 반환 (오염되지 않은 상태)"""
        return self._zero_value
    
    # ====================================================================
    # Sink 감지 (분석 완료 후 호출)
    # ====================================================================
    
    def check_sinks(self, solver):
        """
        Sink API 호출 지점에서 Taint 유무 확인
        
        Args:
            solver: IFDSSolver 객체 (분석 완료된 상태)
        
        Returns:
            List of detected flows: [(sink_instruction, taint_facts)]
        """
        detected_flows = []
        
        for method_analysis in self.dx.get_methods():
            try:
                for bb in method_analysis.get_basic_blocks().get():
                    for inst in bb.get_instructions():
                        # invoke 명령어인지 확인
                        opcode = inst.get_name()
                        if not opcode.startswith('invoke-'):
                            continue
                        
                        # 메서드 시그니처 추출
                        output = inst.get_output()
                        import re
                        match = re.match(r'\{([^}]*)\}\s*,\s*(.+)', output)
                        if not match:
                            continue
                        
                        method_sig = match.group(2).strip()
                        
                        # Sink API인지 확인
                        if method_sig in self.sink_apis:
                            # Solver에서 이 지점의 Taint Fact 조회
                            taint_facts = solver.ifds_results_at(inst)
                            
                            # Zero Value 제외
                            real_taints = {t for t in taint_facts if t != self._zero_value}
                            
                            if real_taints:
                                detected_flows.append((inst, real_taints))
                                print(f"[LEAK] Detected taint at sink: {method_sig}")
                                for taint in real_taints:
                                    print(f"  - Source: {taint.source}, Target: {taint.target}")
            
            except Exception as e:
                print(f"Warning: Failed to check sinks in {method_analysis}: {e}")
                continue
        
        return detected_flows