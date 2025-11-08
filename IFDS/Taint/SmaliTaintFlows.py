# SmaliTaintFlows.py (Androguard API 통합 완료 버전)
# 역할: Androguard Instruction 객체와 Analysis를 활용한 Taint 전파 규칙 구현

import re
from typing import Set, Optional, Dict, List, Any
from Core.FlowFunctions import FlowFunctions # FlowFunctions 팩토리 인터페이스
from Core.FlowFunction import FlowFunction   # FlowFunction 인터페이스
from Core.InterproceduralCFG import InterproceduralCFG, N, M 
from .TaintFact import TaintFact

# ====================================================================
# A. Smali 명령어별 Flow Function 구현체 (D = TaintFact)
# ====================================================================

class IdentityFlowFunction(FlowFunction[TaintFact]):
    """변경이 없는 명령어 (NOP, goto 등)에 사용: Taint를 그대로 통과"""
    def compute_targets(self, source: TaintFact) -> Set[TaintFact]:
        return {source}


class KillFlowFunction(FlowFunction[TaintFact]):
    """상수 할당 명령어에 사용: Taint를 제거 (새로운 상수 값은 안전하다고 가정)"""
    def __init__(self, target_reg: str):
        self.target_reg = target_reg

    def compute_targets(self, source: TaintFact) -> Set[TaintFact]:
        if source.target == self.target_reg:
            return set()  # Kill
        return {source}


class MoveFlowFunction(FlowFunction[TaintFact]):
    """move v0, v1 명령에 사용: Taint 전파"""
    def __init__(self, dest_reg: str, src_reg: str):
        self.dest_reg = dest_reg
        self.src_reg = src_reg

    def compute_targets(self, source: TaintFact) -> Set[TaintFact]:
        targets = set()
        
        # Kill: 목적지 레지스터의 이전 Taint 제거
        if source.target != self.dest_reg:
            targets.add(source)
        
        # Gen: 소스 레지스터가 Taint되어 있으면 전파
        if source.target == self.src_reg:
            targets.add(TaintFact(target=self.dest_reg, source_api=source.source))
            
        return targets


class IgetFlowFunction(FlowFunction[TaintFact]):
    """iget-object v0, v1, LClass;->field:LType; 명령어"""
    def __init__(self, dest_reg: str, obj_reg: str, field_sig: str):
        self.dest_reg = dest_reg
        self.obj_reg = obj_reg
        self.field_sig = field_sig

    def compute_targets(self, source: TaintFact) -> Set[TaintFact]:
        targets = set()
        
        # Kill: dest_reg의 이전 Taint 제거
        if source.target != self.dest_reg:
            targets.add(source)
        
        # Gen: obj_reg가 Taint되어 있으면 전파
        if source.target == self.obj_reg:
            targets.add(TaintFact(target=self.dest_reg, source_api=source.source))
        
        # Gen: 필드 자체가 Taint되어 있으면 전파
        if source.target == self.field_sig:
            targets.add(TaintFact(target=self.dest_reg, source_api=source.source))
            
        return targets


class SputFlowFunction(FlowFunction[TaintFact]):
    """sput-object v0, LClass;->field:LType; 명령어"""
    def __init__(self, src_reg: str, field_sig: str):
        self.src_reg = src_reg
        self.field_sig = field_sig

    def compute_targets(self, source: TaintFact) -> Set[TaintFact]:
        targets = {source}  # 기존 Taint 유지
        
        # Gen: src_reg가 Taint되어 있으면 필드로 전파
        if source.target == self.src_reg:
            targets.add(TaintFact(target=self.field_sig, source_api=source.source))
            
        return targets


class CallFlowFunction(FlowFunction[TaintFact]):
    """invoke-* 명령어의 caller → callee 전파"""
    def __init__(self, call_args: List[str], callee_params: List[str]):
        """
        Args:
            call_args: 호출 시 사용된 레지스터 (예: ["v0", "v1"])
            callee_params: callee 메서드의 파라미터 레지스터 (예: ["p0", "p1"])
        """
        self.arg_to_param = dict(zip(call_args, callee_params))

    def compute_targets(self, source: TaintFact) -> Set[TaintFact]:
        # caller의 Taint를 callee의 파라미터로 매핑
        if source.target in self.arg_to_param:
            param_reg = self.arg_to_param[source.target]
            return {TaintFact(target=param_reg, source_api=source.source)}
        return set()


class ReturnFlowFunction(FlowFunction[TaintFact]):
    """return-* 명령어의 callee → caller 전파"""
    def __init__(self, return_reg: Optional[str], result_reg: Optional[str]):
        """
        Args:
            return_reg: callee의 반환 레지스터 (예: "v0")
            result_reg: caller의 move-result-object 레지스터 (예: "v2")
        """
        self.return_reg = return_reg
        self.result_reg = result_reg

    def compute_targets(self, source: TaintFact) -> Set[TaintFact]:
        # return_reg가 Taint되어 있으면 result_reg로 전파
        if self.return_reg and self.result_reg and source.target == self.return_reg:
            return {TaintFact(target=self.result_reg, source_api=source.source)}
        return set()


# ====================================================================
# B. SmaliTaintFlowFunctions (메인 팩토리 클래스)
# ====================================================================

class SmaliTaintFlowFunctions(FlowFunctions[N, TaintFact, M]):
    """
    Smali Taint Analysis를 위한 FlowFunctions 구현체.
    N은 Androguard Instruction 객체입니다.
    M은 Androguard MethodAnalysis 객체입니다.
    """
    def __init__(self, source_apis: Set[str], icfg):
        self.source_apis = source_apis
        self.icfg = icfg  # InterproceduralCFG

    # ====================================================================
    # 1. Androguard Instruction 파싱 (API 정확 반영)
    # ====================================================================
    def _parse_instruction(self, inst: N) -> Dict[str, Any]:
        """
        Androguard Instruction 객체를 파싱하여 구조화된 딕셔너리 반환
        
        Androguard API 사용:
        - inst.get_name() → opcode (예: "move-object", "invoke-virtual")
        - inst.get_output() → 전체 명령어 문자열 (예: "v0, v1")
        
        Returns:
            {
                'opcode': str,
                'registers': List[str],
                'field_sig': str,      # iget/iput/sget/sput만
                'method_sig': str      # invoke만
            }
        """
        result = {
            'opcode': '',
            'registers': [],
            'field_sig': None,
            'method_sig': None
        }
        
        try:
            # Androguard Instruction 객체에서 직접 추출
            opcode = inst.get_name()
            result['opcode'] = opcode
            
            # get_output()으로 전체 명령어 문자열 획득
            output = inst.get_output()
            
            # invoke 계열: invoke-virtual {v0, v1}, Lcom/example/Class;->method(I)V
            if opcode.startswith('invoke-'):
                match = re.match(r'\{([^}]*)\}\s*,\s*(.+)', output)
                if match:
                    regs_str = match.group(1)
                    result['registers'] = [r.strip() for r in regs_str.split(',') if r.strip()]
                    result['method_sig'] = match.group(2).strip()
            
            # 필드 접근 계열: iget-object v0, v1, LClass;->field:LType;
            elif opcode.startswith(('iget', 'iput', 'sget', 'sput')):
                parts = output.split(',')
                for part in parts:
                    part = part.strip()
                    if '->' in part:
                        result['field_sig'] = part
                    elif re.match(r'^[vp]\d+$', part):
                        result['registers'].append(part)
            
            # move 계열: move-object v0, v1
            elif opcode.startswith('move'):
                regs = [r.strip() for r in output.split(',')]
                result['registers'] = [r for r in regs if re.match(r'^[vp]\d+$', r)]
            
            # const 계열: const-string v0, "hello"
            elif opcode.startswith('const'):
                match = re.match(r'^([vp]\d+)', output)
                if match:
                    result['registers'].append(match.group(1))
            
            # return 계열: return-object v0
            elif opcode.startswith('return'):
                match = re.match(r'^([vp]\d+)', output)
                if match:
                    result['registers'].append(match.group(1))
            
            # 기타: 레지스터 추출
            else:
                result['registers'] = [
                    r.strip() for r in output.split(',')
                    if r.strip() and re.match(r'^[vp]\d+$', r.strip())
                ]
        
        except Exception as e:
            # Fallback: 문자열 파싱
            print(f"Warning: Instruction parsing failed: {e}")
            inst_str = str(inst).strip().split('#')[0].strip()
            parts = inst_str.split(None, 1)
            result['opcode'] = parts[0] if parts else ''
            if len(parts) > 1:
                output = parts[1]
                result['registers'] = re.findall(r'[vp]\d+', output)
        
        return result

    # ====================================================================
    # 2. invoke 인자 추출
    # ====================================================================
    def _extract_call_args(self, info: Dict[str, Any]) -> List[str]:
        """invoke 명령어에서 호출 인자 레지스터 추출"""
        return info.get('registers', [])

    # ====================================================================
    # 3. 메서드 파라미터 추출 (Androguard MethodAnalysis 사용)
    # ====================================================================
    def _get_callee_params(self, method: M) -> List[str]:
        """
        Androguard MethodAnalysis 객체로부터 파라미터 레지스터 추출
        
        사용 API:
        - method.get_method() → EncodedMethod 객체
        - method.get_method().get_descriptor() → 메서드 시그니처
        - method.get_method().get_access_flags() → ACC_STATIC 확인
        
        Returns:
            ["p0", "p1", "p2"]  # 인스턴스 메서드 (p0 = this)
            ["p0", "p1"]        # 정적 메서드
        """
        try:
            # Androguard MethodAnalysis → EncodedMethod
            encoded_method = method.get_method()
            
            # 메서드 디스크립터에서 파라미터 타입 추출
            # 예: "(ILjava/lang/String;)V" → ["I", "Ljava/lang/String;"]
            descriptor = encoded_method.get_descriptor()
            match = re.match(r'.*\(([^)]*)\).*', descriptor)
            if not match:
                return []
            
            params_str = match.group(1)
            param_types = self._parse_param_types(params_str)
            
            # ACC_STATIC 플래그 확인 (0x0008)
            access_flags = encoded_method.get_access_flags()
            is_static = bool(access_flags & 0x0008)
            
            # 파라미터 레지스터 생성
            param_regs = []
            if not is_static:
                param_regs.append('p0')  # this 포인터
                start_idx = 1
            else:
                start_idx = 0
            
            for i, param_type in enumerate(param_types):
                param_regs.append(f'p{start_idx + i}')
                # Wide 타입 (J=long, D=double)은 2개 레지스터 사용
                if param_type in ['J', 'D']:
                    start_idx += 1
            
            return param_regs
        
        except Exception as e:
            print(f"Warning: Failed to extract params from {method}: {e}")
            return []

    def _parse_param_types(self, params_str: str) -> List[str]:
        """
        파라미터 디스크립터 문자열을 타입 리스트로 파싱
        
        예: "ILjava/lang/String;" → ["I", "Ljava/lang/String;"]
        """
        param_types = []
        i = 0
        while i < len(params_str):
            if params_str[i] == 'L':
                # 객체 타입: L...;
                end = params_str.index(';', i) + 1
                param_types.append(params_str[i:end])
                i = end
            elif params_str[i] == '[':
                # 배열 타입: [[I, [Ljava/lang/String;
                j = i + 1
                while j < len(params_str) and params_str[j] == '[':
                    j += 1
                if j < len(params_str):
                    if params_str[j] == 'L':
                        end = params_str.index(';', j) + 1
                        param_types.append(params_str[i:end])
                        i = end
                    else:
                        param_types.append(params_str[i:j+1])
                        i = j + 1
            else:
                # 기본 타입: I, J, Z, B, C, S, F, D
                param_types.append(params_str[i])
                i += 1
        return param_types

    # ====================================================================
    # 4. return-* 명령어 다음의 move-result-* 레지스터 찾기
    # ====================================================================
    def _find_move_result_register(self, call_site: N, return_site: Optional[N]) -> Optional[str]:
        """
        invoke 명령어 다음의 move-result-* 명령어에서 결과 레지스터 추출
        
        call_site: invoke-* 명령어
        return_site: invoke 다음 명령어 (move-result-object 등)
        """
        if not return_site:
            return None
        
        try:
            next_opcode = return_site.get_name()
            if next_opcode.startswith('move-result'):
                next_output = return_site.get_output()
                match = re.match(r'^([vp]\d+)', next_output)
                if match:
                    return match.group(1)
        except:
            pass
        
        return None

    # ====================================================================
    # IFDS FlowFunctions 인터페이스 구현
    # ====================================================================

    def get_normal_flow_function(self, curr: N, succ: N) -> FlowFunction[TaintFact]:
        """일반 명령어의 Taint 전파 규칙"""
        info = self._parse_instruction(curr)
        op = info['opcode']
        regs = info['registers']
        
        # move 계열
        if op in {"move", "move-object", "move-wide"}:
            if len(regs) >= 2:
                return MoveFlowFunction(regs[0], regs[1])
        
        # const 계열 (Taint Kill)
        elif op in {"const-string", "const", "const-wide", "const/4", "const/16"}:
            if regs:
                return KillFlowFunction(regs[0])
        
        # iget 계열 (필드 읽기)
        elif op in {"iget-object", "iget", "iget-wide", "iget-boolean"}:
            if len(regs) >= 2 and info['field_sig']:
                return IgetFlowFunction(regs[0], regs[1], info['field_sig'])
        
        # sput 계열 (정적 필드 쓰기)
        elif op in {"sput-object", "sput", "sput-wide", "sput-boolean"}:
            if regs and info['field_sig']:
                return SputFlowFunction(regs[0], info['field_sig'])
        
        # 기본: Identity
        return IdentityFlowFunction()

    def get_call_flow_function(self, call_stmt: N, destination_method: M) -> FlowFunction[TaintFact]:
        """메서드 호출 시 caller → callee로의 Taint 전파"""
        info = self._parse_instruction(call_stmt)
        call_args = self._extract_call_args(info)
        callee_params = self._get_callee_params(destination_method)
        
        return CallFlowFunction(call_args, callee_params)

    def get_return_flow_function(self, call_site: Optional[N], callee_method: M, 
                                  exit_stmt: N, return_site: Optional[N]) -> FlowFunction[TaintFact]:
        """메서드 반환 시 callee → caller로의 Taint 전파"""
        
        # exit_stmt (return-*) 에서 반환 레지스터 추출
        exit_info = self._parse_instruction(exit_stmt)
        return_reg = exit_info['registers'][0] if exit_info['registers'] else None
        
        # return_site (move-result-*) 에서 결과 레지스터 추출
        result_reg = self._find_move_result_register(call_site, return_site) if call_site else None
        
        return ReturnFlowFunction(return_reg, result_reg)

    def get_call_to_return_flow_function(self, call_site: N, return_site: N) -> FlowFunction[TaintFact]:
        """메서드 호출을 거치지 않는 로컬 Taint 전파 (call-to-return edge)"""
        # 대부분의 경우 Identity로 충분
        return IdentityFlowFunction()