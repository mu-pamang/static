# SmaliICFG.py
# 역할: Androguard Analysis 객체를 사용한 Interprocedural CFG 구현

from typing import List, Collection, Set, Optional
from Core.InterproceduralCFG import InterproceduralCFG, N, M

class SmaliInterproceduralCFG(InterproceduralCFG[N, M]):
    """
    Androguard의 Analysis 객체를 사용하여 ICFG를 제공하는 클래스
    
    N: Androguard Instruction 객체
    M: Androguard MethodAnalysis 객체
    """
    
    def __init__(self, dx):
        """
        Args:
            dx: androguard.core.analysis.analysis.Analysis 객체
        """
        self.dx = dx  # Analysis 객체
        self.method_cache = {}  # instruction → method 캐시
        self.cfg_cache = {}     # method → BasicBlocks 캐시
        
        # Analysis 객체에서 모든 메서드 수집
        self._build_method_cache()
    
    def _build_method_cache(self):
        """모든 Instruction → Method 매핑 구축"""
        for method_analysis in self.dx.get_methods():
            method = method_analysis.get_method()
            
            # BasicBlock을 통해 Instruction 순회
            try:
                for basic_block in method_analysis.get_basic_blocks().get():
                    for inst in basic_block.get_instructions():
                        self.method_cache[id(inst)] = method_analysis
            except:
                # 메서드에 코드가 없는 경우 (추상 메서드 등)
                pass
    
    # ====================================================================
    # InterproceduralCFG 인터페이스 구현
    # ====================================================================
    
    def get_method_of(self, n: N) -> M:
        """주어진 Instruction이 속한 메서드 반환"""
        inst_id = id(n)
        if inst_id in self.method_cache:
            return self.method_cache[inst_id]
        
        # 캐시 미스 시 재탐색
        for method_analysis in self.dx.get_methods():
            try:
                for bb in method_analysis.get_basic_blocks().get():
                    for inst in bb.get_instructions():
                        if inst == n or id(inst) == inst_id:
                            self.method_cache[inst_id] = method_analysis
                            return method_analysis
            except:
                continue
        
        raise ValueError(f"Method not found for instruction: {n}")
    
    def get_preds_of(self, u: N) -> List[N]:
        """주어진 Instruction의 predecessor 반환 (CFG)"""
        method = self.get_method_of(u)
        preds = []
        
        try:
            basic_blocks = method.get_basic_blocks().get()
            
            # 현재 instruction이 속한 BasicBlock 찾기
            current_bb = None
            inst_list = []
            for bb in basic_blocks:
                inst_list = list(bb.get_instructions())
                if u in inst_list:
                    current_bb = bb
                    break
            
            if not current_bb:
                return []
            
            # 같은 BasicBlock 내에서 predecessor 찾기
            idx = inst_list.index(u)
            if idx > 0:
                preds.append(inst_list[idx - 1])
            else:
                # BasicBlock의 첫 번째 instruction이면 이전 BB의 마지막 instruction
                for bb in basic_blocks:
                    # bb가 current_bb의 predecessor인지 확인
                    # (Androguard는 get_prev() 메서드로 이전 BB 획득)
                    try:
                        for prev_bb in bb.get_next():
                            if prev_bb[2] == current_bb:
                                prev_insts = list(prev_bb[2].get_instructions())
                                if prev_insts:
                                    preds.append(prev_insts[-1])
                    except:
                        continue
        
        except Exception as e:
            print(f"Warning: Failed to get predecessors of {u}: {e}")
        
        return preds
    
    def get_succs_of(self, n: N) -> List[N]:
        """주어진 Instruction의 successor 반환 (CFG)"""
        method = self.get_method_of(n)
        succs = []
        
        try:
            basic_blocks = method.get_basic_blocks().get()
            
            # 현재 instruction이 속한 BasicBlock 찾기
            current_bb = None
            inst_list = []
            for bb in basic_blocks:
                inst_list = list(bb.get_instructions())
                if n in inst_list:
                    current_bb = bb
                    break
            
            if not current_bb:
                return []
            
            # 같은 BasicBlock 내에서 successor 찾기
            idx = inst_list.index(n)
            if idx < len(inst_list) - 1:
                succs.append(inst_list[idx + 1])
            else:
                # BasicBlock의 마지막 instruction이면 다음 BB의 첫 instruction
                try:
                    for next_info in current_bb.get_next():
                        # next_info: (condition, target_bb, ...)
                        next_bb = next_info[2]
                        next_insts = list(next_bb.get_instructions())
                        if next_insts:
                            succs.append(next_insts[0])
                except:
                    pass
        
        except Exception as e:
            print(f"Warning: Failed to get successors of {n}: {e}")
        
        return succs
    
    def get_callees_of_call_at(self, n: N) -> Collection[M]:
        """invoke 명령어의 callee 메서드 반환"""
        callees = []
        
        try:
            opcode = n.get_name()
            if not opcode.startswith('invoke-'):
                return callees
            
            # invoke 명령어에서 메서드 시그니처 추출
            output = n.get_output()
            import re
            match = re.match(r'\{[^}]*\}\s*,\s*(.+)', output)
            if not match:
                return callees
            
            method_sig = match.group(1).strip()
            
            # Analysis 객체에서 해당 메서드 검색
            # 메서드 시그니처 포맷: Lcom/example/Class;->method(I)V
            parts = method_sig.split('->')
            if len(parts) != 2:
                return callees
            
            class_name = parts[0]
            method_name_desc = parts[1]
            
            # method_name_desc: "method(I)V"
            paren_idx = method_name_desc.index('(')
            method_name = method_name_desc[:paren_idx]
            method_desc = method_name_desc[paren_idx:]
            
            # Analysis에서 메서드 찾기
            for method_analysis in self.dx.get_methods():
                method = method_analysis.get_method()
                
                # 클래스 이름 비교
                if method.get_class_name() != class_name:
                    continue
                
                # 메서드 이름 비교
                if method.get_name() != method_name:
                    continue
                
                # 디스크립터 비교
                if method.get_descriptor() != method_desc:
                    continue
                
                callees.append(method_analysis)
                break
        
        except Exception as e:
            print(f"Warning: Failed to get callees of {n}: {e}")
        
        return callees
    
    def get_callers_of(self, m: M) -> Collection[N]:
        """주어진 메서드를 호출하는 모든 caller instruction 반환"""
        callers = []
        
        try:
            # MethodAnalysis의 get_xref_from() 사용
            xrefs = m.get_xref_from()
            
            for xref in xrefs:
                # xref: (ClassAnalysis, MethodAnalysis, offset)
                caller_method = xref[1]
                offset = xref[2]
                
                # offset에 해당하는 instruction 찾기
                try:
                    for bb in caller_method.get_basic_blocks().get():
                        for inst in bb.get_instructions():
                            # Androguard instruction의 offset 확인
                            if hasattr(inst, 'get_offset') and inst.get_offset() == offset:
                                callers.append(inst)
                                break
                except:
                    continue
        
        except Exception as e:
            print(f"Warning: Failed to get callers of {m}: {e}")
        
        return callers
    
    def get_calls_from_within(self, m: M) -> Set[N]:
        """주어진 메서드 내의 모든 invoke 명령어 반환"""
        call_sites = set()
        
        try:
            for bb in m.get_basic_blocks().get():
                for inst in bb.get_instructions():
                    opcode = inst.get_name()
                    if opcode.startswith('invoke-'):
                        call_sites.add(inst)
        
        except Exception as e:
            print(f"Warning: Failed to get calls from {m}: {e}")
        
        return call_sites
    
    def get_start_points_of(self, m: M) -> Collection[N]:
        """메서드의 시작 instruction 반환"""
        start_points = []
        
        try:
            basic_blocks = m.get_basic_blocks().get()
            if basic_blocks:
                first_bb = basic_blocks[0]
                first_insts = list(first_bb.get_instructions())
                if first_insts:
                    start_points.append(first_insts[0])
        
        except Exception as e:
            print(f"Warning: Failed to get start points of {m}: {e}")
        
        return start_points
    
    def get_return_sites_of_call_at(self, n: N) -> Collection[N]:
        """invoke 명령어의 return site (다음 instruction) 반환"""
        succs = self.get_succs_of(n)
        return succs  # invoke 다음 instruction (move-result-* 등)
    
    def is_call_stmt(self, stmt: N) -> bool:
        """주어진 instruction이 invoke 명령어인지 확인"""
        try:
            opcode = stmt.get_name()
            return opcode.startswith('invoke-')
        except:
            return False
    
    def is_exit_stmt(self, stmt: N) -> bool:
        """주어진 instruction이 return 명령어인지 확인"""
        try:
            opcode = stmt.get_name()
            return opcode.startswith('return')
        except:
            return False
    
    def is_start_point(self, stmt: N) -> bool:
        """주어진 instruction이 메서드의 시작점인지 확인"""
        try:
            method = self.get_method_of(stmt)
            start_points = self.get_start_points_of(method)
            return stmt in start_points
        except:
            return False
    
    def all_non_call_start_nodes(self) -> Set[N]:
        """call site도 start point도 아닌 모든 instruction 반환"""
        nodes = set()
        
        for method_analysis in self.dx.get_methods():
            try:
                for bb in method_analysis.get_basic_blocks().get():
                    for inst in bb.get_instructions():
                        if not self.is_call_stmt(inst) and not self.is_start_point(inst):
                            nodes.add(inst)
            except:
                continue
        
        return nodes
    
    def is_fall_through_successor(self, stmt: N, succ: N) -> bool:
        """succ가 stmt의 fall-through successor인지 확인"""
        # Fall-through: 조건 분기가 아닌 순차 실행
        succs = self.get_succs_of(stmt)
        if len(succs) == 1 and succs[0] == succ:
            return True
        return False
    
    def is_branch_target(self, stmt: N, succ: N) -> bool:
        """succ가 stmt의 branch target인지 확인"""
        # Branch: 조건 분기로 인한 점프
        succs = self.get_succs_of(stmt)
        if len(succs) > 1 and succ in succs:
            return True
        return False