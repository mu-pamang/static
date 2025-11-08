#!/usr/bin/env python3
# main.py
# Androguard + IFDS 기반 Taint Analysis 실행 스크립트

import sys, os
import argparse
from androguard.misc import AnalyzeAPK


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Taint.SmaliTaintProblem import SmaliTaintProblem
from solver.IFDSSolver import IFDSSolver
from Taint.TaintFact import TaintFact
from Core.FlowFunctions import FlowFunctions


def main():
    parser = argparse.ArgumentParser(description='Androguard IFDS Taint Analysis')
    parser.add_argument('apk', help='APK file to analyze')
    parser.add_argument('--sources', nargs='+', help='Source API methods (e.g., getDeviceId)', 
                        default=['getDeviceId'])
    parser.add_argument('--sinks', nargs='+', help='Sink API methods (e.g., sendTextMessage)',
                        default=['sendTextMessage', 'upload'])
    
    args = parser.parse_args()
    
    print(f"[*] Analyzing APK: {args.apk}")
    print(f"[*] Source APIs: {args.sources}")
    print(f"[*] Sink APIs: {args.sinks}")
    
    # ====================================================================
    # 1. Androguard로 APK 분석
    # ====================================================================
    print("\n[1/4] Loading APK with Androguard...")
    try:
        a, d, dx = AnalyzeAPK(args.apk)
        print(f"  - Package: {a.get_package()}")
        print(f"  - DEX files: {len(d)}")
        print(f"  - Methods: {len(list(dx.get_methods()))}")
    except Exception as e:
        print(f"[ERROR] Failed to load APK: {e}")
        return 1
    
    # ====================================================================
    # 2. Source/Sink API 시그니처 변환
    # ====================================================================
    print("\n[2/4] Building Source/Sink signatures...")
    
    # 간단한 API 이름 → 완전한 시그니처 매핑
    # 실제로는 dx.find_methods()를 사용하여 정확한 시그니처 찾기
    source_sigs = set()
    for src in args.sources:
        # 예: "getDeviceId" → "Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;"
        methods = dx.find_methods(methodname=src)
        for method in methods:
            sig = str(method)
            source_sigs.add(sig)
            print(f"  [Source] {sig}")
    
    sink_sigs = set()
    for sink in args.sinks:
        methods = dx.find_methods(methodname=sink)
        for method in methods:
            sig = str(method)
            sink_sigs.add(sig)
            print(f"  [Sink] {sig}")
    
    if not source_sigs:
        print("[WARNING] No source APIs found!")
    
    if not sink_sigs:
        print("[WARNING] No sink APIs found!")
    
    # ====================================================================
    # 3. IFDS Problem 정의
    # ====================================================================
    print("\n[3/4] Setting up IFDS Taint Analysis...")
    problem = SmaliTaintProblem(dx, source_sigs, sink_sigs)
    
    # Initial Seeds 확인
    seeds = problem.initial_seeds()
    print(f"  - Found {len(seeds)} taint seeds")
    
    # ====================================================================
    # 4. IFDS Solver 실행
    # ====================================================================
    print("\n[4/4] Running IFDS Solver...")
    print("[WARNING] Full IFDS Solver implementation is not complete in this demo.")
    print("[WARNING] This is a framework skeleton. IDESolver needs worklist algorithm.")
    
    # TODO: IDESolver의 실제 worklist 알고리즘 구현 필요
    # solver = IFDSSolver(problem)
    # solver.solve()  # ← 이 부분이 실제 IFDS 분석을 수행
    
    # 임시: 수동으로 Seed 출력
    print("\n[Results] Initial taint seeds:")
    for inst, facts in seeds.items():
        print(f"  - Instruction: {inst}")
        for fact in facts:
            if fact.target != "ZERO":
                print(f"    → Taint: {fact.target} from {fact.source}")
    
    # TODO: Sink 검사
    # flows = problem.check_sinks(solver)
    # print(f"\n[Results] Detected {len(flows)} data leaks")
    
    print("\n[Done] Analysis framework is ready. Implement IDESolver worklist for full analysis.")
    return 0

if __name__ == '__main__':
    sys.exit(main())