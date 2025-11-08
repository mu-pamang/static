from typing import TypeVar, Generic, Any

# V는 Edge Function에서 처리하는 값의 타입을 나타내는 타입 변수입니다.
V = TypeVar('V')

# --- Heros 인터페이스 스텁 (참조용) ---
# Java의 heros.EdgeFunction<V> 인터페이스에 해당하는 기본 클래스입니다.
class EdgeFunction(Generic[V]):
    """
    Java의 heros.EdgeFunction<V> 인터페이스에 대한 개념적 Python 구현입니다.
    """
    def compute_target(self, source: V) -> V:
        raise NotImplementedError

    def compose_with(self, second_function: 'EdgeFunction[V]') -> 'EdgeFunction[V]':
        raise NotImplementedError

    def meet_with(self, other_function: 'EdgeFunction[V]') -> 'EdgeFunction[V]':
        raise NotImplementedError

    def equal_to(self, other: 'EdgeFunction[V]') -> bool:
        raise NotImplementedError

# --- AllTop 구현 ---

class AllTop(EdgeFunction[V]):
    """
    모든 입력 값을 미리 정의된 top element로 매핑하는 Edge Function입니다.
    IFDS/IDE 분석의 초기화 및 격자 연산에서 중요한 역할을 합니다.
    """
    
    def __init__(self, top_element: V):
        """
        AllTop 인스턴스를 생성하고 top element를 저장합니다.
        
        :param top_element: 이 함수가 항상 반환할 값입니다.
        """
        self._top_element: V = top_element

    def compute_target(self, source: V) -> V:
        """
        입력 'source'와 관계없이 항상 top element를 반환합니다.
        
        :param source: 입력 값 (무시됨).
        :return: top element.
        """
        return self._top_element

    def compose_with(self, second_function: 'EdgeFunction[V]') -> 'EdgeFunction[V]':
        """
        다른 함수와 합성합니다 (f_1.compose_with(f_2) == f_2(f_1(x))).
        AllTop(f1)이 항상 top을 반환하므로, 어떤 함수와 합성하든 결과는 AllTop입니다.
        
        :param second_function: 합성할 두 번째 Edge Function (무시됨).
        :return: AllTop 자신.
        """
        # Java 코드의 로직을 따름: AllTop.composeWith(f) = AllTop
        return self

    def meet_with(self, other_function: 'EdgeFunction[V]') -> 'EdgeFunction[V]':
        """
        다른 함수와 meet 연산을 수행합니다.
        
        :param other_function: meet 연산을 수행할 다른 Edge Function.
        :return: meet 연산의 결과 Edge Function.
        """
        # AllTop은 meet 격자의 가장 큰 원소이므로, 무엇과 meet하든 다른 함수가 결과가 됩니다.
        # Top ⊓ f = f
        return other_function

    def equal_to(self, other: 'EdgeFunction[V]') -> bool:
        """
        두 Edge Function이 동일한지 비교합니다.
        
        :param other: 비교할 다른 Edge Function.
        :return: 같으면 True, 아니면 False.
        """
        if isinstance(other, AllTop):
            # 두 AllTop 인스턴스가 동일한 top element를 사용하는지 확인합니다.
            return other._top_element == self._top_element
        return False
        
    def __eq__(self, other: Any) -> bool:
        """Python의 == 연산자 오버로드 (equalTo를 활용)"""
        if not isinstance(other, EdgeFunction):
            return NotImplemented
        return self.equal_to(other)

    def __hash__(self) -> int:
        """해시 가능하도록 구현"""
        return hash(("AllTop", self._top_element))

    def __str__(self) -> str:
        return "alltop"
    

"""\n===== ORIGINAL JAVA FOR REFERENCE =====\n/*******************************************************************************
 /*******************************************************************************
 * Copyright (c) 2012 Eric Bodden.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Lesser Public License v2.1
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 * 
 * Contributors:
 *     Eric Bodden - initial API and implementation
 ******************************************************************************/
package heros.edgefunc;

import heros.EdgeFunction;
import heros.solver.IFDSSolver;


/**
 * This is an internal class implementing an edge function for use in {@link IFDSSolver}.
 * This edge function sets everything to the top value.
 * 
 * @author Eric Bodden
 *
 * @param <V>
 */
public class AllTop<V> implements EdgeFunction<V> {
	
	private final V topElement; 

	public AllTop(V topElement){
		this.topElement = topElement;
	} 

	public V computeTarget(V source) {
		return topElement;
	}

	public EdgeFunction<V> composeWith(EdgeFunction<V> secondFunction) {
		return this;
	}

	public EdgeFunction<V> meetWith(EdgeFunction<V> otherFunction) {
		return otherFunction;
	}

	public boolean equalTo(EdgeFunction<V> other) {
		if(other instanceof AllTop) {
			@SuppressWarnings("rawtypes")
			AllTop allTop = (AllTop) other;
			return allTop.topElement.equals(topElement);
		}		
		return false;
	}

	public String toString() {
		return "alltop";
	}
	
}

\n"""