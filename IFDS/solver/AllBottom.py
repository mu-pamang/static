from typing import TypeVar, Generic, Any, Optional
from Core.EdgeFunction import EdgeFunction
from .EdgeIdentity import EdgeIdentity

# V는 Edge Function에서 처리하는 값의 타입을 나타내는 타입 변수입니다 (Java의 <V>와 유사).
V = TypeVar('V')

# --- Heros 인터페이스 스텁 (참조용) ---
# Java의 EdgeFunction 인터페이스에 해당하는 기본 클래스입니다.
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

# EdgeIdentity 및 AllTop에 대한 스텁 (AllBottom에서 참조하므로 필요)
class EdgeIdentity(EdgeFunction[V]):
    pass

class AllTop(EdgeFunction[V]):
    pass

# --- AllBottom 구현 ---

class AllBottom(EdgeFunction[V]):
    """
    모든 입력 값을 미리 정의된 bottom element로 매핑하는 Edge Function입니다.
    IFDSSolver에서 도달 가능성을 모델링하는 데 주로 사용됩니다.
    """
    
    def __init__(self, bottom_element: V):
        """
        AllBottom 인스턴스를 생성하고 bottom element를 저장합니다.
        
        :param bottom_element: 이 함수가 항상 반환할 값입니다.
        """
        self._bottom_element: V = bottom_element

    def compute_target(self, source: V) -> V:
        """
        입력 'source'와 관계없이 항상 bottom element를 반환합니다.
        
        :param source: 입력 값 (무시됨).
        :return: bottom element.
        """
        return self._bottom_element

    def compose_with(self, second_function: 'EdgeFunction[V]') -> 'EdgeFunction[V]':
        """
        다른 함수와 합성합니다 (f_1.compose_with(f_2) == f_2(f_1(x))).
        AllBottom(f1)이 먼저 적용되므로, 결과는 일반적으로 f2입니다.
        
        :param second_function: 합성할 두 번째 Edge Function.
        :return: 합성 결과 Edge Function.
        """
        # Java 코드의 로직을 따름:
        # AllBottom(f1)은 항상 bottom을 반환하므로, f2가 무엇이든 f2(bottom)이 결과입니다.
        # IFDS/IDE의 특성상 EdgeIdentity와의 특별한 상호작용이 정의되어 있습니다.
        if isinstance(second_function, EdgeIdentity):
            return self # AllBottom.composeWith(EdgeIdentity) = AllBottom
        return second_function # AllBottom.composeWith(f) = f

    def meet_with(self, other_function: 'EdgeFunction[V]') -> 'EdgeFunction[V]':
        """
        다른 함수와 meet 연산을 수행합니다.
        
        :param other_function: meet 연산을 수행할 다른 Edge Function.
        :return: meet 연산의 결과 Edge Function.
        """
        from .EdgeIdentity import EdgeIdentity
        from .AllTop import AllTop
        # AllBottom은 meet 격자의 가장 작은 원소이므로, 무엇과 meet하든 AllBottom이 결과입니다.
        if other_function is self or self.equal_to(other_function):
            return self
        
        # AllTop, EdgeIdentity와의 meet 연산은 AllBottom이 됩니다.
        if isinstance(other_function, (AllTop, EdgeIdentity)):
            return self
            
        # Java와 마찬가지로, 정의되지 않은 다른 유형의 함수는 예외를 발생시킵니다.
        raise ValueError(f"Unexpected edge function type for meet: {type(other_function).__name__}")

    def equal_to(self, other: 'EdgeFunction[V]') -> bool:
        """
        두 Edge Function이 동일한지 비교합니다.
        
        :param other: 비교할 다른 Edge Function.
        :return: 같으면 True, 아니면 False.
        """
        if isinstance(other, AllBottom):
            # 두 AllBottom 인스턴스가 동일한 bottom element를 사용하는지 확인합니다.
            return other._bottom_element == self._bottom_element
        return False
        
    def __eq__(self, other: Any) -> bool:
        """Python의 == 연산자 오버로드 (equalTo를 활용)"""
        if not isinstance(other, EdgeFunction):
            return NotImplemented
        return self.equal_to(other)

    def __hash__(self) -> int:
        """해시 가능하도록 구현"""
        return hash(("AllBottom", self._bottom_element))

    def __str__(self) -> str:
        return "allbottom"


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
 * This class implements an edge function that maps every input to the stated bottom element. This class is normally useful only
 * in the context of an {@link IFDSSolver}, which uses AllBottom to model reachability. Consequently, this class should normally not be useful
 * in the context of custom IDE problems.
 * 
 * @author Eric Bodden
 *
 * @param <V>
 */
public class AllBottom<V> implements EdgeFunction<V> {
	
	private final V bottomElement;

	public AllBottom(V bottomElement){
		this.bottomElement = bottomElement;
	} 

	public V computeTarget(V source) {
		return bottomElement;
	}

	public EdgeFunction<V> composeWith(EdgeFunction<V> secondFunction) {
		//note: this only makes sense within IFDS, see here:
		//https://github.com/Sable/heros/issues/37
		if (secondFunction instanceof EdgeIdentity)
			return this;
		return secondFunction;
	}

	public EdgeFunction<V> meetWith(EdgeFunction<V> otherFunction) {
		if(otherFunction == this || otherFunction.equalTo(this)) return this;
		if(otherFunction instanceof AllTop) {
			return this;
		}
		if(otherFunction instanceof EdgeIdentity) {
			return this;
		}
		throw new IllegalStateException("unexpected edge function: "+otherFunction);
	}

	public boolean equalTo(EdgeFunction<V> other) {
		if(other instanceof AllBottom) {
			@SuppressWarnings("rawtypes")
			AllBottom allBottom = (AllBottom) other;
			return allBottom.bottomElement.equals(bottomElement);
		}		
		return false;
	}
	
	public String toString() {
		return "allbottom";
	}

}

\n"""