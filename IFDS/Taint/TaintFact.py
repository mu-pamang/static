# TaintFact.py
# 역할: IFDS가 추적할 데이터-플로우 사실(D)의 구체적인 구현체

class TaintFact:
    """
    IFDS가 추적할 하나의 오염된 사실(Data-Flow Fact, D)을 나타내는 Python 클래스.
    """
    
    def __init__(self, target: str, source_api: str):
        # target: 'vN' (레지스터) 또는 'Lcls;->fld:type' (필드)
        self.target = target  
        # source_api: 'Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;'
        self.source = source_api

    def __hash__(self):
        """IFDS Set 연산을 위한 필수 정의: target과 source가 같으면 동일 Fact."""
        return hash((self.target, self.source))

    def __eq__(self, other):
        """IFDS Set 연산을 위한 필수 정의: target과 source가 모두 일치해야 동일 Fact."""
        if not isinstance(other, TaintFact):
            return False
        return self.target == other.target and self.source == other.source

    def __repr__(self):
        return f"D({self.target}, source={self.source})"