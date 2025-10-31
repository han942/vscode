"""
Gemma 모델을 이용한 RouteLLM 기반 Router 구현
CPU 환경에서 실행 가능한 경량 모델 사용
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from routellm.controller import Controller
from routellm.routers.routers import Router
import numpy as np
from typing import List, Dict, Optional
import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaRouter:
    """
    Gemma 모델을 이용한 Router 시스템
    - Weak Model: gemma-2-2b-it (2B)
    - Strong Model: gemma-2-9b-it (9B)
    """
    
    def __init__(
        self,
        weak_model_name: str = "google/gemma-2-2b-it",
        strong_model_name: str = "google/gemma-2-9b-it",
        router_type: str = "mf",  # 'mf', 'bert', 'sw_ranking', 'causal_llm', 'random'
        device: str = "cpu",
        use_4bit: bool = True,  # CPU에서는 False 권장
    ):
        self.weak_model_name = weak_model_name
        self.strong_model_name = strong_model_name
        self.router_type = router_type
        self.device = device
        self.use_4bit = use_4bit and torch.cuda.is_available()
        
        logger.info(f"Initializing GemmaRouter")
        logger.info(f"Weak Model: {weak_model_name}")
        logger.info(f"Strong Model: {strong_model_name}")
        logger.info(f"Router Type: {router_type}")
        logger.info(f"Device: {device}")
        
        # RouteLLM Controller 초기화
        self._init_controller()
        
        # 로컬 모델 로드 (필요시)
        self.weak_model = None
        self.strong_model = None
        self.tokenizer = None
        
    def _init_controller(self):
        """RouteLLM Controller 초기화"""
        try:
            # Ollama를 통한 로컬 모델 라우팅 설정
            self.controller = Controller(
                routers=[self.router_type],
                strong_model=f"ollama/{self.strong_model_name}",
                weak_model=f"ollama/{self.weak_model_name}",
            )
            logger.info("RouteLLM Controller initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize RouteLLM Controller: {e}")
            logger.info("Will use custom implementation")
            self.controller = None
    
    def load_local_models(self):
        """로컬에서 Gemma 모델 직접 로드 (Ollama 없이 사용할 경우)"""
        logger.info("Loading local models...")
        
        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.weak_model_name)
        
        # 4-bit quantization 설정 (GPU 사용시)
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        # Weak Model 로드
        logger.info(f"Loading weak model: {self.weak_model_name}")
        self.weak_model = AutoModelForCausalLM.from_pretrained(
            self.weak_model_name,
            quantization_config=quantization_config,
            device_map=self.device if self.device != "cpu" else None,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Strong Model 로드 (필요시)
        logger.info(f"Loading strong model: {self.strong_model_name}")
        self.strong_model = AutoModelForCausalLM.from_pretrained(
            self.strong_model_name,
            quantization_config=quantization_config,
            device_map=self.device if self.device != "cpu" else None,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            low_cpu_mem_usage=True,
        )
        
        logger.info("Models loaded successfully")
    
    def estimate_query_complexity(self, query: str) -> float:
        """
        쿼리의 복잡도를 추정하여 0~1 사이의 점수 반환
        높을수록 강력한 모델이 필요함을 의미
        """
        # 간단한 휴리스틱 기반 복잡도 추정
        complexity_score = 0.0
        
        # 1. 길이 기반 (긴 쿼리는 복잡할 가능성)
        length_score = min(len(query) / 500, 1.0) * 0.2
        complexity_score += length_score
        
        # 2. 키워드 기반
        complex_keywords = [
            'explain', 'analyze', 'compare', 'evaluate', 'synthesize',
            'code', 'algorithm', 'mathematics', 'proof', 'theory',
            'complex', 'detailed', 'comprehensive', 'in-depth'
        ]
        simple_keywords = [
            'what is', 'who is', 'when', 'where', 'list',
            'simple', 'quick', 'brief', 'short'
        ]
        
        query_lower = query.lower()
        for keyword in complex_keywords:
            if keyword in query_lower:
                complexity_score += 0.15
        
        for keyword in simple_keywords:
            if keyword in query_lower:
                complexity_score -= 0.1
        
        # 3. 질문 유형 (수식, 코드 등)
        if any(char in query for char in ['=', '+', '*', '/', '^', '∫', '∑']):
            complexity_score += 0.2
        
        if any(keyword in query_lower for keyword in ['```', 'def ', 'class ', 'function']):
            complexity_score += 0.25
        
        # 0~1 범위로 정규화
        complexity_score = max(0.0, min(1.0, complexity_score))
        
        return complexity_score
    
    def route_query(
        self,
        query: str,
        threshold: float = 0.5,
        use_routellm: bool = True
    ) -> Dict[str, any]:
        """
        쿼리를 분석하여 적절한 모델로 라우팅
        
        Args:
            query: 사용자 질문
            threshold: 라우팅 임계값 (0~1)
            use_routellm: RouteLLM 프레임워크 사용 여부
        
        Returns:
            라우팅 결과 딕셔너리
        """
        if use_routellm and self.controller is not None:
            # RouteLLM을 통한 라우팅
            try:
                response = self.controller.chat.completions.create(
                    model=f"router-{self.router_type}-{threshold}",
                    messages=[{"role": "user", "content": query}]
                )
                return {
                    "routed_model": "strong" if response.model == self.strong_model_name else "weak",
                    "response": response.choices[0].message.content,
                    "method": "routellm"
                }
            except Exception as e:
                logger.warning(f"RouteLLM routing failed: {e}, using fallback")
        
        # 휴리스틱 기반 라우팅
        complexity = self.estimate_query_complexity(query)
        routed_model = "strong" if complexity > threshold else "weak"
        
        return {
            "routed_model": routed_model,
            "complexity_score": complexity,
            "threshold": threshold,
            "method": "heuristic"
        }
    
    def generate_response(
        self,
        query: str,
        model_choice: str = "weak",
        max_length: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        선택된 모델로 응답 생성
        """
        if self.weak_model is None or self.strong_model is None:
            raise RuntimeError("Models not loaded. Call load_local_models() first.")
        
        model = self.strong_model if model_choice == "strong" else self.weak_model
        
        # 입력 준비
        messages = [{"role": "user", "content": query}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        if self.device != "cpu":
            inputs = inputs.to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 디코딩
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 입력 제거 (응답만 추출)
        response = response.split("model\n")[-1].strip()
        
        return response


class RouterEvaluator:
    """Router 성능 평가 클래스"""
    
    def __init__(self, router: GemmaRouter):
        self.router = router
        self.results = []
    
    def evaluate_queries(
        self,
        queries: List[str],
        ground_truth_complexity: Optional[List[str]] = None,
        thresholds: List[float] = [0.3, 0.5, 0.7]
    ) -> Dict:
        """
        여러 쿼리에 대해 라우팅 평가
        
        Args:
            queries: 평가할 쿼리 리스트
            ground_truth_complexity: 실제 복잡도 레이블 ('simple', 'complex')
            thresholds: 테스트할 임계값 리스트
        """
        results = {
            'queries': [],
            'threshold_analysis': {}
        }
        
        for threshold in thresholds:
            correct = 0
            strong_count = 0
            weak_count = 0
            
            for i, query in enumerate(queries):
                routing_result = self.router.route_query(
                    query,
                    threshold=threshold,
                    use_routellm=False
                )
                
                routed_model = routing_result['routed_model']
                complexity = routing_result['complexity_score']
                
                if routed_model == 'strong':
                    strong_count += 1
                else:
                    weak_count += 1
                
                # Ground truth와 비교 (있는 경우)
                if ground_truth_complexity:
                    expected = ground_truth_complexity[i]
                    is_correct = (
                        (expected == 'complex' and routed_model == 'strong') or
                        (expected == 'simple' and routed_model == 'weak')
                    )
                    if is_correct:
                        correct += 1
                
                results['queries'].append({
                    'query': query,
                    'threshold': threshold,
                    'routed_to': routed_model,
                    'complexity_score': complexity
                })
            
            results['threshold_analysis'][threshold] = {
                'strong_model_pct': strong_count / len(queries) * 100,
                'weak_model_pct': weak_count / len(queries) * 100,
                'accuracy': correct / len(queries) * 100 if ground_truth_complexity else None
            }
        
        return results
    
    def print_evaluation_report(self, results: Dict):
        """평가 결과 리포트 출력"""
        print("\n" + "="*60)
        print("Router Evaluation Report")
        print("="*60)
        
        print("\nThreshold Analysis:")
        for threshold, stats in results['threshold_analysis'].items():
            print(f"\nThreshold: {threshold}")
            print(f"  Strong Model: {stats['strong_model_pct']:.1f}%")
            print(f"  Weak Model: {stats['weak_model_pct']:.1f}%")
            if stats['accuracy'] is not None:
                print(f"  Accuracy: {stats['accuracy']:.1f}%")
        
        print("\n" + "="*60)


# 사용 예시
if __name__ == "__main__":
    # 1. Router 초기화
    router = GemmaRouter(
        weak_model_name="google/gemma-2-2b-it",
        strong_model_name="google/gemma-2-9b-it",
        router_type="mf",
        device="cpu",
        use_4bit=False  # CPU 사용
    )
    
    # 2. 테스트 쿼리
    test_queries = [
        "What is the capital of France?",  # Simple
        "Explain the concept of quantum entanglement and its implications for quantum computing.",  # Complex
        "How do I boil an egg?",  # Simple
        "Implement a binary search tree in Python with insertion, deletion, and balancing.",  # Complex
        "What time is it?",  # Simple
        "Compare and contrast the philosophical approaches of Kant and Hegel regarding metaphysics.",  # Complex
    ]
    
    ground_truth = ["simple", "complex", "simple", "complex", "simple", "complex"]
    
    # 3. Router 평가
    evaluator = RouterEvaluator(router)
    results = evaluator.evaluate_queries(
        test_queries,
        ground_truth_complexity=ground_truth,
        thresholds=[0.3, 0.5, 0.7]
    )
    
    # 4. 결과 출력
    evaluator.print_evaluation_report(results)
    
    # 5. 개별 쿼리 라우팅 테스트
    print("\n" + "="*60)
    print("Individual Query Routing Examples:")
    print("="*60)
    
    for query in test_queries[:3]:
        result = router.route_query(query, threshold=0.5, use_routellm=False)
        print(f"\nQuery: {query}")
        print(f"Routed to: {result['routed_model'].upper()} model")
        print(f"Complexity Score: {result['complexity_score']:.3f}")
    
    # 6. 실제 응답 생성 (모델 로드 후)
    # Uncomment to test actual model inference
    # router.load_local_models()
    # response = router.generate_response("What is machine learning?", model_choice="weak")
    # print(f"\nResponse: {response}")