# RAG Evaluation System

Hệ thống đánh giá toàn diện cho mô hình RAG (Retrieval-Augmented Generation) bằng tiếng Việt.

## Tổng quan

Hệ thống này cung cấp nhiều phương pháp đánh giá khác nhau để đánh giá chất lượng của câu trả lời được tạo ra bởi mô hình RAG, bao gồm:

- **Faithfulness (Tính trung thực)**: Câu trả lời có nhất quán với ngữ cảnh được truy xuất không?
- **Relevance (Tính liên quan)**: Câu trả lời có liên quan đến câu hỏi không?
- **Context Relevance (Tính liên quan của ngữ cảnh)**: Tài liệu được truy xuất có liên quan đến câu hỏi không?
- **Completeness (Tính đầy đủ)**: Câu trả lời có giải quyết đầy đủ câu hỏi không?
- **Groundedness (Tính dựa trên cơ sở)**: Mọi tuyên bố trong câu trả lời có được hỗ trợ bởi ngữ cảnh không?

## Cấu trúc thư mục

```
LLM-as-a-judge/
├── __init__.py              # Lớp cơ sở và tiện ích chung
├── faithfulness.py          # Đánh giá tính trung thực
├── relevance.py             # Đánh giá tính liên quan
├── context_relevance.py     # Đánh giá tính liên quan của ngữ cảnh
├── completeness.py          # Đánh giá tính đầy đủ
├── groundedness.py          # Đánh giá tính dựa trên cơ sở
├── evaluation_runner.py     # Hệ thống đánh giá tổng hợp
├── example_usage.py         # Ví dụ sử dụng
└── README.md               # Tài liệu này
```

## Cách sử dụng

### Đánh giá đơn lẻ

```python
from benchmark.eval.LLM-as-a-judge.evaluation_runner import evaluate_rag_output

# Đánh giá một câu trả lời RAG
result = evaluate_rag_output(
    question="Làm thế nào để đăng ký tài khoản?",
    answer="Để đăng ký, truy cập trang đăng ký và điền thông tin.",
    context="Hướng dẫn đăng ký: Truy cập website → Nhấp đăng ký → Điền form",
    methods=["faithfulness_llm", "relevance_llm"]  # Chỉ định phương pháp đánh giá
)

print(f"Overall score: {result['summary']['overall_score']}")
```

### Đánh giá toàn diện

```python
from benchmark.eval.LLM-as-a-judge.evaluation_runner import RAGEvaluator
from benchmark.eval.LLM-as-a-judge import create_evaluation_input

# Tạo dữ liệu đánh giá
input_data = create_evaluation_input(
    question="Chi phí dịch vụ là bao nhiêu?",
    answer="Dịch vụ cơ bản có giá 50,000 VND/tháng.",
    context="Bảng giá: Cơ bản 50k, Premium 150k, Enterprise 500k VND/tháng.",
    retrieved_docs=[{"page_content": "Chi tiết bảng giá..."}]
)

# Chạy đánh giá toàn diện
evaluator = RAGEvaluator()
results = evaluator.evaluate(input_data)

# Xem kết quả
print(f"Tổng điểm: {results['summary']['overall_score']:.3f}")
for category, data in results['summary']['category_scores'].items():
    print(f"{category}: {data['score']:.3f}")
```

### Đánh giá hàng loạt

```python
from benchmark.eval.LLM-as-a-judge.evaluation_runner import BatchEvaluator

batch_evaluator = BatchEvaluator()
batch_results = batch_evaluator.evaluate_batch(
    evaluation_inputs=[input_data1, input_data2, input_data3],
    save_results=True,
    output_file="evaluation_report.json"
)
```

## Phương pháp đánh giá chi tiết

### 1. Faithfulness (Tính trung thực)
- **LLM-based**: Sử dụng LLM để đánh giá tính nhất quán
- **Semantic**: Trích xuất facts và kiểm chứng với ngữ cảnh

### 2. Relevance (Tính liên quan)
- **LLM-based**: Đánh giá mức độ liên quan đến câu hỏi
- **Semantic**: Tương đồng ngữ nghĩa giữa câu hỏi và câu trả lời
- **Keyword**: Phù hợp từ khóa và loại câu hỏi

### 3. Context Relevance (Tính liên quan của ngữ cảnh)
- **LLM-based**: Đánh giá ngữ cảnh với câu hỏi
- **Document-level**: Đánh giá từng tài liệu riêng lẻ
- **Coverage**: Mức độ bao phủ các khía cạnh của câu hỏi

### 4. Completeness (Tính đầy đủ)
- **LLM-based**: Đánh giá mức độ đầy đủ
- **Aspect-based**: Kiểm tra từng khía cạnh của câu hỏi
- **Depth**: Đánh giá độ sâu thông tin

### 5. Groundedness (Tính dựa trên cơ sở)
- **LLM-based**: Kiểm chứng từng tuyên bố
- **Claim verification**: Trích xuất và xác minh claims
- **Factual consistency**: Kiểm tra mâu thuẫn thông tin

## Kết quả đánh giá

Mỗi phương pháp đánh giá trả về:
- **score**: Điểm số từ 0.0 đến 1.0
- **reasoning**: Giải thích chi tiết về đánh giá
- **metadata**: Thông tin bổ sung (số lượng claims, từ khóa, v.v.)

## Tích hợp với API

Hệ thống có thể tích hợp với API chat hiện tại:

```python
from benchmark.eval.LLM-as-a-judge.evaluation_runner import evaluate_from_chat_response

# Sau khi nhận response từ API chat
evaluation = evaluate_from_chat_response(
    chat_request={"question": user_question},
    chat_response=chat_response,
    retrieved_docs=retrieved_documents  # Nếu có
)
```

## Yêu cầu hệ thống

- Python 3.8+
- LangChain OpenAI
- sentence-transformers (tùy chọn, cho semantic similarity)
- numpy (tùy chọn)

## Chạy ví dụ

```bash
cd benchmark/eval/LLM-as-a-judge
python example_usage.py
```

## Mở rộng hệ thống

Để thêm phương pháp đánh giá mới:

1. Tạo class kế thừa `BaseEvaluator`
2. Implement phương thức `evaluate()`
3. Thêm vào `RAGEvaluator.evaluators` trong `evaluation_runner.py`

## Lưu ý

- Một số phương pháp sử dụng LLM (OpenAI) và có thể phát sinh chi phí
- Các phương pháp semantic yêu cầu thư viện sentence-transformers
- Hệ thống được tối ưu hóa cho tiếng Việt
