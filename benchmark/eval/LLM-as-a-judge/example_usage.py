"""
Example Usage of RAG Evaluation System

This script demonstrates how to use the RAG evaluation system
with sample data and shows the evaluation results.
"""

from .evaluation_runner import evaluate_rag_output, RAGEvaluator
from . import create_evaluation_input


def run_evaluation_examples():
    """Run example evaluations to demonstrate the system"""

    print("🚀 RAG Evaluation System - Example Usage\n")

    # Example 1: Good RAG response
    print("📝 Example 1: High-quality RAG response")
    print("-" * 50)

    good_input = create_evaluation_input(
        question="Làm thế nào để đăng ký tài khoản trên hệ thống?",
        answer="Để đăng ký tài khoản, bạn cần:\n1. Truy cập trang đăng ký trên website\n2. Điền đầy đủ thông tin cá nhân bao gồm họ tên, email, và mật khẩu\n3. Xác nhận địa chỉ email bằng cách nhấp vào liên kết được gửi\n4. Đăng nhập bằng tài khoản mới tạo",
        context="Hướng dẫn đăng ký tài khoản:\n- Truy cập website và nhấp 'Đăng ký'\n- Điền thông tin: họ tên, email, mật khẩu\n- Xác nhận email để kích hoạt tài khoản\n- Sau khi kích hoạt, có thể đăng nhập bình thường",
        retrieved_docs=[
            {"page_content": "Quy trình đăng ký tài khoản gồm 4 bước chính..."},
            {"page_content": "Hệ thống yêu cầu xác nhận email để bảo mật..."}
        ]
    )

    results = evaluate_rag_output(
        question=good_input.question,
        answer=good_input.answer,
        context=good_input.context,
        retrieved_docs=good_input.retrieved_docs,
        methods=["faithfulness_llm", "relevance_llm", "completeness_llm", "groundedness_llm"]
    )

    print_evaluation_results(results)

    # Example 2: Poor RAG response
    print("\n📝 Example 2: Low-quality RAG response")
    print("-" * 50)

    poor_input = create_evaluation_input(
        question="Chi phí của dịch vụ premium là bao nhiêu?",
        answer="Dịch vụ này rất tốt và đáng tin cậy.",
        context="Bảng giá dịch vụ:\n- Cơ bản: 50,000 VND/tháng\n- Premium: 150,000 VND/tháng\n- Enterprise: 500,000 VND/tháng\nTất cả giá đã bao gồm VAT.",
        retrieved_docs=[
            {"page_content": "Chi tiết bảng giá các gói dịch vụ..."}
        ]
    )

    results = evaluate_rag_output(
        question=poor_input.question,
        answer=poor_input.answer,
        context=poor_input.context,
        retrieved_docs=poor_input.retrieved_docs,
        methods=["faithfulness_llm", "relevance_llm", "completeness_llm", "groundedness_llm"]
    )

    print_evaluation_results(results)

    # Example 3: Comprehensive evaluation
    print("\n📊 Example 3: Full evaluation suite")
    print("-" * 50)

    comprehensive_evaluator = RAGEvaluator()
    full_results = comprehensive_evaluator.evaluate(good_input)

    print(f"Overall Score: {full_results['summary']['overall_score']:.3f}")
    print("\nDetailed Category Scores:")
    for category, data in full_results['summary']['category_scores'].items():
        print(f"  {category}: {data['score']:.3f} ({data['methods_used']}/{data['methods_available']} methods)")

    print(f"\nEvaluation completed in {full_results['metadata']['evaluation_time']:.2f} seconds")
    print(f"Methods run: {full_results['metadata']['methods_run']}")


def print_evaluation_results(results: dict):
    """Print evaluation results in a readable format"""
    print(f"Overall Score: {results['summary']['overall_score']:.3f}")

    print("\nCategory Breakdown:")
    for category, data in results['summary']['category_scores'].items():
        print(f"  {category}: {data['score']:.3f}")

    print(f"\nSuccessful evaluations: {results['metadata']['methods_run']}")
    print(f"Evaluation time: {results['metadata']['evaluation_time']:.2f} seconds")


def create_sample_evaluation_data():
    """Create sample data for testing different scenarios"""

    samples = [
        {
            "name": "Perfect Answer",
            "question": "Quy trình thanh toán như thế nào?",
            "answer": "Quy trình thanh toán gồm 3 bước: 1) Chọn sản phẩm, 2) Nhập thông tin thanh toán, 3) Xác nhận giao dịch.",
            "context": "Quy trình thanh toán: Chọn sản phẩm → Nhập thông tin thẻ → Xác nhận → Hoàn tất.",
            "expected_score": 0.9
        },
        {
            "name": "Incomplete Answer",
            "question": "Làm thế nào để khôi phục mật khẩu?",
            "answer": "Bạn có thể khôi phục mật khẩu.",
            "context": "Khôi phục mật khẩu: Nhấp 'Quên mật khẩu' → Nhập email → Nhận mã OTP → Đặt mật khẩu mới.",
            "expected_score": 0.3
        },
        {
            "name": "Irrelevant Answer",
            "question": "Giờ làm việc của hỗ trợ khách hàng?",
            "answer": "Sản phẩm của chúng tôi có nhiều màu sắc đẹp.",
            "context": "Hỗ trợ khách hàng: 8:00-18:00 từ thứ 2 đến thứ 6.",
            "expected_score": 0.1
        }
    ]

    return samples


if __name__ == "__main__":
    run_evaluation_examples()
