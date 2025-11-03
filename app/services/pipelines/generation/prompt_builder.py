from langchain_core.prompts import ChatPromptTemplate


def create_answer_prompt():
    """Create prompt template for answer generation"""
    return ChatPromptTemplate.from_messages([
        ("system", "Bạn là trợ lý nội bộ của doanh nghiệp. Trả lời bằng tiếng Việt, súc tích, dựa trên ngữ cảnh. Nếu thông tin không có trong ngữ cảnh, hãy nói bạn không chắc và gợi ý hỏi thêm."),
        ("human", "Câu hỏi: {question}\n\nNgữ cảnh:\n{context}\n\nHãy trả lời rõ ràng cho nhân viên."),
    ])


def calculate_confidence(num_docs: int) -> float:
    """Calculate confidence score based on number of retrieved documents"""
    return min(0.95, 0.5 + 0.1 * num_docs)
