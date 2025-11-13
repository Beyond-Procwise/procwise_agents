import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.model_selector import RAGPipeline


def _make_pipeline() -> RAGPipeline:
    return RAGPipeline.__new__(RAGPipeline)


def test_postprocess_answer_structures_numbered_list():
    pipeline = _make_pipeline()
    raw = (
        "Thanks for the question—happy to help clarify what we mean by 'expense policy.'. "
        "Simply put, an expense policy is a formal set of guidelines that defines how employees, contractors, "
        "and approved representatives can claim business-related expenses. It ensures consistency, accountability, "
        "and compliance across the organization. According to the current policy, the Expense Policy is owned by the "
        "Chief Procurement Officer and approved by the Executive Finance Committee, with an effective date of July 14, "
        "2025, and a review cycle set annually or as needed. The policy applies to all Group Employees, Contractors, "
        "and Approved Representatives. Key elements include: 1. Eligible Expenses: Only business-related costs incurred "
        "during work activities are reimbursable. 2. Documentation Requirements: All claims must be supported by valid "
        "receipts and clear justification. 3. Prohibited Expenses: Certain personal or non-work-related costs (e.g., "
        "entertainment, fines, or luxury items) are explicitly excluded—details on these are outlined in the 'Business "
        "Expenses That Cannot Be Claimed' section. 4. Approval & Oversight: Spending limits and delegation of authority "
        "are governed by separate policies, ensuring proper financial controls. If you're looking to understand what’s "
        "not allowed or need help with a specific claim, I can pull up the full list of exclusions or assist with a sample "
        "submission."
    )

    expected = (
        "Thanks for the question—happy to help clarify what we mean by 'expense policy.'. Simply put, an expense policy "
        "is a formal set of guidelines that defines how employees, contractors, and approved representatives can claim "
        "business-related expenses. It ensures consistency, accountability, and compliance across the organization.\n\n"
        "According to the current policy, the Expense Policy is owned by the Chief Procurement Officer and approved by the "
        "Executive Finance Committee, with an effective date of July 14, 2025, and a review cycle set annually or as needed. "
        "The policy applies to all Group Employees, Contractors, and Approved Representatives. Key elements include:\n\n"
        "1. Eligible Expenses: Only business-related costs incurred during work activities are reimbursable.\n\n"
        "2. Documentation Requirements: All claims must be supported by valid receipts and clear justification.\n\n"
        "3. Prohibited Expenses: Certain personal or non-work-related costs (e.g., entertainment, fines, or luxury items) are "
        "explicitly excluded—details on these are outlined in the 'Business Expenses That Cannot Be Claimed' section.\n\n"
        "4. Approval & Oversight: Spending limits and delegation of authority are governed by separate policies, ensuring "
        "proper financial controls.\n\n"
        "If you're looking to understand what’s not allowed or need help with a specific claim, I can pull up the full list of "
        "exclusions or assist with a sample submission."
    )

    result = pipeline._postprocess_answer(raw)

    assert result == expected
