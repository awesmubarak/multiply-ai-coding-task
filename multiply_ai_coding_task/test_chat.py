import pytest
from unittest.mock import patch
from datetime import date
from multiply_ai_coding_task.factfind import (
    User,
    Goal,
    GoalType,
    NewHomeGoalInformation,
    NewCarInformation,
    OtherGoalInformation
)
from multiply_ai_coding_task.chat import (  # Update to match your actual module structure
    ConversationState,
    ExtractedInformation,
    chat_response,
    _skeleton,
    _cleanup_json,
    _merge
)

@pytest.fixture
def base_state():
    return ConversationState()

# Unit tests for helper functions
def test_skeleton_serialization():
    info = ExtractedInformation()
    info.user.first_name = "John"
    result = _skeleton(info)
    assert '"first_name": "John"' in result


def test_cleanup_json_invalid():
    txt = """```json
    {"key": "value"
    ```"""
    assert _cleanup_json(txt) is None

# Test merging functionality
def test_merge_user_fields():
    info = ExtractedInformation()
    payload = {"first_name": "Alice", "email": "alice@example.com"}
    _merge(info, payload)
    assert info.user.first_name == "Alice"
    assert info.user.email == "alice@example.com"

def test_merge_new_home_goal():
    info = ExtractedInformation()
    payload = {
        "goal_type": "new_home",
        "goal_name": "Country house",
        "location": "London",
        "house_price": "500000",
        "deposit_amount": "100000",
        "purchase_date": "2025-12-01"
    }
    _merge(info, payload)

    goal = info.user.goals[0]
    assert goal.goal_type == GoalType.NEW_HOME
    assert goal.goal_specific_information.location == "London"
    assert goal.goal_specific_information.house_price == 500000.0
    assert goal.goal_specific_information.purchase_date == date(2025, 12, 1)

def test_merge_invalid_goal_type():
    info = ExtractedInformation()
    payload = {
        "goal_type": "invalid_goal",
        "goal_name": "Invalid Goal"
    }
    _merge(info, payload)
    assert len(info.user.goals) == 0  # No goal should be added

# Integration tests with mocked LLM
@patch("multiply_ai_coding_task.chat.llm")  # Update to match your module structure
def test_basic_info_collection(mock_llm, base_state):
    mock_llm.return_value = """```json
    {
        "assistant": "Great! Let&#x27;s start with your personal details...",
        "extracted": {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john@example.com"
        },
        "finished": false
    }
    ```"""

    updated = chat_response(base_state)
    assert updated.extracted_information.user.first_name == "John"
    assert updated.extracted_information.user.last_name == "Doe"
    assert updated.extracted_information.user.email == "john@example.com"
    assert len(updated.new_messages) == 1
    assert not updated.finished

@patch("multiply_ai_coding_task.chat.llm")  # Update to match your module structure
def test_full_goal_flow(mock_llm, base_state):
    # First response - personal info
    mock_llm.side_effect = [
        """```json
        {
            "assistant": "Thanks! Now about your goals...",
            "extracted": {
                "first_name": "Sarah",
                "date_of_birth": "1990-05-15"
            },
            "finished": false
        }
        ```""",
        """```json
        {
            "assistant": "Got your home goal!",
            "extracted": {
                "goal_type": "new_home",
                "goal_name": "London Flat",
                "location": "London",
                "house_price": 750000,
                "deposit_amount": 150000,
                "purchase_date": "2026-06-01"
            },
            "finished": true
        }
        ```"""
    ]

    # First chat iteration
    state1 = chat_response(base_state)
    assert state1.extracted_information.user.first_name == "Sarah"
    assert state1.extracted_information.user.date_of_birth == '1990-05-15'

    # Second chat iteration
    state2 = chat_response(state1)
    assert len(state2.extracted_information.user.goals) == 1
    goal = state2.extracted_information.user.goals[0]
    assert goal.goal_type == GoalType.NEW_HOME
    assert goal.goal_specific_information.location == "London"
    assert goal.goal_specific_information.house_price == 750000.0
    assert goal.goal_specific_information.deposit_amount == 150000.0
    assert goal.goal_specific_information.purchase_date == date(2026, 6, 1)
    assert state2.finished

# Edge cases
def test_numeric_parsing():
    info = ExtractedInformation()
    payload = {
        "goal_type": "new_car",
        "car_price": "£35,000",
        "purchase_date": "2025-07-01"
    }
    _merge(info, payload)
    car_info = info.user.goals[0].goal_specific_information
    assert car_info.car_price == 35000.0
