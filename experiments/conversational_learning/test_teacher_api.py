#!/usr/bin/env python3
"""
Test Teacher API

Quick test to verify the LLM teacher API is working with proper role alternation.
"""

import sys
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.llm_teacher import LLMTeacher, TeacherConfig


def test_teacher_connection():
    """Test basic teacher connection and conversation start."""
    print("🧪 Testing Teacher API Connection")
    print("=" * 40)
    
    # Create teacher
    config = TeacherConfig(
        api_url="http://localhost:1234/v1/chat/completions",
        model="google/gemma-3-12b",
        temperature=0.7
    )
    
    teacher = LLMTeacher(config)
    
    print("1. Testing conversation start...")
    try:
        opening = teacher.start_conversation()
        print(f"✅ Teacher opening: '{opening}'")
    except Exception as e:
        print(f"❌ Start conversation failed: {e}")
        return False
    
    print("\n2. Testing student response...")
    try:
        response = teacher.respond_to_student("hi")
        print(f"✅ Teacher response: '{response}'")
    except Exception as e:
        print(f"❌ Student response failed: {e}")
        return False
    
    print("\n3. Testing another response...")
    try:
        response2 = teacher.respond_to_student("mama")
        print(f"✅ Teacher response 2: '{response2}'")
    except Exception as e:
        print(f"❌ Second response failed: {e}")
        return False
    
    print("\n✅ All teacher API tests passed!")
    return True


if __name__ == "__main__":
    success = test_teacher_connection()
    if success:
        print("\n🎉 Teacher API is working correctly!")
        print("You can now run the main demo safely.")
    else:
        print("\n❌ Teacher API has issues. Check your LLM setup.")
        exit(1)