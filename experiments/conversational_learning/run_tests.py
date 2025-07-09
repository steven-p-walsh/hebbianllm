#!/usr/bin/env python3
"""
Test runner for all phase implementations.

Runs all phase tests in sequence to verify the complete system.
"""

import sys
import subprocess
from pathlib import Path

def run_test(test_file):
    """Run a test file and return success status."""
    test_path = Path(__file__).parent / "tests" / test_file
    
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, str(test_path)], 
                              capture_output=False, 
                              text=True, 
                              cwd=test_path.parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False

def main():
    """Run all phase tests."""
    test_files = [
        "test_phase1_dopamine.py",
        "test_phase2_attention.py", 
        "test_phase3_fatigue.py",
        "test_phase4_replay.py",
        "test_phase5_optimization.py"
    ]
    
    print("🧪 Running All Phase Tests")
    print("=" * 60)
    
    results = {}
    
    for test_file in test_files:
        success = run_test(test_file)
        results[test_file] = success
        
        if success:
            print(f"✅ {test_file} PASSED")
        else:
            print(f"❌ {test_file} FAILED")
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_file:30s} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The neuro-inspired enhancement system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())