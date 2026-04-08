import pytest
import sys
import os

def main():
    """Run all tests using pytest."""
    # Ensure the app directory is in the path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
    
    print("="*60)
    print("🚀 Starting Research RAG Test Suite")
    print("="*60)
    
    # Run pytest with coverage and verbose output
    # Note: We use -v for verbosity and -p no:warnings to keep output clean
    exit_code = pytest.main([
        "-v",
        "tests",
        "--tb=short",
        "-p", "no:warnings"
    ])
    
    if exit_code == 0:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(f"❌ SOME TESTS FAILED (Exit code: {exit_code})")
        print("="*60)
        
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
