"""
CLI script for running BOP-JAX tests and examples.
"""

import sys
import os
from pathlib import Path


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BOP-JAX Testing and Examples")
    parser.add_argument(
        "command",
        choices=["test", "minimal", "example", "version"],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    # Get package root directory
    package_root = Path(__file__).parent.parent.parent
    
    if args.command == "version":
        try:
            import bop_jax
            print(f"BOP-JAX version: {bop_jax.__version__}")
        except ImportError:
            print("BOP-JAX not installed")
        
        try:
            import jax
            print(f"JAX version: {jax.__version__}")
        except ImportError:
            print("JAX not available")
            
        try:
            import jax_md
            print("JAX-MD available")
        except ImportError:
            print("JAX-MD not available")
    
    elif args.command == "minimal":
        # Run minimal test
        minimal_test = package_root / "examples" / "minimal_test.py"
        if minimal_test.exists():
            os.system(f"python {minimal_test}")
        else:
            print("Minimal test script not found")
    
    elif args.command == "example":
        # Run full example
        example_script = package_root / "examples" / "simple_bop_calculation.py"
        if example_script.exists():
            os.system(f"python {example_script}")
        else:
            print("Example script not found")
    
    elif args.command == "test":
        # Run test suite
        test_script = package_root / "tests" / "test_bop_components.py"
        if test_script.exists():
            os.system(f"python {test_script}")
        else:
            print("Test suite not found")


if __name__ == "__main__":
    main()