import sys
import os
import builtins

# Mock __import__ to fail for force_feedback_mpc
real_import = builtins.__import__

def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    if "force_feedback_mpc" in name:
        raise ImportError(f"Mocked ImportError for {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = mock_import

# Add python directory to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), 'python'))

try:
    from soft_mpc.utils import SoftContactModel1D
    import numpy as np
    
    # Test instantiation and set_contactType
    model = SoftContactModel1D(1000, 10, np.zeros(3), 1, '1Dz', 'LOCAL')
    print("Successfully instantiated SoftContactModel1D with 1Dz")
    print(f"Mask type: {model.maskType}")
    
    if model.maskType == 2:
        print("Mask type is correct (2 for z)")
    else:
        print(f"Mask type is incorrect: {model.maskType}")
        exit(1)

except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)

print("Verification passed!")
