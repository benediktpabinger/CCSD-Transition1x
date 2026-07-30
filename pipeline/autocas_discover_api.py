"""
Quick API discovery for scine_autocas 3.0.0.

Run this first to understand the installed API:
    python autocas_discover_api.py

Output tells you the exact import paths, class names, and signatures
needed to write autocas_active_space.py correctly.
"""
import inspect
import pkgutil
import sys


def main():
    try:
        import scine_autocas
    except ImportError:
        sys.exit("scine_autocas not installed — load Python/3.11.3-GCCcore-12.3.0 first")

    print(f"scine_autocas {scine_autocas.__version__}")
    print(f"Location: {scine_autocas.__file__}")
    print()

    # Walk all submodules
    print("=== All modules ===")
    for m in pkgutil.walk_packages(scine_autocas.__path__, scine_autocas.__name__ + '.'):
        print(f"  {m.name}")
    print()

    # Collect all classes across all submodules
    print("=== All classes ===")
    for finder, mod_name, is_pkg in pkgutil.walk_packages(
            scine_autocas.__path__, scine_autocas.__name__ + '.'):
        try:
            mod = __import__(mod_name, fromlist=['*'])
        except Exception as e:
            print(f"  [{mod_name}] import error: {e}")
            continue
        for cls_name in dir(mod):
            obj = getattr(mod, cls_name)
            if inspect.isclass(obj) and obj.__module__ == mod_name:
                try:
                    sig = inspect.signature(obj.__init__)
                    print(f"  {mod_name}.{cls_name}{sig}")
                except Exception:
                    print(f"  {mod_name}.{cls_name}  (no signature)")
    print()

    # Check top-level exports
    print("=== Top-level exports ===")
    for name in dir(scine_autocas):
        obj = getattr(scine_autocas, name)
        if inspect.isclass(obj):
            try:
                sig = inspect.signature(obj.__init__)
                print(f"  scine_autocas.{name}{sig}")
            except Exception:
                print(f"  scine_autocas.{name}")
    print()

    # Try a minimal pyscf round-trip to confirm block2 works
    print("=== block2 DMRG sanity check ===")
    try:
        import block2
        print(f"  block2 imported OK")
    except Exception as e:
        print(f"  block2 import failed: {e}")

    try:
        import pyscf
        print(f"  pyscf {pyscf.__version__} OK")
    except ImportError:
        print("  pyscf NOT INSTALLED — install with: pip install pyscf --user")

    print()
    print("Done. Use the class signatures above to implement autocas_active_space.py.")


if __name__ == '__main__':
    main()
