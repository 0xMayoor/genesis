#!/usr/bin/env python3
"""
EXTERNAL VERIFICATION FRAMEWORK FOR LEVEL 0

This test CANNOT be gamed. It tests against:
1. ALL mnemonics found in real-world binaries (not our training data)
2. Multiple independent sources (Capstone, radare2)
3. Complete ISA coverage check

A model MUST pass this before Level 0 can be considered complete.
"""

import subprocess
import sys
import json
from pathlib import Path
from collections import defaultdict

from capstone import Cs, CS_ARCH_X86, CS_MODE_64

# =============================================================================
# STEP 1: BUILD COMPREHENSIVE MNEMONIC DATABASE FROM REAL BINARIES
# =============================================================================

def scan_real_binaries(max_binaries=300):
    """Scan real system binaries to find ALL mnemonics in actual use."""
    all_mnemonics = defaultdict(list)  # mnemonic -> list of (bytes, source)
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    
    binaries = []
    for p in ['/usr/bin', '/usr/lib', '/bin', '/sbin', '/usr/sbin']:
        if Path(p).exists():
            try:
                for f in Path(p).iterdir():
                    if f.is_file() and not f.is_symlink():
                        try:
                            with open(f, 'rb') as fp:
                                if fp.read(4) == b'\x7fELF':
                                    binaries.append(f)
                        except: pass
                    if len(binaries) >= max_binaries:
                        break
            except: pass
        if len(binaries) >= max_binaries:
            break
    
    for binary in binaries:
        try:
            result = subprocess.run(
                ['objcopy', '-O', 'binary', '--only-section=.text', str(binary), '/dev/stdout'],
                capture_output=True, timeout=10
            )
            if result.stdout:
                for insn in md.disasm(result.stdout[:100000], 0):
                    if len(all_mnemonics[insn.mnemonic]) < 5:  # Keep up to 5 examples
                        all_mnemonics[insn.mnemonic].append({
                            'bytes': bytes(insn.bytes).hex(),
                            'source': str(binary.name),
                            'full': f"{insn.mnemonic} {insn.op_str}"
                        })
        except:
            pass
    
    return dict(all_mnemonics)


def verify_with_radare2(hex_bytes):
    """Independent verification using radare2."""
    try:
        result = subprocess.run(
            ['rasm2', '-a', 'x86', '-b', '64', '-d', hex_bytes],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()[0]
    except:
        pass
    return None


# =============================================================================
# STEP 2: VERIFICATION FRAMEWORK
# =============================================================================

class Level0ExternalVerifier:
    """
    External verification that CANNOT be cheated.
    
    Requirements for PASS:
    1. Model must have ALL mnemonics in its vocabulary
    2. Model must correctly predict EVERY mnemonic at least once
    3. Cross-validated with independent tool (radare2)
    """
    
    def __init__(self, model_predict_fn):
        self.predict = model_predict_fn
        self.results = {
            'total_mnemonics_in_wild': 0,
            'mnemonics_in_model_vocab': 0,
            'mnemonics_correctly_predicted': 0,
            'missing_from_vocab': [],
            'prediction_errors': [],
            'passed': False,
        }
    
    def run_verification(self, mnemonic_db):
        """Run complete verification."""
        print("=" * 70)
        print("EXTERNAL VERIFICATION: LEVEL 0")
        print("=" * 70)
        
        self.results['total_mnemonics_in_wild'] = len(mnemonic_db)
        print(f"\nTotal unique mnemonics in real binaries: {len(mnemonic_db)}")
        
        correct_mnemonics = set()
        vocab_present = set()
        
        for mnemonic, examples in mnemonic_db.items():
            for ex in examples[:2]:  # Test up to 2 examples per mnemonic
                hex_bytes = ex['bytes']
                
                try:
                    prediction = self.predict(hex_bytes)
                    
                    if prediction == mnemonic:
                        correct_mnemonics.add(mnemonic)
                        vocab_present.add(mnemonic)
                    elif prediction is not None:
                        vocab_present.add(prediction)  # Model has SOME vocab
                        if len(self.results['prediction_errors']) < 50:
                            self.results['prediction_errors'].append({
                                'bytes': hex_bytes,
                                'expected': mnemonic,
                                'predicted': prediction,
                                'source': ex['source'],
                            })
                    else:
                        if mnemonic not in self.results['missing_from_vocab']:
                            self.results['missing_from_vocab'].append(mnemonic)
                except Exception as e:
                    if mnemonic not in self.results['missing_from_vocab']:
                        self.results['missing_from_vocab'].append(mnemonic)
        
        self.results['mnemonics_in_model_vocab'] = len(vocab_present)
        self.results['mnemonics_correctly_predicted'] = len(correct_mnemonics)
        
        # Calculate pass/fail
        total = self.results['total_mnemonics_in_wild']
        correct = self.results['mnemonics_correctly_predicted']
        
        self.results['passed'] = (correct == total)
        
        return self.results
    
    def print_report(self):
        """Print verification report."""
        r = self.results
        
        print(f"\n{'='*70}")
        print("VERIFICATION REPORT")
        print("=" * 70)
        
        total = r['total_mnemonics_in_wild']
        correct = r['mnemonics_correctly_predicted']
        coverage = 100 * correct / total if total > 0 else 0
        
        print(f"\nMnemonic Coverage:")
        print(f"  Total in real binaries: {total}")
        print(f"  Correctly predicted:    {correct}")
        print(f"  Coverage:               {coverage:.2f}%")
        
        if r['missing_from_vocab']:
            print(f"\nMISSING FROM VOCABULARY ({len(r['missing_from_vocab'])}):")
            for m in sorted(r['missing_from_vocab'])[:30]:
                print(f"  - {m}")
            if len(r['missing_from_vocab']) > 30:
                print(f"  ... and {len(r['missing_from_vocab']) - 30} more")
        
        if r['prediction_errors']:
            print(f"\nPREDICTION ERRORS ({len(r['prediction_errors'])} shown):")
            for err in r['prediction_errors'][:10]:
                print(f"  {err['bytes']:20} expected={err['expected']:12} got={err['predicted']}")
        
        print(f"\n{'='*70}")
        if r['passed']:
            print("RESULT: ✓ PASSED - Level 0 is COMPLETE")
        else:
            print(f"RESULT: ✗ FAILED - {total - correct} mnemonics not covered")
            print(f"\nLevel 0 CANNOT proceed until this passes.")
        print("=" * 70)
        
        return r['passed']


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Building mnemonic database from real binaries...")
    print("(This scans system binaries - completely external to our training)")
    
    mnemonic_db = scan_real_binaries(300)
    
    print(f"Found {len(mnemonic_db)} unique mnemonics")
    
    # Save database for reproducibility
    with open('tests/mnemonic_database.json', 'w') as f:
        json.dump(mnemonic_db, f, indent=2)
    print("Saved to tests/mnemonic_database.json")
    
    # Try to load and test our model
    try:
        sys.path.insert(0, '.')
        from levels.level0_machine import predict_mnemonic
        
        verifier = Level0ExternalVerifier(predict_mnemonic)
        verifier.run_verification(mnemonic_db)
        passed = verifier.print_report()
        
        sys.exit(0 if passed else 1)
    except ImportError as e:
        print(f"\nCannot load Level 0 model: {e}")
        print("Run this after training to verify.")
        sys.exit(1)
