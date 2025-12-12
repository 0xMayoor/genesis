#!/usr/bin/env python3
"""
BULLETPROOF LEVEL 0 VERIFICATION

Tests EACH of the 369 mnemonics found in real-world binaries.
A mnemonic is PASSED only if ALL its test cases are correct.
Overall PASS requires 369/369 = 100.00%

NO EXCEPTIONS. NO EXCUSES.
"""

import json
import sys
from pathlib import Path

def load_mnemonic_database():
    """Load the complete mnemonic database."""
    db_path = Path(__file__).parent / 'complete_mnemonic_db.json'
    if not db_path.exists():
        print(f"ERROR: {db_path} not found. Run external_verify_level0.py first.")
        sys.exit(1)
    
    with open(db_path) as f:
        return json.load(f)


def verify_level0(predict_fn):
    """
    Verify Level 0 model against EVERY mnemonic.
    
    Returns: (passed, total, results_dict)
    """
    db = load_mnemonic_database()
    
    results = {
        'per_mnemonic': {},  # mnemonic -> {correct, total, passed}
        'total_mnemonics': len(db),
        'passed_mnemonics': 0,
        'failed_mnemonics': [],
        'missing_mnemonics': [],  # Not in vocabulary at all
        'overall_passed': False,
    }
    
    for mnemonic, examples in sorted(db.items()):
        correct = 0
        total = len(examples)
        errors = []
        
        for ex in examples:
            try:
                prediction = predict_fn(ex['bytes'])
                if prediction == mnemonic:
                    correct += 1
                else:
                    errors.append({
                        'bytes': ex['bytes'],
                        'expected': mnemonic,
                        'got': prediction
                    })
            except Exception as e:
                errors.append({
                    'bytes': ex['bytes'],
                    'expected': mnemonic,
                    'got': f'ERROR: {e}'
                })
        
        passed = (correct == total)
        results['per_mnemonic'][mnemonic] = {
            'correct': correct,
            'total': total,
            'passed': passed,
            'errors': errors[:3]  # Keep first 3 errors
        }
        
        if passed:
            results['passed_mnemonics'] += 1
        elif correct == 0:
            results['missing_mnemonics'].append(mnemonic)
        else:
            results['failed_mnemonics'].append(mnemonic)
    
    results['overall_passed'] = (results['passed_mnemonics'] == results['total_mnemonics'])
    
    return results


def print_report(results):
    """Print detailed verification report."""
    print("=" * 80)
    print("LEVEL 0 VERIFICATION: EACH MNEMONIC INDIVIDUALLY")
    print("=" * 80)
    
    total = results['total_mnemonics']
    passed = results['passed_mnemonics']
    
    print(f"\nTOTAL MNEMONICS: {total}")
    print(f"PASSED:          {passed}")
    print(f"FAILED:          {total - passed}")
    print(f"COVERAGE:        {100*passed/total:.2f}%")
    
    # Show each mnemonic status
    print(f"\n{'MNEMONIC':<20} {'STATUS':<10} {'CORRECT':<15}")
    print("-" * 50)
    
    # First show failures
    if results['missing_mnemonics']:
        print(f"\n*** MISSING FROM VOCABULARY ({len(results['missing_mnemonics'])}) ***")
        for m in sorted(results['missing_mnemonics']):
            r = results['per_mnemonic'][m]
            print(f"  {m:<20} MISSING     {r['correct']}/{r['total']}")
            if r['errors']:
                print(f"      Example: {r['errors'][0]['bytes']} -> got '{r['errors'][0]['got']}'")
    
    if results['failed_mnemonics']:
        print(f"\n*** PREDICTION ERRORS ({len(results['failed_mnemonics'])}) ***")
        for m in sorted(results['failed_mnemonics']):
            r = results['per_mnemonic'][m]
            print(f"  {m:<20} PARTIAL     {r['correct']}/{r['total']}")
            if r['errors']:
                print(f"      Example: {r['errors'][0]['bytes']} -> got '{r['errors'][0]['got']}'")
    
    # Summary
    print("\n" + "=" * 80)
    if results['overall_passed']:
        print("✓ LEVEL 0 VERIFIED: 100% COVERAGE ON ALL 369 MNEMONICS")
        print("=" * 80)
        return True
    else:
        print(f"✗ LEVEL 0 FAILED: {passed}/{total} mnemonics ({100*passed/total:.2f}%)")
        print(f"\n  Missing from vocabulary: {len(results['missing_mnemonics'])}")
        print(f"  Prediction errors:       {len(results['failed_mnemonics'])}")
        print(f"\n  LEVEL 0 IS NOT COMPLETE. DO NOT PROCEED.")
        print("=" * 80)
        return False


def main():
    print("Loading mnemonic database...")
    db = load_mnemonic_database()
    print(f"Database: {len(db)} mnemonics, {sum(len(v) for v in db.values())} test cases")
    
    # Load Level 0 model
    print("\nLoading Level 0 model...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from levels.level0_machine import predict_mnemonic
        print("Model loaded.")
    except ImportError as e:
        print(f"ERROR: Cannot load Level 0 model: {e}")
        sys.exit(1)
    
    # Run verification
    print("\nRunning verification (testing each mnemonic individually)...")
    results = verify_level0(predict_mnemonic)
    
    # Print report
    passed = print_report(results)
    
    # Save detailed results
    results_path = Path(__file__).parent / 'level0_verification_results.json'
    
    # Make results JSON serializable
    json_results = {
        'total_mnemonics': results['total_mnemonics'],
        'passed_mnemonics': results['passed_mnemonics'],
        'failed_mnemonics': results['failed_mnemonics'],
        'missing_mnemonics': results['missing_mnemonics'],
        'overall_passed': results['overall_passed'],
        'coverage_percent': 100 * results['passed_mnemonics'] / results['total_mnemonics'],
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
