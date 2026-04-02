#!/usr/bin/env python
"""Quick test of the optimized changes."""

from generate_answer import generate_answer

try:
    result = generate_answer("What is diversification?", top_k=8)
    print(f"✅ Success! Generated {len(result['answer'])} character answer")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Strategy: fixed (optimized)")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
