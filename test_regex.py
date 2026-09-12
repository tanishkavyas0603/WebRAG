import re

# Test with sample think blocks
test_content = """<think>
Thinking Process:
1. Analyze
2. Extract
</think>

HTTP is a protocol."""

result = re.sub(r'<think>.*?</think>\s*', '', test_content, flags=re.DOTALL).strip()
print("Test 1 (with closing tag) - OLD REGEX:")
print("  Input length:", len(test_content))
print("  Output length:", len(result))
print("  Result:", repr(result))
print("  Success:", "HTTP" in result and "<think>" not in result)

# Test without closing tag
test_content2 = """<think>
Thinking Process:
1. Analyze
2. Extract

HTTP is a protocol."""

result2 = re.sub(r'<think>.*?</think>\s*', '', test_content2, flags=re.DOTALL).strip()
print("\n--- Test 2 (no closing tag) - OLD REGEX ---")
print("  Input length:", len(test_content2))
print("  Output length:", len(result2))
print("  Result:", repr(result2))
print("  Think tag removed:", "<think>" not in result2)

# Test new regex pattern
print("\n" + "="*60)
print("NEW REGEX PATTERN")
print("="*60)

result_new = re.sub(r'<think>.*?(?:</think>|(?=\n\n))', '', test_content2, flags=re.DOTALL).strip()
print("\n--- Test 2 (no closing tag) - NEW REGEX ---")
print("  Input length:", len(test_content2))
print("  Output length:", len(result_new))
print("  Result:", repr(result_new))
print("  Think tag removed:", "<think>" not in result_new)

result_new_1 = re.sub(r'<think>.*?(?:</think>|(?=\n\n))', '', test_content, flags=re.DOTALL).strip()
print("\n--- Test 1 (with closing tag) - NEW REGEX ---")
print("  Input length:", len(test_content))
print("  Output length:", len(result_new_1))
print("  Result:", repr(result_new_1))
print("  Think tag removed:", "<think>" not in result_new_1)
