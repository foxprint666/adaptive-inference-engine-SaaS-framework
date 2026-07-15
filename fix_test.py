"""Fix script for integration_test_real.py"""
with open('integration_test_real.py', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Fix 1: sklearn training array - use int-encoded contract type
    if "contract_type\"] for r in churn_data" in line:
        line = line.replace("contract_type\"]", "contract_type_int\"]")

    # Fix 2: prediction features dict for telecom-b - use int not string
    if "\"contract_type\": rec[\"contract_type\"]}" in line:
        line = line.replace("\"contract_type\": rec[\"contract_type\"]}", "\"contract_type\": rec[\"contract_type_int\"]}")

    # Fix 3: features dict comprehension with string contract_type
    if '"support_tickets\", "contract_type"]' in line:
        line = line.replace('"support_tickets", "contract_type"]', '"support_tickets", "contract_type_int"]')

    new_lines.append(line)

with open('integration_test_real.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

# Also fix features dict built inline for prediction
content = open('integration_test_real.py', encoding='utf-8').read()

# Phase 2 telecom-b features
old = '        features = {k: rec[k] for k in\n                    ["customer_age", "tenure_months", "monthly_spend", "support_tickets", "contract_type"]}'
new = '        features = {"customer_age": rec["customer_age"], "tenure_months": rec["tenure_months"], "monthly_spend": rec["monthly_spend"], "support_tickets": rec["support_tickets"], "contract_type": rec["contract_type_int"]}'
content = content.replace(old, new)

# Phase 3 telecom-b features
old2 = '        features = {k: rec[k] for k in\n                    ["customer_age", "tenure_months", "monthly_spend", "support_tickets", "contract_type"]}\n        result = runtime_b.predict(features)'
new2 = '        features = {"customer_age": rec["customer_age"], "tenure_months": rec["tenure_months"], "monthly_spend": rec["monthly_spend"], "support_tickets": rec["support_tickets"], "contract_type": rec["contract_type_int"]}\n        result = runtime_b.predict(features)'
content = content.replace(old2, new2)

with open('integration_test_real.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed all contract_type references")

# Verify it parses
import ast
try:
    ast.parse(content)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax error: {e}")
