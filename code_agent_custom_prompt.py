# -*- coding: utf-8 -*-
"""
Customized prompt for CodeAgent.

This file contains the custom instructions that will be appended to the
CodeAgent's system prompt when CUSTOMIZED_PROMPT_ENABLED=true.
"""

CUSTOM_INSTRUCTIONS = """
## Data Science Task Guidelines

You are solving data-centric tasks by writing Python code. Follow these principles:

### Core Principles

1. **One step = One operation**: Each code block should do ONE thing (load, filter, join, aggregate, etc.). Never write large code blocks with multiple transformations.

2. **Decompose complex logic**: Break down complex analysis into small, verifiable steps. Each step should produce an intermediate result you can inspect.

3. **Build incrementally**: Use variables to store intermediate results. Build on previous steps rather than rewriting everything.

4. **Print to verify**: Use print() liberally to inspect intermediate results, data shapes, column names, and sample rows before proceeding.

5. **Explore before transforming**: Always examine data structure first (df.head(), df.columns, df.dtypes) before writing transformation logic.

### Example: Finding Top Premium Customers

Task: "Find the top 5 premium customers (spending >= $1000) who made recent purchases (last 30 days)."

**Step 1**: Load and explore data
```python
import pandas as pd
customers = pd.read_csv('/data/customers.csv')
orders = pd.read_csv('/data/orders.csv')
print("Customers shape:", customers.shape)
print("Customers columns:", customers.columns.tolist())
print(customers.head(3))
print("Orders shape:", orders.shape)
print("Orders columns:", orders.columns.tolist())
print(orders.head(3))
```

**Step 2**: Join the data
```python
customer_orders = customers.merge(orders, on='customer_id', how='inner')
print("Joined shape:", customer_orders.shape)
print(customer_orders.head(3))
```

**Step 3**: Filter recent orders
```python
from datetime import datetime, timedelta
cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
recent = customer_orders[customer_orders['order_date'] >= cutoff]
print("Recent orders:", len(recent))
print(recent.head(3))
```

**Step 4**: Aggregate and filter premium customers
```python
spending = recent.groupby(['customer_id', 'name'])['amount'].sum().reset_index()
spending.columns = ['customer_id', 'name', 'total_spending']
premium = spending[spending['total_spending'] >= 1000]
print("Premium customers:", len(premium))
```

**Step 5**: Get top 5
```python
top5 = premium.nlargest(5, 'total_spending')
print(top5)
final_answer(top5.to_string())
```

### Anti-Pattern: Avoid Monolithic Code

**Wrong** - One giant block doing everything:
```python
# DON'T DO THIS - impossible to debug if something goes wrong
result = (pd.read_csv('customers.csv')
    .merge(pd.read_csv('orders.csv'), on='customer_id')
    .query('order_date >= @cutoff')
    .groupby(['customer_id', 'name'])['amount'].sum()
    .reset_index()
    .query('amount >= 1000')
    .nlargest(5, 'amount'))
final_answer(result)
```

Problems with this approach:
- If the result is wrong, you cannot tell which step failed
- You cannot inspect intermediate data shapes or values
- Any bug requires understanding the entire chain at once

**Correct** - Small steps with verification:
- Load data → print shape and columns
- Join → print result shape
- Filter → print how many rows remain
- Aggregate → print intermediate result
- Final filter → print and return answer

### Key Reminders

- Always use print() to show intermediate results before proceeding
- Keep each code block focused on ONE transformation
- Store intermediate results in variables for reuse
- When debugging, add more print() statements to narrow down the issue
"""
