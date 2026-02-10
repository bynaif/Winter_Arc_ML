"""
Input: numbers = [1, 2, 3, 4, 5, 6]
Output: [4, 16, 36] (Squares of even numbers).
Constraint: Use a List Comprehension.
"""

numbers = [1,2,3,4,5,6]
squares = [x**2 for x in numbers if x%2==0]
print(squares)