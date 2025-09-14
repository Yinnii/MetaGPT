import re

# Path to your .txt file
filename = '/home/yin/Projects/MetaGPT/logs/20250903.txt'

# Lists to store extracted values
prompt_tokens = []
completion_tokens = []

# Regular expressions to match the patterns
prompt_pattern = re.compile(r'prompt_tokens:\s*([^\s,]+),')
completion_pattern = re.compile(r'completion_tokens:\s*([^\s]+)')

with open(filename, 'r') as file:
    for line in file:
        prompt_match = prompt_pattern.search(line)
        if prompt_match:
            prompt_tokens.append(int(prompt_match.group(1)))
        completion_match = completion_pattern.search(line)
        if completion_match:
            completion_tokens.append(int(completion_match.group(1)))

print("Prompt Tokens:", prompt_tokens)
print("Completion Tokens:", completion_tokens)
print("Total Prompt Tokens:", sum(prompt_tokens))
print("Total Completion Tokens:", sum(completion_tokens))