# Git Writer

Analyze the staged diff. Your tasks:

1. Suggest a clear, conventional commit message:
   - Use the format: <type>(<scope>): <subject>
   - Types: feat, fix, refactor, docs, chore
   - Scope: optional (e.g., rag_index, git_writer)
   - Subject: concise summary, imperative mood, max 50 characters
   - Body (optional): explain the change in 72 characters per line
   - Example: feat(rag_index): Centralize embedding model init

2. Indicate if the change should be split into smaller commits (explain why)

3. Perform a brief code review: check for clarity, structure, and maintainability

4. Check if new logic or code paths have corresponding test coverage
