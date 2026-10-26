# Resume Parser (Ollama)

A self-hosted pipeline that turns PDF resumes into structured JSON.

- Runs fully locally with Ollama (no cloud services)
- Extracts personal info, contact details, skills, summary,
  languages, work experience, education and projects
- Validates output against a JSON schema

Requires Python 3.11 and Ollama (models: qwen3:1.7b, qwen2.5vl:3b).
Never commit real resumes or candidate data.
