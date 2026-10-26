# Resume AI Benchmark

Goal:

Build the world's most accurate self-hosted Resume Parser.

The system will benchmark multiple pipelines.

Pipeline A

PDF
↓

Qwen2.5-VL

↓

JSON

Pipeline B

PDF
↓

Qwen2.5-VL

↓

Markdown

↓

Python

↓

JSON

Pipeline C

PDF

↓

Docling

↓

Markdown

↓

Qwen 1.7B

↓

Python

↓

JSON

Every pipeline should generate identical JSON schema.

The application must benchmark

- speed
- accuracy
- memory
- hallucinations
- missing fields

The application should be modular.

No duplicated code.
