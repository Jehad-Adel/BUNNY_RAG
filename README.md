# BUNNY_RAG
Financial RAG system leveraging OpenAI’s Structured Outputs to automate SQL generation, reporting, and causal reasoning for ARR and expense tracking.

 Financial Intelligence AI (FinAI)
An intelligent financial reporting and analytics engine that bridges the gap between natural language queries and complex accounting data. This project leverages LLMs to automate financial data extraction and provide deep causal analysis.
 Overview
The system is designed to handle financial data through a two-staged evolution:
 1. Natural Language Reporting (PoC): Translating user queries into structured SQL to generate instant reports (e.g., "Show me expenses by vendor").
 2. Causal Reasoning & Interpretation: Moving beyond tables to explain the "Why" behind the numbers (e.g., "Why did my ARR drop this month?").
    
 Key Features
 * Text-to-SQL Engine: Uses OpenAI’s Structured Outputs to reliably generate valid SQL queries based on a predefined database schema.
 * Pydantic Integration: Ensures that the model's output strictly adheres to the data models required for financial accuracy.
 * Automated Financial Reconciliation: Joins ledger entries with account metadata to produce exportable CSV/Excel reports.
 * Advanced Causal Analysis: A multi-layered reasoning approach that inspects:
   * Subscription status changes (churned, paused, canceled).
   * Credit vs. Revenue fluctuations.
   * Invoice-level discounts and line-item drivers.
 * Agentic Workflow: Built using LangChain to orchestrate the flow between data retrieval and human-readable explanations.
   
 Tech Stack
 * LLM: OpenAI (via LangChain)
 * Validation: Pydantic (Structured Outputs)
 * Data Processing: Python, SQL
 * Integrations: Financial Ledger Systems / Datalabs
   
 Roadmap
 * [x] Phase 1: Structured RAG for natural-language reporting.
 * [ ] Phase 2: Implementation of interpretive reasoning layers.
 * [ ] Phase 3: Integration with multi-source accounting data.
       
  Cost Efficiency
The architecture is optimized for production, utilizing "lightweight" API calls for data retrieval, keeping operational costs at a fraction of a cent per query.
