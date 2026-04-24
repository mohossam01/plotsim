"""plotsim.scaffold — Groq/Llama domain config scaffolder (Phase A).

What it does:
    Accepts a plain-language domain description from the user, calls the Groq
    API (Llama 3) with a prompt engineered to produce a valid plotsim YAML
    config, and returns the parsed config. The only module in the project
    that makes network calls — the generation engine (Phase B) is fully
    offline. Handles API errors, rate limits, and malformed LLM responses
    by re-prompting with the pydantic validation error as feedback.

Input:
    User prompt (str), optional example chip (B2B SaaS, HR, Education,
    E-commerce), Groq API key from env.

Output:
    A validated PlotsimConfig (round-trippable via dump_config). Caller
    typically surfaces the YAML in the UI for user review before acceptance.

Implemented in Mission 009.
"""
