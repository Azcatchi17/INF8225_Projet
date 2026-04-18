"""Gemini-only agentic pipeline (Gemini detects AND reviews masks, no DINO).

Shares MedSAM / state / metrics / Gemini client code with
`agentic.dino_gemini`; overrides the detection stage with `detect_boxes`.
"""
