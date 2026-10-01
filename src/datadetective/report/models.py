"""The structured shape of an analysis report.

This is the Phase 4 deliverable. Instead of returning free text, the agent
must produce data that matches this model. Pydantic validates incoming data
against these fields: if a required field is missing or has the wrong type,
it raises a ValidationError that describes exactly what was wrong. That
error is what drives the retry loop in builder.py.

Five fields are filled by the model from the analysis output. chart_files is
not trusted to the model: our own code fills it from the files the sandbox
actually wrote, since that is a fact we already know for certain.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnalysisReport(BaseModel):
    question: str = Field(description="The original business question.")
    finding: str = Field(description="One-sentence headline answer.")
    # min_length=1 means an empty list fails validation, which triggers a
    # retry. A finding with no numbers behind it is not a real analysis.
    supporting_numbers: list[str] = Field(
        min_length=1,
        description="The key figures behind the finding, as short strings.",
    )
    method: str = Field(description="How the answer was computed.")
    caveats: list[str] = Field(
        description="Assumptions or limitations. Empty list if none."
    )
    # Filled by our code from the sandbox outputs, not by the model, so it
    # has a default and the model is never asked to produce it.
    chart_files: list[str] = Field(default_factory=list)
