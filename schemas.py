"""Schemas for synthetic data generation and fine-tuning extraction."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


# --- 1. Job Requirements Schemas ---
class JobRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_name: str = Field(
        min_length=1, 
        description="One atomic technical or operational skill_name."
    )


# --- 2. Skill Matching Schemas ---
class RequirementEvaluation(BaseModel):
    requirement_id: int = Field(
        description="The supplied zero-based numeric requirement ID."
    )
    matched: bool = Field(
        description="Whether the CV satisfies this requirement."
    )
    cv_evidence: str = Field(
        default="", 
        description="Exact CV excerpt if matched is true; empty string if false."
    )


# --- 3. Work Experience Schemas ---
class WorkRole(BaseModel):
    role_title: str = Field(
        description="Title of the candidate's position."
    )
    start_date: Optional[str] = Field(
        default=None, 
        description="Role start date in YYYY-MM format, or null when missing."
    )
    end_date: Optional[str] = Field(
        default=None, 
        description="Role end date in YYYY-MM format, Present, or null when missing."
    )
    match_rationale: str = Field(
        description="Brief comparison of this role with the target job."
    )
    is_relevant: bool = Field(
        description="Whether the role provides directly relevant target-job experience."
    )


class OverallExperienceOutput(BaseModel):
    target_job_title: str = Field(
        description="Target role title from the listing."
    )
    target_overall_years: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        description="Explicit overall experience required, or null when unspecified."
    )
    candidate_roles: list[WorkRole] = Field(
        description="Professional roles supported by the candidate CV."
    )


# --- Master Synthesis Schema (for 02_synthesize.py) ---
class SynthesisAnnotation(BaseModel):
    job_requirements: list[JobRequirement] = Field(
        description="Unique atomic technical or operational capabilities extracted from the job description."
    )
    skill_matches: list[RequirementEvaluation] = Field(
        description="One match decision for every extracted job requirement, ordered by requirement_id."
    )
    overall_experience: OverallExperienceOutput = Field(
        description="Extracted overall target job information and candidate role experience."
    )