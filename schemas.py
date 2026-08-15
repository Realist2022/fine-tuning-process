from typing import List, Optional

from pydantic import BaseModel, Field

# --- AI GENERATED SCHEMAS (Semantic Engine) ---
class JobRequirement(BaseModel):
    skill_name: str = Field(min_length=1)

class JobRequirementsOutput(BaseModel):
    job_requirements: List[JobRequirement]

class SkillsEvaluation(BaseModel):
    requirement_category: str
    job_requirements: List[JobRequirement]
    matched_cv_skills: List[str]
    missing_cv_skills: List[str]
    rationale: str

class RequirementInput(BaseModel):
    requirement_id: int = Field(ge=0)
    skill_name: str = Field(min_length=1)

class SkillMatcherInput(BaseModel):
    requirements: List[RequirementInput]

class SkillMatch(BaseModel):
    requirement_id: int = Field(ge=0)
    matched: bool

class SkillMatcherOutput(BaseModel):
    evaluations: List[SkillMatch]

class CandidateRole(BaseModel):
    role_title: str = Field(min_length=1)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    match_rationale: str
    is_relevant: bool

class OverallExperience(BaseModel):
    target_job_title: str
    target_overall_years: Optional[float] = None
    candidate_roles: List[CandidateRole]

class OverallExperienceOutput(BaseModel):
    overall_experience: OverallExperience

# These fields are used only while producing and checking labels. They are not
# part of the deployed Llama response contract.
class SynthesisSkillMatch(SkillMatch):
    cv_evidence: str = ""

class SynthesisCandidateRole(CandidateRole):
    cv_evidence: str = ""

class SynthesisOverallExperience(BaseModel):
    target_job_title: str = Field(min_length=1)
    target_overall_years: Optional[float] = Field(default=None, ge=0)
    target_years_evidence: str = ""
    candidate_roles: List[SynthesisCandidateRole]

class SynthesisAnnotation(BaseModel):
    job_requirements: List[JobRequirement]
    skill_matches: List[SynthesisSkillMatch]
    overall_experience: SynthesisOverallExperience

# --- PYTHON GENERATED SCHEMAS (Deterministic Engine) ---
class Pillar(BaseModel):
    score: float
    raw: str
    applicable: bool

class Scorecard(BaseModel):
    final_relevance: float
    pillar_a: Pillar
    pillar_b: Pillar
    counted_roles: List[str]

class Metrics(BaseModel):
    total_requirements: int
    total_matched: int
    match_percentage: float
    final_relevance: float

class Check(BaseModel):
    name: str
    expected: str
    actual: str
    passed: bool

class Evaluation(BaseModel):
    passed: bool
    checks: List[Check]

class FullGuestimatorOutput(BaseModel):
    """The final artifact combining AI semantics and Python math"""
    skills_evaluation: SkillsEvaluation
    overall_experience: OverallExperience
    scorecard: Scorecard
    metrics: Metrics
    evaluation: Evaluation