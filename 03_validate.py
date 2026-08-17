import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Any

from pydantic import ValidationError

from prompts import (
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    OVERALL_EXPERIENCE_SYSTEM_PROMPT,
    SKILL_MATCHER_SYSTEM_PROMPT,
)

# Only import the master schema we actually created!
from schemas import SynthesisAnnotation


LLAMA_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_payload}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{assistant_payload}<|eot_id|>"""


def _format_example(system_prompt: str, user_payload: str, assistant_payload: dict) -> str:
    """Formats the extracted data into the strict Llama 3 prompt template."""
    return LLAMA_TEMPLATE.format(
        system_prompt=system_prompt,
        user_payload=user_payload,
        assistant_payload=json.dumps(assistant_payload, ensure_ascii=False),
    )


# ==========================================
# Custom Exceptions
# ==========================================
class ValidationException(Exception):
    """Base exception for all pipeline validation failures."""
    pass

class GroundingError(ValidationException):
    """Raised when extracted text is not found in the source document."""
    pass

class DateFormatError(ValidationException):
    """Raised when a date string violates the required YYYY-MM schema."""
    pass


# ==========================================
# Core Validators
# ==========================================
class TextValidator:
    """Handles string normalization and source-grounding validation."""
    
    @staticmethod
    def is_grounded(excerpt: str, source_text: str) -> bool:
        if not excerpt or not excerpt.strip():
            return False
            
        clean_excerpt = re.sub(r"\s+", " ", excerpt.casefold()).strip()
        clean_source = re.sub(r"\s+", " ", source_text.casefold()).strip()
        
        return clean_excerpt in clean_source


class DateValidator:
    """Handles date format validation against strict schema constraints."""
    
    DATE_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

    @classmethod
    def is_valid_format(cls, date_text: Optional[str]) -> bool:
        if date_text is None or date_text.strip().casefold() == "present":
            return True
            
        return bool(cls.DATE_PATTERN.fullmatch(date_text.strip()))


# ==========================================
# Annotation Validator (The Orchestrator)
# ==========================================
class AnnotationValidator:
    """Validates structured LLM outputs against source documents and formatting rules."""

    @staticmethod
    def validate_requirements(ai_requirements: list, job_description: str) -> None:
        if not ai_requirements:
            raise ValidationException("Job requirements list cannot be empty.")

        for req in ai_requirements:
            if not TextValidator.is_grounded(req.skill_name, job_description):
                raise GroundingError(f"Skill not grounded in job description: '{req.skill_name}'")

    @staticmethod
    def validate_skills(ai_matches: list, cv_text: str) -> None:
        for match in ai_matches:
            if getattr(match, 'matched', False):
                evidence = getattr(match, 'cv_evidence', '')
                if not TextValidator.is_grounded(evidence, cv_text):
                    raise GroundingError(f"CV evidence not grounded in source text: '{evidence}'")

    @staticmethod
    def validate_experience(ai_experience: Any, job_description: str, cv_text: str) -> None:
        target_title = getattr(ai_experience, 'target_job_title', '')
        if not TextValidator.is_grounded(target_title, job_description):
            raise GroundingError("Target job title not grounded in job description.")

        candidate_roles = getattr(ai_experience, 'candidate_roles', [])
        for role in candidate_roles:
            role_title = getattr(role, 'role_title', '')
            if not TextValidator.is_grounded(role_title, cv_text):
                raise GroundingError(f"Candidate role title not grounded in CV: '{role_title}'")

            start_date = getattr(role, 'start_date', None)
            if not DateValidator.is_valid_format(start_date):
                raise DateFormatError(f"Invalid start date format (expected YYYY-MM): '{start_date}'")
                
            end_date = getattr(role, 'end_date', None)
            if not DateValidator.is_valid_format(end_date):
                raise DateFormatError(f"Invalid end date format (expected YYYY-MM or Present): '{end_date}'")


# ==========================================
# Main Execution File Logic
# ==========================================
def validate_and_format(
    input_file="synthetic_semantic.jsonl",
    output_file="train_semantic.jsonl",
    rejected_file="rejected_semantic.jsonl",
):
    accepted = Counter()
    rejected = Counter()
    seen_examples = set()
    duplicate_examples = 0

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile, open(rejected_file, "w", encoding="utf-8") as reject_outfile:
        
        for row_number, line in enumerate(infile, start=1):
            try:
                row = json.loads(line)
                job_description = row["job_description"]
                cv_text = row["cv_text"]
                annotation = SynthesisAnnotation.model_validate(row["annotation"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as error:
                reason = f"annotation.invalid: {error}"
                rejected[reason] += 1
                reject_outfile.write(json.dumps({"row": row_number, "task": "annotation", "reason": reason}) + "\n")
                continue

            examples = []
            
            # -----------------------------------------------------
            # Task 1: Grade and Format Job Requirements
            # -----------------------------------------------------
            try:
                AnnotationValidator.validate_requirements(annotation.job_requirements, job_description)
                
                # Format to match prompt expected output {"job_requirements": [...]}
                job_output = {
                    "job_requirements": [req.model_dump() for req in annotation.job_requirements]
                }
                
                examples.append((
                    "job_requirements",
                    _format_example(
                        JOB_REQUIREMENTS_SYSTEM_PROMPT,
                        f"JOB DESCRIPTION:\n{job_description}",
                        job_output,
                    ),
                ))
            except Exception as error:
                rejected[str(error)] += 1
                reject_outfile.write(json.dumps({"row": row_number, "task": "job_requirements", "reason": str(error)}) + "\n")

            # -----------------------------------------------------
            # Task 2: Grade and Format Skill Matcher
            # -----------------------------------------------------
            if annotation.job_requirements:
                try:
                    AnnotationValidator.validate_skills(annotation.skill_matches, cv_text)
                    
                    # Build the User Payload (Needs requirement ID and skill name)
                    matcher_input_reqs = [
                        {"requirement_id": index, "skill_name": item.skill_name}
                        for index, item in enumerate(annotation.job_requirements)
                    ]
                    requirements_json = json.dumps(matcher_input_reqs, ensure_ascii=False)
                    
                    # Format to match prompt expected output {"evaluations": [...]}
                    # We exclude cv_evidence here because the Llama model only outputs the boolean decisions!
                    matcher_output = {
                        "evaluations": [
                            match.model_dump(exclude={"cv_evidence"}) 
                            for match in annotation.skill_matches
                        ]
                    }
                    
                    examples.append((
                        "skill_matcher",
                        _format_example(
                            SKILL_MATCHER_SYSTEM_PROMPT,
                            f"JOB REQUIREMENTS:\n{requirements_json}\n\nCANDIDATE CV:\n{cv_text}",
                            matcher_output,
                        ),
                    ))
                except Exception as error:
                    rejected[str(error)] += 1
                    reject_outfile.write(json.dumps({"row": row_number, "task": "skill_matcher", "reason": str(error)}) + "\n")

            # -----------------------------------------------------
            # Task 3: Grade and Format Overall Experience
            # -----------------------------------------------------
            try:
                AnnotationValidator.validate_experience(annotation.overall_experience, job_description, cv_text)
                
                # Format to match prompt expected output {"overall_experience": {...}}
                experience_output = {
                    "overall_experience": annotation.overall_experience.model_dump()
                }
                
                examples.append((
                    "overall_experience",
                    _format_example(
                        OVERALL_EXPERIENCE_SYSTEM_PROMPT,
                        f"JOB DESCRIPTION:\n{job_description}\n\nCANDIDATE CV:\n{cv_text}",
                        experience_output,
                    ),
                ))
            except Exception as error:
                rejected[str(error)] += 1
                reject_outfile.write(json.dumps({"row": row_number, "task": "overall_experience", "reason": str(error)}) + "\n")

            # -----------------------------------------------------
            # Write out passing examples
            # -----------------------------------------------------
            for task, example in examples:
                if example in seen_examples:
                    duplicate_examples += 1
                    continue
                seen_examples.add(example)
                outfile.write(
                    json.dumps({"task": task, "source_row": row_number, "text": example}, ensure_ascii=False)
                    + "\n"
                )
                accepted[task] += 1

    print(f"Wrote {sum(accepted.values())} unique task examples to {output_file}")
    print(f"Accepted by task: {dict(accepted)}")
    print(f"Skipped {duplicate_examples} duplicate examples")
    print(f"Rejected by reason: {dict(rejected)}")
    print(f"Rejection details: {Path(rejected_file)}")


if __name__ == "__main__":
    validate_and_format()