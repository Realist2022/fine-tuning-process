import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from prompts import (
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    OVERALL_EXPERIENCE_SYSTEM_PROMPT,
    SKILL_MATCHER_SYSTEM_PROMPT,
)
from schemas import (
    CandidateRole,
    JobRequirementsOutput,
    OverallExperience,
    OverallExperienceOutput,
    RequirementInput,
    SkillMatch,
    SkillMatcherInput,
    SkillMatcherOutput,
    SynthesisAnnotation,
)


LLAMA_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_payload}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{assistant_payload}<|eot_id|>"""
DATE_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
MONTH_YEAR_PATTERN = re.compile(
    r"^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+(\d{4})$",
    re.IGNORECASE,
)
NUMERIC_MONTH_YEAR_PATTERN = re.compile(r"^(0?[1-9]|1[0-2])/(\d{4})$")
MONTH_NUMBERS = {
    name: index
    for index, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _is_source_excerpt(excerpt: str, source: str) -> bool:
    return bool(excerpt.strip()) and _normalize(excerpt) in _normalize(source)


def _is_valid_date(value: Optional[str]) -> bool:
    return value is None or value == "Present" or bool(DATE_PATTERN.fullmatch(value))


def _normalize_source_date(value: Optional[str], source: str) -> Optional[str]:
    if _is_valid_date(value):
        return value
    if value is None or not _is_source_excerpt(value, source):
        raise ValueError("experience.invalid_role_date")
    if value.casefold() in {"present", "current", "now"}:
        return "Present"
    month_year = MONTH_YEAR_PATTERN.fullmatch(value.strip())
    if month_year:
        month = MONTH_NUMBERS[month_year.group(1).casefold()]
        return f"{month_year.group(2)}-{month:02d}"
    numeric_month_year = NUMERIC_MONTH_YEAR_PATTERN.fullmatch(value.strip())
    if numeric_month_year:
        return f"{numeric_month_year.group(2)}-{int(numeric_month_year.group(1)):02d}"
    raise ValueError("experience.invalid_role_date")


def _format_example(system_prompt: str, user_payload: str, assistant_payload: dict) -> str:
    return LLAMA_TEMPLATE.format(
        system_prompt=system_prompt,
        user_payload=user_payload,
        assistant_payload=json.dumps(assistant_payload, ensure_ascii=False),
    )


def _validate_requirements(annotation: SynthesisAnnotation, job_description: str):
    requirements = annotation.job_requirements
    normalized_skills = [_normalize(item.skill_name) for item in requirements]
    if not requirements:
        raise ValueError("requirements.empty")
    if len(normalized_skills) != len(set(normalized_skills)):
        raise ValueError("requirements.duplicate")
    if not all(
        _is_source_excerpt(item.skill_name, job_description) for item in requirements
    ):
        raise ValueError("requirements.not_source_grounded")
    return requirements


def _matches_by_zero_based_id(annotation: SynthesisAnnotation, requirement_count: int):
    matches = annotation.skill_matches
    received_ids = [match.requirement_id for match in matches]
    if len(received_ids) != len(set(received_ids)):
        raise ValueError("skill_match.duplicate_id")

    zero_based = set(range(requirement_count))
    one_based = set(range(1, requirement_count + 1))
    received = set(received_ids)
    if received == zero_based:
        return {match.requirement_id: match for match in matches}
    if received == one_based:
        return {match.requirement_id - 1: match for match in matches}
    raise ValueError("skill_match.incomplete_ids")


def _validate_experience(annotation: SynthesisAnnotation, job_description: str, cv_text: str):
    experience = annotation.overall_experience
    if not _is_source_excerpt(experience.target_job_title, job_description):
        raise ValueError("experience.target_title_not_grounded")
    if experience.target_overall_years is None:
        if experience.target_years_evidence.strip():
            raise ValueError("experience.unexpected_years_evidence")
    elif not _is_source_excerpt(experience.target_years_evidence, job_description):
        raise ValueError("experience.years_not_grounded")

    normalized_roles = []
    for role in experience.candidate_roles:
        if not _is_source_excerpt(role.role_title, cv_text):
            raise ValueError("experience.role_title_not_grounded")
        if not _is_source_excerpt(role.cv_evidence, cv_text):
            raise ValueError("experience.role_evidence_not_grounded")
        if not role.match_rationale.strip():
            raise ValueError("experience.empty_rationale")
        normalized_roles.append(
            role.model_copy(
                update={
                    "start_date": _normalize_source_date(role.start_date, cv_text),
                    "end_date": _normalize_source_date(role.end_date, cv_text),
                }
            )
        )
    return experience.model_copy(update={"candidate_roles": normalized_roles})


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
                reject_outfile.write(
                    json.dumps({"row": row_number, "task": "annotation", "reason": reason})
                    + "\n"
                )
                continue

            examples = []
            try:
                requirements = _validate_requirements(annotation, job_description)
                job_output = JobRequirementsOutput(job_requirements=requirements)
                examples.append(
                    (
                        "job_requirements",
                        _format_example(
                            JOB_REQUIREMENTS_SYSTEM_PROMPT,
                            f"JOB DESCRIPTION:\n{job_description}",
                            job_output.model_dump(),
                        ),
                    )
                )
            except (TypeError, ValueError, ValidationError) as error:
                rejected[str(error)] += 1
                reject_outfile.write(
                    json.dumps(
                        {"row": row_number, "task": "job_requirements", "reason": str(error)}
                    )
                    + "\n"
                )
                requirements = None

            if requirements is not None:
                try:
                    matches_by_id = _matches_by_zero_based_id(annotation, len(requirements))
                    if any(
                        match.matched and not _is_source_excerpt(match.cv_evidence, cv_text)
                        for match in matches_by_id.values()
                    ):
                        raise ValueError("skill_match.positive_evidence_not_grounded")
                    matcher_input = SkillMatcherInput(
                        requirements=[
                            RequirementInput(requirement_id=index, skill_name=item.skill_name)
                            for index, item in enumerate(requirements)
                        ]
                    )
                    matcher_output = SkillMatcherOutput(
                        evaluations=[
                            SkillMatch(requirement_id=index, matched=matches_by_id[index].matched)
                            for index in range(len(requirements))
                        ]
                    )
                    requirements_json = json.dumps(
                        [item.model_dump() for item in matcher_input.requirements],
                        ensure_ascii=False,
                    )
                    examples.append(
                        (
                            "skill_matcher",
                            _format_example(
                                SKILL_MATCHER_SYSTEM_PROMPT,
                                f"JOB REQUIREMENTS:\n{requirements_json}\n\nCANDIDATE CV:\n{cv_text}",
                                matcher_output.model_dump(),
                            ),
                        )
                    )
                except (TypeError, ValueError, ValidationError) as error:
                    rejected[str(error)] += 1
                    reject_outfile.write(
                        json.dumps(
                            {"row": row_number, "task": "skill_matcher", "reason": str(error)}
                        )
                        + "\n"
                    )

            try:
                experience = _validate_experience(annotation, job_description, cv_text)
                experience_output = OverallExperienceOutput(
                    overall_experience=OverallExperience(
                        target_job_title=experience.target_job_title,
                        target_overall_years=experience.target_overall_years,
                        candidate_roles=[
                            CandidateRole.model_validate(
                                role.model_dump(exclude={"cv_evidence"})
                            )
                            for role in experience.candidate_roles
                        ],
                    )
                )
                examples.append(
                    (
                        "overall_experience",
                        _format_example(
                            OVERALL_EXPERIENCE_SYSTEM_PROMPT,
                            f"JOB DESCRIPTION:\n{job_description}\n\nCANDIDATE CV:\n{cv_text}",
                            experience_output.model_dump(),
                        ),
                    )
                )
            except (TypeError, ValueError, ValidationError) as error:
                rejected[str(error)] += 1
                reject_outfile.write(
                    json.dumps(
                        {"row": row_number, "task": "overall_experience", "reason": str(error)}
                    )
                    + "\n"
                )

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
