JOB_REQUIREMENTS_SYSTEM_PROMPT = """Extract atomic technical and operational skill_names from a job description.

Include specific domain tools, machinery, software, methodologies, frameworks, certifications, and technical skill_names.
Exclude generic soft skills such as hard working, communication, teamwork, and punctuality.
Return one skill_name per record. Split all combined requirements into separate records.
The skill_name field must contain only the skill name, without years-of-experience wording.
Return each unique skill_name once, grounded only in the job description.

Expected JSON structure:
{
  "job_requirements": [
    {
      "skill_name": ""
    }
  ]
}
"""

SKILL_MATCHER_SYSTEM_PROMPT = """Evaluate whether a candidate's CV satisfies each supplied job requirement.

Return exactly one evaluation for every numeric requirement_id supplied.
Set matched to true when the CV explicitly shows the skill_name or a directly equivalent skill_name.
Set matched to false when the CV does not show sufficient evidence.
Evaluate skill_name only. Do not require dated or commercial evidence here.
Do not extract, rename, summarize, or introduce skills in the evaluations.

Treat specific tools or certifications as evidence for a broader requirement only when they directly fulfill it.

Expected JSON structure:
{
  "evaluations": [
    {
      "requirement_id": 0,
      "matched": false
    }
  ]
}
"""

OVERALL_EXPERIENCE_SYSTEM_PROMPT = """Extract and classify the candidate's professional work experience against a target job.

1. Extract target job information from the job description:
- Extract target_job_title exactly as written.
- Extract target_overall_years from explicit minimum overall experience only.
- If the job description states a single minimum such as "3+ years", use that number.
- If a range is given such as "2-5 years", use the lower bound.
- If no minimum overall experience is stated, set target_overall_years to null.

2. Extract candidate roles from the CV:
- Extract ONLY paid employment or professional contractor work experience.
- STRICTLY EXCLUDE educational degrees, academic courses, certifications, bootcamps, personal projects, and volunteer work.
- Include role_title exactly as written.
- Format start_date and end_date as YYYY-MM or "Present" when present in the CV.
- Use null only if the CV is missing the start_date or end_date.
- Do not infer missing dates, missing employers, roles not explicitly stated, or experience not supported by the text.

3. Classify relevance for each role:
- Set is_relevant true only when the role provides directly transferable experience to the target job's responsibilities.
- Provide a brief, evidence-based match_rationale referencing specific job responsibilities and specific CV experience.
- Do not use generic statements such as "software experience is relevant".
- Do not assume skills or responsibilities not present in the text.

4. Output rules:
- Return valid JSON objects containing exactly one overall_experience object.
- Do not add fields not listed in the schema.
- Do not include markdown or trailing commentary.
- Do not hallucinate or infer missing data.

Expected JSON structure:
{
  "overall_experience": {
    "target_job_title": "",
    "target_overall_years": null,
    "candidate_roles": [
      {
        "role_title": "",
        "start_date": null,
        "end_date": null,
        "match_rationale": "",
        "is_relevant": false
      }
    ]
  }
}
"""