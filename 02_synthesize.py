import json

from openai import OpenAI
from dotenv import load_dotenv

from schemas import SynthesisAnnotation


SYSTEM_PROMPT = """Create a source-grounded annotation for three CV-to-job evaluation tasks.

Use only the supplied CV and job description. Never infer skills, job titles, employers,
dates, certifications, or work experience that are not explicitly supported by the source.

job_requirements: Extract unique atomic technical or operational requirements from the job
description only. Exclude soft skills. Each skill_name must be a phrase present in the job
description.

skill_matches: Return exactly one match for each job_requirement, using zero-based IDs in
the same order. Set matched true only if cv_evidence is an exact excerpt from the CV that
demonstrates that skill or a directly equivalent skill. Use an empty cv_evidence when
matched is false.

overall_experience: Extract the target title exactly as written in the job description.
Set target_overall_years only for an explicit minimum overall-experience requirement and
quote the supporting job-description text in target_years_evidence. Include only paid
employment or professional contractor roles. For every role, cv_evidence must be an exact
CV excerpt supporting the role. Do not include education, projects, volunteer work, or
certifications as roles. Only mark a role relevant when its rationale names concrete
responsibilities evidenced in both sources.
"""


def generate_synthetic_data(
    input_file="raw_cv_jd_pairs.jsonl",
    output_file="synthetic_semantic.jsonl",
    max_samples=500,
):
    load_dotenv()
    client = OpenAI()
    generated = 0
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if generated >= max_samples:
                break

            data = json.loads(line)
            job_description = data["job_description"]
            cv_text = data["cv_text"]
            prompt = json.dumps(
                {"job_description": job_description, "cv_text": cv_text},
                ensure_ascii=False,
            )

            try:
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=SynthesisAnnotation,
                )

                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("The model returned no structured output")

                row = {
                    "job_description": job_description,
                    "cv_text": cv_text,
                    "annotation": parsed.model_dump(),
                }
                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                generated += 1
                print(f"Generated semantic sample {generated}/{max_samples}")

            except Exception as error:
                print(f"Skipped row due to error: {error}")


if __name__ == "__main__":
    generate_synthetic_data()
