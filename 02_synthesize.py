import json
from openai import OpenAI
from dotenv import load_dotenv

from schemas import SynthesisAnnotation


SYSTEM_PROMPT = """Create a source-grounded annotation for three CV-to-job evaluation tasks.

Use only the supplied CV and job description. Never infer skills, job titles, employers,
dates, certifications, or work experience that are not explicitly supported by the source.

job_requirements:
- Extract unique atomic technical and operational skill_names from the job description only.
- Include specific domain tools, machinery, software, methodologies, frameworks, certifications, and technical skill_names.
- Exclude generic soft skills such as hard working, communication, teamwork, and punctuality.
- Return one skill_name per record. Split all combined requirements into separate records.
- The skill_name field must contain only the skill name, without years-of-experience wording.
- Return each unique skill_name once, grounded strictly in the job description.

skill_matches:
- Return exactly one evaluation for every job_requirement, using zero-based requirement_id values in identical order.
- Set matched to true when the CV explicitly shows the skill_name or a directly equivalent skill_name.
- Treat specific tools or certifications as evidence for a broader requirement only when they directly fulfill it.
- Evaluate skill_name only; do not require dated or commercial evidence.
- Set matched to false when the CV does not show sufficient evidence.
- Set cv_evidence to an exact excerpt from the CV when matched is true; use an empty string when matched is false.
- Do not extract, rename, summarize, or introduce skills in the evaluations.

overall_experience:
- Extract target_job_title exactly as written in the job description.
- Set target_overall_years from explicit minimum overall experience only. If a single minimum is given (e.g., '3+ years'), use that number. If a range is given (e.g., '2-5 years'), use the lower bound. If not stated, set to null.
- candidate_roles: Extract ONLY paid employment or professional contractor work experience.
- STRICTLY EXCLUDE educational degrees, academic courses, certifications, bootcamps, personal projects, and volunteer work.
- Format start_date and end_date as 'YYYY-MM' (e.g., '2025-07') or 'Present' when present in the CV; use null if missing. Do not infer missing dates.
- For every role, set is_relevant to true only when the role provides directly transferable experience to the target job's responsibilities.
- match_rationale: Provide a brief, concrete comparison referencing specific job responsibilities and CV experience.
"""


def generate_synthetic_data(
    input_file="raw_cv_jd_pairs.jsonl",
    output_file="synthetic_semantic.jsonl",
    max_samples=1000,
):
    load_dotenv()
    client = OpenAI()
    generated = 0
    
    print(f"Starting generation (Max limit: {max_samples})...")
    
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if generated >= max_samples:
                print("Reached maximum sample limit.")
                break

            data = json.loads(line)
            job_description = data["job_description"]
            cv_text = data["cv_text"]
            
            # Extract the dictionaries so we can pass them through
            macro_dict = data.get("macro_dict", {})
            micro_dict = data.get("micro_dict", {})

            # ONLY put the text in the prompt for the LLM
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

                # Reassemble the row with the dictionaries included
                row = {
                    "job_description": job_description,
                    "cv_text": cv_text,
                    "macro_dict": macro_dict,
                    "micro_dict": micro_dict,
                    "annotation": parsed.model_dump(),
                }
                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
                generated += 1
                
                if generated % 10 == 0:
                    print(f"Generated semantic sample {generated}...")

            except Exception as error:
                print(f"Skipped row due to error: {error}")
                
    print(f"Finished! Successfully generated {generated} synthetic examples.")


if __name__ == "__main__":
    generate_synthetic_data()