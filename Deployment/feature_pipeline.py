import re
import numpy as np


def preserve_tech_syntax(text):
    if not isinstance(text, str):
        return ""
    return ' '.join(re.sub(r'[^a-z0-9\+\#\.]', ' ', str(text).lower()).split())


CORE_SKILL_COUNT = 30

def build_onet_maps(df_occ):
    onet_skill_map = {}
    onet_core_map = {}
    onet_bench_map = {}
    onet_full_importance_map = {}

    for code, group in df_occ.groupby('ONET_SOC_Code'):
        group_unique = group.sort_values('Importance_Score', ascending=False).drop_duplicates('Clean_Feature')

        onet_skill_map[code] = set(group_unique['Clean_Feature'].tolist())

        full_dict = dict(zip(group_unique['Clean_Feature'], group_unique['Importance_Score']))
        onet_full_importance_map[code] = full_dict

        top_core = group_unique.nlargest(CORE_SKILL_COUNT, 'Importance_Score')
        core_dict = dict(zip(top_core['Clean_Feature'], top_core['Importance_Score']))
        onet_core_map[code] = core_dict

        onet_bench_map[code] = sum(core_dict.values())

    return onet_skill_map, onet_core_map, onet_bench_map, onet_full_importance_map


def extract_features(extracted_skills_list, candidate_code, onet_skill_map, onet_bench_map, onet_full_map):
    job_skills = onet_skill_map.get(candidate_code, set())

    if not job_skills or not extracted_skills_list:
        return 0, 0, 0.0, 0.0

    aligned_lower = set([preserve_tech_syntax(s) for s in extracted_skills_list])

    overlap_count = len(aligned_lower.intersection(job_skills))

    if overlap_count == 0 and len(aligned_lower) == 1:
        resume_doc = list(aligned_lower)[0]
        resume_doc_padded = f" {resume_doc} "
        matched_skills = set()
        for skill in job_skills:
            if f" {skill} " in resume_doc_padded:
                matched_skills.add(skill)
        aligned_lower = matched_skills
        overlap_count = len(aligned_lower)
        candidate_skill_count = len(resume_doc.split()) / 2.0
    else:
        candidate_skill_count = len(aligned_lower)

    skill_overlap_pct = overlap_count / len(job_skills) if len(job_skills) > 0 else 0.0

    full_dict = onet_full_map.get(candidate_code, {})
    benchmark = onet_bench_map.get(candidate_code, 0.0)

    matched_total_importance = sum([full_dict[skill] for skill in aligned_lower if skill in full_dict])
    core_fit_score = min(1.0, (matched_total_importance / benchmark)) if benchmark > 0 else 0.0

    return candidate_skill_count, overlap_count, skill_overlap_pct, core_fit_score
