"""Helper functions for obtaining features."""
import itertools
import re
import typing

import numpy as np

REQUIRED_SKILL_MULTIPLIER = 3
OPTIONAL_SKILL_MULTIPLIER = 1

##### elastic search feature helpers ####


def pairwise(iterable):
    """Create pairwise elements from iterable."""
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    # pylint: disable=invalid-name
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# pylint: disable=dangerous-default-value
def compute_hourly_rate_boost(
    hourly_rate,
    intervals=[0, 10, 15, 20, 30, 40, 50],
    boosts=[0, 20, 15, 12, 10, 7, 3, -5],
):
    """Compute hourly rate boost."""
    if hourly_rate == intervals[0]:
        return boosts[0]
    if hourly_rate >= intervals[-1]:
        return boosts[-1]

    for i, (start, finish) in enumerate(pairwise(intervals[1:-2])):
        if hourly_rate in range(start, finish):
            return boosts[i + 2]
    return None


def get_interview_score(developer):
    """Get interview score of a developer."""
    interview_status = None
    interview_score = 0
    #     1	Fail
    #     2	Pass
    #     3	No Show
    #     4	Second No Show
    #     5	Redirected
    all_statuses = []
    for technical_interview in developer.get("technicalInterviews", []):
        score = technical_interview.get("score", 0) or 0
        all_statuses.append(technical_interview["status"])
        if score > interview_score:
            interview_score = score
            interview_status = technical_interview["status"]

    if 3 in all_statuses:
        #         if 2 in all_statuses:
        #             return interview_score // 1.5
        return -2
    if 4 in all_statuses:
        return -3

    if interview_score:
        if interview_status == 2:
            return interview_score
        if interview_status == 1:
            return interview_score // 2
    return interview_score if interview_score != 0 else np.nan


def get_interview_score_raw(developer):
    """Get raw interview score."""
    scores = [
        ti["scores"]
        for ti in developer.get("technicalInterviews", [])
        if ti["status"] in (1, 2, 5) and not np.isnan(ti.get("score", np.nan))
    ]
    return np.max(scores) if scores else np.nan


def get_mcq_score(developer_source, related_challenge_ids=None):
    """Get MCQ score."""
    max_percentile = 0
    if "skillScores" not in developer_source:
        return 0

    for mcq in developer_source["skillScores"]:
        if mcq.get("challengeId") not in related_challenge_ids:
            continue

        if mcq.get("percentile") > max_percentile:
            max_percentile = mcq.get("percentile")

    return max_percentile


def get_related_challenge_ids_for_skill(
    skill_name: str, mcq_name_to_id_dict: typing.Dict[str, int]
) -> typing.List[int]:
    """Given a skill name return the challenge IDs related to that skill."""
    skill_name_dict = {
        "react": "reactjs",
        "node.js": "nodejs",
        "vue.js": "vuejs",
        "android/kotlin": "android",
        "go/golang": "golang",
        "python(django)": "django",
        "rest/restful apis": "rest api",
        "firebase": "firebase database",
        "express.js": "express",
        "aws": "aws ",
    }

    if skill_name in ["ios - swift", "ios development"]:
        return [135]

    if skill_name in ("html", "css"):
        skill_name = "html/css"
    else:
        skill_name = skill_name_dict[skill_name]

    if skill_name not in mcq_name_to_id_dict:
        return []
    challenge_id = mcq_name_to_id_dict[skill_name]
    if not np.isnan(challenge_id):
        return [int(challenge_id)]

    return []


def get_skill_score_suggested(developer, skill_id):
    """Get suggested skill score."""
    rounded_project_yoe = 0.0
    self_declared_yoe = 0.0

    matchable_skills = developer.get("matchableSkills", [])
    for matchable_skill in matchable_skills:
        if skill_id not in [matchable_skill.get("id"), matchable_skill.get("skillId")]:
            continue

        yoe_keys = ["yearsOfExperience", "projectYearsOfExperience"]
        for yoe_key in yoe_keys:
            if yoe_key not in matchable_skill:
                continue
            rounded_project_yoe = matchable_skill[yoe_key]

    self_declared_skills = developer.get("selfDeclaredSkills", [])

    for self_declared_skill in self_declared_skills:
        if skill_id not in [
            self_declared_skill.get("id"),
            self_declared_skill.get("skillId"),
        ]:
            continue
        yoe_keys = ["score", "yearsOfExperience"]
        for yoe_key in yoe_keys:
            if yoe_key not in self_declared_skill:
                continue
            self_declared_yoe = self_declared_skill[yoe_key]

    skill_score = rounded_project_yoe
    if self_declared_yoe > 0:
        skill_score = min(rounded_project_yoe, self_declared_yoe)

    if np.isfinite(skill_score):
        return skill_score

    return 0


def get_seniority_score_suggested(developer, seniorities=[1, 2, 3, 4, 6]):
    """Get suggested seniority score."""
    # seniorities = ["IC3", "IC4", "IC5", "IC6", "IC7"]
    seniority_values = {
        1: 3,
        2: 4,
        3: 5,
        4: 6,
        6: 7,
    }
    seniority_id = developer.get("workExperienceSummary", {}).get(
        "seniorityId"
    ) or developer.get("workExperienceSeniorityId")
    if seniority_id not in seniorities:
        return 0

    return seniority_values.get(seniority_id, 0)


def get_mcq_pcts(developer_source, related_challenge_ids=None):
    """Get developer's MCQ percentiles (for all related challenges)."""
    if "skillScores" not in developer_source:
        return []

    if related_challenge_ids:
        return [
            d["percentile"]
            for d in developer_source["skillScores"]
            if d["challengeId"] in related_challenge_ids
        ]

    return [
        d["percentile"]
        for d in developer_source["skillScores"]
        if d["percentile"] is not None and d["challengeId"] != 143
    ]


def get_skills_yoe(
    developer,
    required_skill_ids: typing.List[int],
):
    """Get a developer' years of experience with respect to the required skills."""

    scores = []
    for skill_id in required_skill_ids:
        scores.append(get_skill_score_suggested(developer, skill_id))
    return scores


def get_developer_projects(developer_source, skill_ids=None):
    """Get a list of projects of the developer"""
    if skill_ids:
        return [
            p
            for p in developer_source.get("projects", [])
            if "skillIds" in p.keys()
            and (len(np.intersect1d(p["skillIds"], skill_ids)) > 0)
        ]
    return developer_source.get("projects", [])


def get_seniority_avg(developer_source):
    """Return the average seniority scores over 5 seniority categories."""
    if len(developer_source.get("seniorityCategories", [])) == 5:
        return np.mean(
            [d["seniorityAvgScore"] for d in developer_source["seniorityCategories"]]
        )
    return np.nan


def get_yoe(developer):
    """Get a developer's years of experience given her/his developer ID."""
    return (
        developer.get("calculatedYearsOfExperience", None)
        or developer.get("workExperienceYearsOfExperience", None)
        or np.nan
    )


##### elastic search feature helpers end ####

##### other features helpers ####


def generate_job_skills_features(skill_ids, skill_id_2_fea_name_dict):
    """Generate job-skills features."""
    skill_features_dict = {}
    for i in skill_id_2_fea_name_dict.values():
        skill_features_dict[i] = 0
    for skill_id in [i for i in skill_ids if i in skill_id_2_fea_name_dict]:
        skill_features_dict[skill_id_2_fea_name_dict[skill_id]] = 1
    return skill_features_dict


def generate_acc_historical_features(df_opp, created_ts):
    """Generate Automated Coding Challenge historical features."""
    return {
        "acc_opp_count": df_opp.loc[
            df_opp["opportunity_created_date"] < created_ts
        ].shape[0],
        "acc_rc_count": df_opp.loc[df_opp["chosen_date"] < created_ts].shape[0],
        "acc_trial_count": df_opp.loc[df_opp["trial_date"] < created_ts].shape[0],
        "acc_start_count": df_opp.loc[df_opp["start_date"] < created_ts].shape[0],
        "acc_fire_count": df_opp.loc[
            (df_opp["lost_date"] < created_ts) & (df_opp["is_fire"] == 1)
        ].shape[0],
        "acc_complete_count": df_opp.loc[df_opp["complete_date"] < created_ts].shape[0],
    }


def generate_dev_historical_features(full_df, created_ts):
    """Generate developer's historical features."""
    selfserve_criterion = full_df["serve_type"] == "1.self-serve"
    return {
        "dev_shortlist_count": full_df.loc[
            (full_df["mjm_create_ts"] < created_ts) & (~selfserve_criterion)
        ].shape[0],
        "dev_ps_count": full_df.loc[
            (full_df["serve_date"] < created_ts) & (~selfserve_criterion)
        ].shape[0],
        "dev_selfserve_count": full_df.loc[
            (full_df["mjm_create_ts"] < created_ts) & (selfserve_criterion)
        ].shape[0],
        "dev_selfserve_si_count": full_df.loc[
            (full_df["si_date"] < created_ts) & (selfserve_criterion)
        ].shape[0],
        "dev_si_count": full_df.loc[(full_df["si_date"] < created_ts)].shape[0],
        "dev_rc_count": full_df.loc[(full_df["chosen_date"] < created_ts)].shape[0],
        "dev_trial_count": full_df.loc[(full_df["trial_date"] < created_ts)].shape[0],
        "dev_start_count": full_df.loc[(full_df["start_date"] < created_ts)].shape[0],
        "dev_fire_count": full_df.loc[
            (full_df["lost_date"] < created_ts) & (full_df["is_fire"] == 1)
        ].shape[0],
        "dev_complete_count": full_df.loc[
            (full_df["complete_date"] < created_ts)
        ].shape[0],
    }


def read_feature_names(feature_set_file):
    """Read feature names from file."""
    with open(feature_set_file, "r", encoding="UTF-8") as file:
        feature_columns = file.read()
        feature_columns = feature_columns.split(",")
        feature_columns = [
            re.findall(r"'(.*?)'", x, re.DOTALL) for x in feature_columns
        ]
        feature_columns = [x[0] for x in feature_columns if len(x) > 0]
    return feature_columns
