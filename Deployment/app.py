import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from feature_pipeline import build_onet_maps, extract_features, preserve_tech_syntax
from model_inference import get_salary_shap_data, predict_salary
from nlp_processing import align_resume_to_ontology, extract_text_from_pdf
from recommendation_engine import get_top_matches, get_single_job_score

st.set_page_config(
    page_title="Job Recommendation and Salary Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

CORE_SKILL_COUNT = 30


# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------

@st.cache_data
def load_pipeline_maps():
    df_occ_raw = pd.read_csv("Datasets/Unified_Occupation_Dataset.csv")
    valid_types = ['TechnologySkills', 'Skill', 'Knowledge']
    df_occ_clean = df_occ_raw[df_occ_raw['Feature_Type'].isin(valid_types)].copy()
    df_occ_clean['Clean_Feature'] = df_occ_clean['Feature_Name'].apply(preserve_tech_syntax)
    return build_onet_maps(df_occ_clean)


@st.cache_data
def load_ontology_data():
    occ_data = pd.read_csv("Datasets/Occupation Data.csv")
    unified_data = pd.read_csv("Datasets/Unified_Occupation_Dataset.csv")

    occ_data['Display_Name'] = occ_data['Title'] + " (" + occ_data['O*NET-SOC Code'] + ")"
    unified_data['Display_Name'] = unified_data['Title'] + " (" + unified_data['ONET_SOC_Code'] + ")"

    valid_jobs = set(unified_data['Display_Name'])
    occ_data = occ_data[occ_data['Display_Name'].isin(valid_jobs)]
    desc_map = occ_data.set_index('Display_Name')['Description'].to_dict()

    valid_types = ['TechnologySkills', 'Skill', 'Knowledge']
    req_skills_df = unified_data[unified_data['Feature_Type'].isin(valid_types)].copy()
    req_skills_df['Clean_Name'] = req_skills_df['Feature_Name'].apply(preserve_tech_syntax)

    skills_map = {}
    for name, group in req_skills_df.groupby('Display_Name'):
        unique_skills = group.sort_values('Importance_Score', ascending=False).drop_duplicates('Clean_Name')
        skills_map[name] = unique_skills[['Feature_Name', 'Clean_Name', 'Importance_Score', 'Feature_Type']]

    job_list = sorted(skills_map.keys())
    return job_list, desc_map, skills_map


@st.cache_data
def load_course_data():
    return pd.read_csv("Datasets/Unified_Course_Dataset.csv")


onet_skill_map, onet_core_map, onet_bench_map, onet_full_map = load_pipeline_maps()
job_list, desc_map, skills_map = load_ontology_data()
df_courses_master = load_course_data()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()


def calculate_hybrid_score(faiss_confidence: float, core_fit_decimal: float, model_weight=0.5, tech_weight=0.5):
    """Combine FAISS inner-product score with O*NET core fit score."""
    normalized_model_pct = max(0.0, faiss_confidence) * 100.0
    tech_fit_pct = core_fit_decimal * 100.0
    hybrid_score = (normalized_model_pct * model_weight) + (tech_fit_pct * tech_weight)
    return normalized_model_pct, tech_fit_pct, hybrid_score


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Portal:", ["Candidate Portal", "Ontology Dashboard"])
st.sidebar.divider()
st.sidebar.caption("Establishing a Data Driven Approach to Ontology Based Job Recommendation and Salary Prediction")


# ------------------------------------------------------------------
# Page 1: Candidate Portal
# ------------------------------------------------------------------

if page == "Candidate Portal":
    st.title("Candidate Job Matching Portal")
    st.markdown("Upload a resume to analyse O*NET competency alignment, predict annual median salary, and generate upskilling pathways.")

    # --- Section 1: Resume Input ---
    st.subheader("1. Resume Input")
    uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"], key="resume_uploader")

    with st.expander("⚠️ Important PDF Formatting Guidelines", expanded=False):
        st.warning("""
        **To ensure our NLP models can accurately extract your skills, please review these parsing limitations:**
        
        * **Standard PDFs Only:** The system strictly requires `.pdf` files. It will fail on `.docx`, `.png`, or password-protected/DRM-restricted files.
        * **No Scanned Documents (Zero OCR):** We rely on embedded text layers. If your resume was printed and scanned, or exported as a flattened image from tools like Canva/Photoshop, the ML will extract zero words. If you cannot highlight the text with your cursor, we cannot read it.
        * **Beware Multi-Column Layouts:** Our parser reads horizontally (left-to-right). Modern two-column resumes may cause the system to blindly combine words from the left column with the right column, breaking the context. Single-column layouts are strongly preferred.
        * **Avoid Invisible Tables:** Using hidden tables to align dates and Occupation Titles will cause the parser to flatten the text, potentially mixing your timeline out of order.
        * **Standard Fonts & Icons:** Highly customized fonts without proper Unicode mapping, or graphical bullet points (like FontAwesome icons), will be extracted as unreadable gibberish or empty boxes (□□□).
        """)

    if "raw_resume_text" not in st.session_state:
        st.session_state["raw_resume_text"] = ""
    if "last_file_hash" not in st.session_state:
        st.session_state["last_file_hash"] = None
    if "trigger_reparse" not in st.session_state:
        st.session_state["trigger_reparse"] = False

    if st.button("Parse / Re-Parse Resume"):
        st.session_state["raw_resume_text"] = ""
        st.session_state["trigger_reparse"] = True
        st.rerun()

    if uploaded_file is not None:
        current_hash = get_file_hash(uploaded_file)
        if st.session_state["trigger_reparse"] or current_hash != st.session_state["last_file_hash"]:
            with st.spinner("Extracting text..."):
                st.session_state["raw_resume_text"] = extract_text_from_pdf(uploaded_file)
            st.session_state["last_file_hash"] = current_hash
            st.session_state["trigger_reparse"] = False

    edited_resume_text = st.text_area("Review and Edit Parsed Text", key="raw_resume_text", height=200)

    # --- ANALYZE BUTTON ---
    if st.button("Analyze Resume", type="primary"):
        if not edited_resume_text.strip():
            st.error("Please provide resume text first.")
        else:
            with st.spinner("Extracting competency features and calculating market value..."):
                aligned_skills = align_resume_to_ontology(edited_resume_text)
                st.session_state["aligned_skills"] = aligned_skills
                st.session_state["aligned_skills_string"] = " ".join(aligned_skills)
                st.session_state["predicted_salary"] = predict_salary(st.session_state["aligned_skills_string"])
                st.session_state["salary_analyzed"] = True
                st.session_state["matching_complete"] = False
                st.session_state["selected_job"] = None  # reset target job on re-analyze

    # --- Section 2 & 3: Hidden until analysis is done ---
    if st.session_state.get("salary_analyzed", False):
        aligned_skills = st.session_state["aligned_skills"]
        aligned_skills_string = st.session_state["aligned_skills_string"]
        predicted_salary = st.session_state["predicted_salary"]

        # --- Section 2: Candidate Profile & Market Value ---
        st.divider()
        st.subheader("2. Candidate Profile & Market Value")

        col_kpi1, col_kpi2 = st.columns(2)
        col_kpi1.metric("Extracted O*NET Competencies", len(aligned_skills))
        col_kpi2.metric("Estimated Annual Median Salary based on Extracted Competencies", f"${predicted_salary:,.2f}")

        st.markdown("**Aligned Competencies Identified:**")
        formatted_skills = [skill.title() for skill in aligned_skills]
        st.info(" • ".join(formatted_skills) if formatted_skills else "No skills extracted.")

        st.markdown("#### Annual Median Salary Drivers (Based on Extracted Competencies)")
        st.caption("This SHAP waterfall plot breaks down exactly how much each specific competency feature increases or decreases your market value from the national baseline.")

        if aligned_skills:
            with st.spinner("Calculating SHAP..."):
                try:
                    base_val, final_val, df_shap = get_salary_shap_data(aligned_skills_string)

                    measure_list = ["absolute"] + ["relative"] * len(df_shap) + ["total"]
                    x_labels = ["National Baseline"] + df_shap['Skill'].tolist() + ["Your Predicted Median Salary"]
                    y_values = [base_val] + df_shap['Dollar_Impact'].tolist() + [final_val]

                    text_labels = [f"${base_val:,.0f}"] + \
                                  [f"{'+' if val > 0 else ''}${val:,.0f}" for val in df_shap['Dollar_Impact']] + \
                                  [f"${final_val:,.0f}"]

                    fig = go.Figure(go.Waterfall(
                        name="Salary Impact",
                        orientation="h",
                        measure=measure_list,
                        y=x_labels,
                        x=y_values,
                        textposition="outside",
                        text=text_labels,
                        hoverinfo="text",
                        hovertext=[
                            f"<b>National Baseline</b><br>Starting average for all roles: ${base_val:,.0f}"
                        ] + [
                            f"<b>{row['Skill']}</b><br>Market impact: {'+' if row['Dollar_Impact'] > 0 else ''}${row['Dollar_Impact']:,.0f}"
                            for _, row in df_shap.iterrows()
                        ] + [
                            f"<b>Final Predicted Salary</b><br>Total Estimated Value: ${final_val:,.0f}"
                        ],
                        connector={"line": {"color": "rgba(255, 255, 255, 0.1)", "width": 1, "dash": "dot"}},
                        decreasing={"marker": {"color": "#ff4b4b"}},
                        increasing={"marker": {"color": "#00cc96"}},
                        totals={"marker": {"color": "#1f77b4", "line": {"color": "white", "width": 1}}}
                    ))

                    fig.update_layout(
                        showlegend=False,
                        height=550,
                        margin=dict(l=20, r=60, t=30, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(autorange="reversed", title=""),
                        xaxis=dict(title="Estimated Annual Median Salary ($)", showgrid=True, gridcolor="rgba(255,255,255,0.1)")
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Could not generate SHAP plot: {e}")
        else:
            st.warning("No skills extracted. We need recognised competencies to calculate salary drivers.")

        # --- Section 3: Job Recommendation Engine ---
        st.divider()
        st.subheader("3. Job Recommendation Engine")

        # --- Workflow A: Discover Top 10 (runs automatically) ---
        st.markdown("#### Top 10 Occupation Matches")
        with st.spinner("Running FAISS retrieval and hybrid scoring..."):
            top_50_matches = get_top_matches(aligned_skills, top_k=50)

            ranking_features = []
            for rank, match in enumerate(top_50_matches, start=1):
                job_code = match['O*NET Code']
                c_len, overlap, pct, core_fit = extract_features(
                    aligned_skills, job_code, onet_skill_map, onet_bench_map, onet_full_map
                )
                ranking_features.append({
                    'Original_Match_Data': match,
                    'O*NET Code': job_code,
                    'FAISS_Score': match['Match Confidence'],
                    'Candidate_Skill_Count': c_len,
                    'Overlap_Count': overlap,
                    'Skill_Overlap_Pct': pct,
                    'Core_Fit_Score': core_fit
                })

            df_rank = pd.DataFrame(ranking_features)

            metrics = df_rank.apply(
                lambda row: calculate_hybrid_score(row['FAISS_Score'], row['Core_Fit_Score']),
                axis=1, result_type='expand'
            )
            df_rank[['Model_Confidence_Pct', 'Tech_Fit_Pct', 'Hybrid_Score']] = metrics

            final_top_10_df = df_rank.sort_values(by='Hybrid_Score', ascending=False).head(10)

            final_display_data = []
            for _, row in final_top_10_df.iterrows():
                match = row['Original_Match_Data']
                final_display_data.append({
                    "Occupation Title": match['Occupation Title'],
                    "O*NET Code": match['O*NET Code'],
                    "Sort_Value": row['Hybrid_Score'],
                    "Model Score": f"{row['Model_Confidence_Pct']:.1f}%",
                    "Job Fit Score": f"{row['Tech_Fit_Pct']:.1f}%",
                    "Hybrid Match Score": f"{row['Hybrid_Score']:.1f}%"
                })

        st.success("Candidate Retrieval Complete!")

        with st.expander("💡 How is your Hybrid Match Score calculated?", expanded=False):
            st.info(r"""
                Our architecture combines semantic retrieval with objective industry benchmarks:

                * **Model Score (50%):** Inner-product similarity from the fine-tuned MPNet Model index, normalised to a 0–100% range.
                * **Job Fit Score (50%):** Measures the weighted importance of the skills you possess relative to the Top 30 critical competencies for the occupation, sourced directly from O*NET.

                ---

                ### Mathematical Foundation

                **1. Model Score:**
                $$
                \text{Model Score} = \max(0, \text{Inner Product Score}) \times 100\%
                $$

                **2. Job Fit Score (O*NET Weighting):**
                $$
                \text{Job Fit Score} = \min\left(100\%, \frac{\sum \text{Importance of Matched Competencies}}{\sum \text{Importance of Top 30 Occupation Competencies}} \times 100 \right)
                $$

                **3. Final Hybrid Match Score:**
                $$
                \text{Score}_{\text{Hybrid}} = (0.5 \times \text{Model Score}) + (0.5 \times \text{Job Fit Score})
                $$
            """)

        if final_display_data:
            df_matches = pd.DataFrame(final_display_data)
            display_cols = ["Occupation Title", "O*NET Code", "Model Score", "Job Fit Score", "Hybrid Match Score"]
            st.dataframe(df_matches[display_cols], use_container_width=True, hide_index=True)

            df_matches_sorted = df_matches.sort_values(by="Sort_Value", ascending=True)
            fig_bar = px.bar(
                df_matches_sorted, x="Sort_Value", y="Occupation Title", orientation='h',
                title="Final Hybrid Match Score by Occupation",
                labels={'Sort_Value': 'Hybrid Match %'}
            )
            fig_bar.update_traces(marker_color='#1f77b4')
            fig_bar.update_layout(xaxis_title="Hybrid Match Score (%)", xaxis_range=[0, 100])
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Workflow B: Target Specific Occupation (appears after Top 10) ---
        st.divider()
        st.markdown("#### Target a Specific Occupation")
        st.caption("Select a role below to see a detailed comptency gap analysis and personalised upskilling pathway.")

        target_job = st.selectbox("Select Target Occupation", job_list, key="target_job_select")
        st.session_state["selected_job"] = target_job
        if target_job in desc_map:
            st.info(f"**Occupation Description:**\n\n{desc_map[target_job]}")

        if st.button("Analyse Target Occupation", type="secondary"):
            st.session_state["matching_complete"] = True

        if st.session_state.get("matching_complete", False):
            selected_job = st.session_state["selected_job"]
            target_job_code = selected_job.split('(')[-1].replace(')', '').strip()

            with st.spinner(f"Analysing Job Fit Score for {selected_job}..."):
                all_matches = get_top_matches(aligned_skills, top_k=1000)
                stage1_rank = 1000
                faiss_confidence = 0.0
                for rank, match in enumerate(all_matches, start=1):
                    if match['O*NET Code'] == target_job_code:
                        stage1_rank = rank
                        faiss_confidence = match['Match Confidence']
                        break

                c_len, overlap, pct, core_fit = extract_features(
                    aligned_skills, target_job_code,
                    onet_skill_map, onet_bench_map, onet_full_map
                )

                model_pct, tech_pct, hybrid_score = calculate_hybrid_score(faiss_confidence, core_fit)

                job_req_df = skills_map.get(selected_job, pd.DataFrame())
                core_matched_df = bonus_matched_df = missing_skills_df = pd.DataFrame()

                if not job_req_df.empty:
                    aligned_lower = set(preserve_tech_syntax(s) for s in aligned_skills)
                    job_req_df = job_req_df.copy()
                    job_req_df['Is_Matched'] = job_req_df['Clean_Name'].isin(aligned_lower)

                    top_core_skills = job_req_df.nlargest(CORE_SKILL_COUNT, 'Importance_Score')
                    non_core_skills = job_req_df.drop(top_core_skills.index)

                    core_matched_df = top_core_skills[top_core_skills['Is_Matched']].sort_values(by='Importance_Score', ascending=False)
                    bonus_matched_df = non_core_skills[non_core_skills['Is_Matched']].sort_values(by='Importance_Score', ascending=False)
                    missing_skills_df = top_core_skills[~top_core_skills['Is_Matched']].sort_values(by='Importance_Score', ascending=False)

            st.success("Analysis Complete!")

            tab1, tab2 = st.tabs(["📊 Job Fit Metrics", "📚 Upskilling Pathway"])

            with tab1:
                st.markdown(f"### Alignment Overview: {selected_job.split('(')[0]}")
                col1, col2 = st.columns(2)
                col1.metric("Hybrid Match Score", f"{hybrid_score:.1f}%")
                col2.metric("Missing Core Competencies", len(missing_skills_df))

                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=hybrid_score, domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Hybrid Match"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue" if hybrid_score > 70 else "darkorange"},
                        'steps': [{'range': [0, 50], 'color': "lightgray"}, {'range': [50, 80], 'color': "gray"}]
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=60, b=10))

                st.plotly_chart(fig, use_container_width=True)

                col_match, col_miss = st.columns(2)
                with col_match:
                    st.success(f"**Verified Core Competencies ({len(core_matched_df)}/{CORE_SKILL_COUNT}):**")
                    if not core_matched_df.empty:
                        st.dataframe(core_matched_df[['Feature_Name', 'Feature_Type', 'Importance_Score']], hide_index=True)
                    st.info(f"**Bonus Competencies ({len(bonus_matched_df)}):**")
                    if not bonus_matched_df.empty:
                        st.dataframe(bonus_matched_df[['Feature_Name', 'Feature_Type', 'Importance_Score']], hide_index=True)
                with col_miss:
                    st.error(f"**Missing Core Competencies ({len(missing_skills_df)}):**")
                    if not missing_skills_df.empty:
                        st.dataframe(missing_skills_df[['Feature_Name', 'Feature_Type', 'Importance_Score']], hide_index=True)

            with tab2:
                st.markdown("### Personalised Interactive Upskilling")
                if not missing_skills_df.empty:
                    with st.spinner("Simulating score and salary projections..."):
                        missing_skills_dict = dict(zip(missing_skills_df['Clean_Name'], missing_skills_df['Importance_Score']))
                        missing_skills_original_names = dict(zip(missing_skills_df['Clean_Name'], missing_skills_df['Feature_Name']))
                        course_recommendations = []

                        for index, row in df_courses_master.iterrows():
                            course_skills = set([s.strip().lower() for s in str(row.get('Skill_Text', '')).split(',') if s.strip()])
                            covered_skills = course_skills.intersection(set(missing_skills_dict.keys()))

                            if covered_skills:
                                simulated_skills = aligned_skills + list(covered_skills)
                                c_len, overlap, pct, sim_core_fit = extract_features(
                                    simulated_skills, target_job_code, onet_skill_map, onet_bench_map, onet_full_map
                                )

                                sim_faiss_score = get_single_job_score(simulated_skills, target_job_code)
                                _, _, projected_hybrid = calculate_hybrid_score(sim_faiss_score, sim_core_fit)
                                projected_hybrid = min(100.0, projected_hybrid)
                                actual_boost = projected_hybrid - hybrid_score

                                projected_salary = predict_salary(" ".join(simulated_skills))
                                salary_boost = projected_salary - predicted_salary

                                skill_details = [
                                    {"Target Competencies to Learn": missing_skills_original_names[sk], "Importance Score": missing_skills_dict[sk]}
                                    for sk in covered_skills
                                ]
                                course_recommendations.append({
                                    "Course Title": row.get('Title', 'Unknown'),
                                    "Platform": row.get('Platform', 'Unknown'),
                                    "Duration": row.get('Duration', 'N/A'),
                                    "Rating": row.get('Rating', 'N/A'),
                                    "Skill_Details": sorted(skill_details, key=lambda x: x['Importance Score'], reverse=True),
                                    "Projected_Hybrid": projected_hybrid,
                                    "Actual_Boost_Pct": actual_boost,
                                    "Projected_Salary": projected_salary,
                                    "Salary_Boost": salary_boost,
                                    "Covered_Keys": covered_skills
                                })

                        if course_recommendations:
                            course_recommendations.sort(key=lambda x: x['Actual_Boost_Pct'], reverse=True)
                            top_courses = course_recommendations[:5]
                            course_options = {f"Rank #{i+1}: {c['Course Title']} ({c['Platform']})": c for i, c in enumerate(top_courses)}

                            st.info("👇 **Select a recommended course below to preview your projected market improvements.**")
                            selected_course_key = st.selectbox("Target Upskilling Course:", options=list(course_options.keys()), label_visibility="collapsed")
                            selected_course = course_options[selected_course_key]

                            st.divider()
                            st.markdown(f"#### Recommended Online Course:")
                            st.markdown(f"#### {selected_course['Course Title']}")

                            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                            col_metric1.metric("Current Hybrid Match Score", f"{hybrid_score:.1f}%")
                            col_metric2.metric("Projected Hybrid Match Score", f"{selected_course['Projected_Hybrid']:.1f}%", f"+{selected_course['Actual_Boost_Pct']:.1f}% Boost")
                            col_metric3.metric("Projected Annual Median Salary", f"${selected_course['Projected_Salary']:,.2f}", f"+${selected_course['Salary_Boost']:,.2f}")
                            col_metric4.metric("Core Competencies Bridged", f"{len(selected_course['Covered_Keys'])}")

                            st.markdown("<br>", unsafe_allow_html=True)
                            col_info1, col_info2 = st.columns([1, 2])
                            with col_info1:
                                st.markdown(f"**Platform:** {selected_course['Platform']}")
                                st.markdown(f"**Est. Duration:** {selected_course['Duration']}")
                                st.markdown(f"**Rating:** ⭐ {selected_course['Rating']:.2f}")
                            with col_info2:
                                st.markdown("**Core Comptencies You Will Master:**")
                                df_course_skills = pd.DataFrame(selected_course['Skill_Details'])
                                df_course_skills['Importance Score'] = df_course_skills['Importance Score'].apply(lambda x: f"{x:.2f} / 5.0")
                                st.dataframe(df_course_skills, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No specific courses found covering your missing skills.")
                else:
                    st.success("🎉 You possess all critical core skills!")


# ------------------------------------------------------------------
# Page 2: Ontology Dashboard
# ------------------------------------------------------------------

elif page == "Ontology Dashboard":
    st.title("O*NET Ontology Competency Features Dashboard")
    st.markdown("A general dashboard of O*NET competency features information.")

    @st.cache_data
    def load_ontology_dashboard_data():
        df = pd.read_csv("Datasets/Unified_Occupation_Dataset.csv")
        valid_types = ['TechnologySkills', 'Skill', 'Knowledge']
        return df[df['Feature_Type'].isin(valid_types)].copy()

    df_ont = load_ontology_dashboard_data()

    st.markdown("### High-Level Taxonomy Metrics")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Occupations Profiled", f"{df_ont['ONET_SOC_Code'].nunique():,}")
    kpi2.metric("Total Unique Features", f"{df_ont['Feature_Name'].nunique():,}")
    kpi3.metric("Average Importance Score", f"{df_ont['Importance_Score'].mean():.2f} / 5.0")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        top_occ = (
            df_ont.groupby(['ONET_SOC_Code', 'Title'])
            .agg(
                Total_Competencies=('Feature_Name', 'nunique'),
                Avg_Importance=('Importance_Score', 'mean'),
                Max_Importance=('Importance_Score', 'max')
            )
            .reset_index()
            .sort_values('Total_Competencies', ascending=False)
            .head(10)
        )

        fig1 = px.bar(
            top_occ.sort_values('Total_Competencies', ascending=True),
            x='Total_Competencies', y='Title', orientation='h',
            title="Top 10 Occupations by Total Competency Features needed",
            template="plotly_dark",
            color='Total_Competencies',
            color_continuous_scale='Tealgrn',
            hover_data={
                'ONET_SOC_Code': True,
                'Avg_Importance': ':.2f',
                'Max_Importance': ':.2f',
                'Total_Competencies': True,
                'Title': False
            },
            labels={
                'Total_Competencies': 'Unique Competencies Required',
                'Avg_Importance': 'Avg Importance',
                'Max_Importance': 'Peak Importance',
                'ONET_SOC_Code': 'O*NET Code'
            }
        )
        fig1.update_layout(
            margin=dict(t=50, b=20, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            yaxis=dict(categoryorder='total ascending')
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        feature_density = df_ont.groupby('Title').size().reset_index(name='Feature_Count')
        fig2 = px.histogram(
            feature_density, x="Feature_Count", nbins=30,
            title="Taxonomy Depth (Features per Occupation)",
            labels={"Feature_Count": "Number of Required Skills/Features", "count": "Number of Occupations"},
            template="plotly_dark", color_discrete_sequence=["orange"]
        )
        fig2.update_layout(margin=dict(t=50, b=20, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        tech_df = df_ont[df_ont['Feature_Type'] == 'TechnologySkills']
        top_tech = tech_df.groupby('Feature_Name')['Importance_Score'].mean().sort_values(ascending=False).head(15).reset_index()
        fig3 = px.bar(
            top_tech.sort_values(by="Importance_Score", ascending=True),
            x="Importance_Score", y="Feature_Name", orientation='h',
            title="Top 15 Most Important Technology Skills (Global Avg)",
            template="plotly_dark", 
            color="Importance_Score", 
            color_continuous_scale="Blues"
        )
        fig3.update_layout(
            margin=dict(t=50, b=20, l=0, r=0), 
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
            coloraxis_showscale=False
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        skill_df = df_ont[df_ont['Feature_Type'] == 'Skill']
        top_skill = skill_df.groupby('Feature_Name')['Importance_Score'].mean().sort_values(ascending=False).head(15).reset_index()
        fig4 = px.bar(
            top_skill.sort_values(by="Importance_Score", ascending=True),
            x="Importance_Score", y="Feature_Name", orientation='h',
            title="Top 15 Most In-Demand Skills (Global Avg)",
            template="plotly_dark",
            color="Importance_Score",
            color_continuous_scale="Tealgrn"
        )
        fig4.update_layout(
            margin=dict(t=50, b=20, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:
        know_df = df_ont[df_ont['Feature_Type'] == 'Knowledge']
        top_know = know_df.groupby('Feature_Name')['Importance_Score'].mean().sort_values(ascending=False).head(15).reset_index()
        fig5 = px.scatter(
            top_know, x="Importance_Score", y="Feature_Name",
            size="Importance_Score", color="Importance_Score",
            title="Top 15 Most Critical Knowledge Domains (Global Avg)",
            template="plotly_dark", color_continuous_scale="ylorrd",
            size_max=18
        )
        fig5.update_layout(
            margin=dict(t=50, b=20, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            yaxis=dict(categoryorder='total ascending')
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.box(
            df_ont, x="Feature_Type", y="Importance_Score", color="Feature_Type",
            title="Importance Score Distribution by Type",
            template="plotly_dark"
        )
        fig6.update_layout(showlegend=False, margin=dict(t=50, b=20, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig6, use_container_width=True)