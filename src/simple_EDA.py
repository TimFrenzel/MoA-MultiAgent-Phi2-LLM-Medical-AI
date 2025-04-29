"""
Author: Tim Frenzel
Version: 1.3
Usage:  python src/simple_EDA.py

Objective of the Code:
------------
Performs Exploratory Data Analysis (EDA) on the full Synthea dataset stored
in the DuckDB database (synthea.duckdb), which references the merged Parquet files.
Connects to the database, runs queries to analyze distributions and statistics
for potential features (X) and the target variable (y - condition codes),
generates plots using matplotlib/seaborn, and saves outputs to 'results/eda'.
Focuses on efficiency by performing aggregations within DuckDB where possible.

Data Description:
------------
- **Dataset:** The analysis targets the full Synthea dataset available within the
  `synthea.duckdb` database located at `data/duckdb/`.
- **Patient Records:** The exact number of synthetic patient records is determined
  dynamically by querying the 'patients' table (see `get_table_counts` function).
- **Demographics:** Patient demographics (age, gender) are analyzed and visualized
  by the `analyze_patient_demographics` function. Results are saved in the
  'results/eda' directory.
- **Clinical Data Included:** The analysis considers data from the following tables:
    - patients
    - encounters
    - conditions (potential target variable 'y')
    - observations
    - medications
    - procedures
    - allergies
    - immunizations
- **Subsetting/Selection:** No specific subset or pre-selection criteria are applied
  to the data *before* analysis begins within this script. The script aims to provide
  an overview of the entire dataset as present in the DuckDB file. Filtering may
  occur within specific analysis functions (e.g., removing null values).
"""
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import numpy as np # For numeric checks
import sys
from datetime import datetime

# --- Configuration ---
ROOT        = pathlib.Path(__file__).resolve().parents[1]
DB_PATH     = ROOT / "data" / "duckdb" / "synthea.duckdb"
RESULTS_DIR = ROOT / "results" / "eda"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Database Path: {DB_PATH}")
print(f"Results Directory: {RESULTS_DIR}")

# --- Database Connection ---
try:
    # Connect in read-only mode
    con = duckdb.connect(database=str(DB_PATH), read_only=True)
    print("Successfully connected to DuckDB.")
except Exception as e:
    print(f"[ERROR] Connecting to DuckDB: {e}")
    sys.exit(1)

# --- EDA Functions ---

def save_plot(fig, filename):
    """Helper function to save plots."""
    try:
        filepath = RESULTS_DIR / filename
        fig.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {filepath}")
    except Exception as e:
        print(f"[ERROR] Saving plot {filename}: {e}")

def get_table_counts(connection):
    """Get row counts for key tables."""
    counts = {}
    # Updated list based on final available tables
    tables_to_count = [
        'patients', 'encounters', 'conditions', 'observations',
        'medications', 'procedures', 'allergies', 'immunizations'
    ]
    print("\n--- Table Counts ---")
    for table in tables_to_count:
        try:
            # Use DESCRIBE to check if table/view exists first
            connection.execute(f'DESCRIBE {table}')
            count = connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            counts[table] = count
            print(f"  {table}: {count:,}")
        except duckdb.CatalogException:
            print(f"  {table}: View not found (Skipped)")
        except Exception as e:
            print(f"[ERROR] Could not get count for {table}: {e}")
    return counts

def analyze_patient_demographics(connection):
    """Analyze patient age and gender.
       Uses columns: ID, BIRTHDATE, GENDER from patients view.
    """
    print("\n--- Analyzing Patient Demographics ---")
    try:
        # Using current date to calculate age. Can be refined based on data timestamps.
        # Fetch necessary columns only
        query = f"""
            SELECT GENDER, BIRTHDATE
            FROM patients
            WHERE BIRTHDATE IS NOT NULL AND GENDER IS NOT NULL
        """
        patients_df = connection.execute(query).df()

        if patients_df.empty:
            print("  No valid patient demographic data found.")
            return
            
        # Calculate age efficiently in pandas, coercing date errors
        patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'], errors='coerce')
        # Drop rows where date conversion failed
        patients_df = patients_df.dropna(subset=['BIRTHDATE'])
        
        today = pd.Timestamp('today')
        patients_df['Age'] = (today - patients_df['BIRTHDATE']).dt.days / 365.25
        patients_df = patients_df[patients_df['Age'] >= 0] # Filter out potential future dates or calculation errors

        if patients_df.empty:
            print("  No valid patient demographic data remaining after date cleaning.")
            return

        print(f"  Number of patients analyzed: {len(patients_df):,}")

        # Age Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(patients_df['Age'], bins=30, kde=False)
        plt.title('Patient Age Distribution')
        plt.xlabel('Age (Years)')
        plt.ylabel('Number of Patients')
        save_plot(plt.gcf(), 'patient_age_distribution.png')

        # Gender Distribution
        plt.figure(figsize=(6, 4))
        gender_counts = patients_df['GENDER'].value_counts().reset_index()
        gender_counts.columns = ['GENDER', 'count']
        sns.barplot(data=gender_counts, x='GENDER', y='count')
        plt.title('Patient Gender Distribution')
        plt.xlabel('Gender')
        plt.ylabel('Number of Patients')
        save_plot(plt.gcf(), 'patient_gender_distribution.png')

    except Exception as e:
        print(f"[ERROR] Analyzing patient demographics: {e}")

def analyze_encounters(connection, top_n=20):
    """Analyze encounter types based on DESCRIPTION.
       Uses columns: DESCRIPTION from encounters view.
    """
    print(f"\n--- Analyzing Top {top_n} Encounter Descriptions ---")
    try:
        query = f"""
            SELECT DESCRIPTION, COUNT(*) as count 
            FROM encounters 
            WHERE DESCRIPTION IS NOT NULL
            GROUP BY DESCRIPTION 
            ORDER BY count DESC
            LIMIT {top_n}
        """
        encounters_df = connection.execute(query).df()

        if encounters_df.empty:
            print("  No encounter data found or DESCRIPTION is null.")
            return

        plt.figure(figsize=(12, 8))
        sns.barplot(data=encounters_df, y='DESCRIPTION', x='count', orient='h')
        plt.title(f'Top {top_n} Most Frequent Encounter Descriptions')
        plt.xlabel('Number of Encounters')
        plt.ylabel('Encounter Description')
        save_plot(plt.gcf(), 'encounter_description_distribution.png')

    except Exception as e:
        print(f"[ERROR] Analyzing encounters: {e}")

def analyze_conditions_per_patient(connection):
    """Analyze the number of conditions per patient.
       Uses columns: PATIENT from conditions view.
    """
    print("\n--- Analyzing Conditions per Patient ---")
    try:
        query = """
            SELECT COUNT(*) AS condition_count
            FROM conditions
            GROUP BY PATIENT
        """
        # Fetch only the aggregated counts
        conditions_per_patient_df = connection.execute(query).df()

        if conditions_per_patient_df.empty:
            print("  No condition data found.")
            return

        # Save summary statistics
        summary_stats = conditions_per_patient_df['condition_count'].describe()
        summary_stats_df = pd.DataFrame(summary_stats)
        csv_path = RESULTS_DIR / 'conditions_per_patient_summary.csv'
        summary_stats_df.to_csv(csv_path)
        print(f"  Saved condition summary stats: {csv_path}")
        print(summary_stats)

        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(conditions_per_patient_df['condition_count'], bins=50, kde=False)
        plt.title('Distribution of Number of Conditions per Patient')
        plt.xlabel('Number of Conditions')
        plt.ylabel('Number of Patients')
        plt.yscale('log') # Use log scale due to skew
        save_plot(plt.gcf(), 'conditions_per_patient_distribution.png')

    except Exception as e:
        print(f"[ERROR] Analyzing conditions per patient: {e}")

def analyze_condition_codes(connection, top_n=20):
    """Analyze the most frequent condition codes/descriptions (TARGET VARIABLE).
       Uses columns: DESCRIPTION, CODE from conditions view.
    """
    print(f"\n--- Analyzing Top {top_n} Condition Codes/Descriptions (Potential Target y) ---")
    try:
        # Using DESCRIPTION for readability
        query = f"""
            SELECT DESCRIPTION, CODE, COUNT(*) AS count
            FROM conditions
            WHERE DESCRIPTION IS NOT NULL
            GROUP BY DESCRIPTION, CODE
            ORDER BY count DESC
            LIMIT {top_n}
        """
        condition_codes_df = connection.execute(query).df()

        if condition_codes_df.empty:
            print("  No condition data found to analyze codes.")
            return

        print(f"  Top {top_n} conditions:")
        print(condition_codes_df)

        # Plot distribution
        plt.figure(figsize=(12, 8))
        # Shorten long descriptions for plot labels
        condition_codes_df['PLOT_LABEL'] = condition_codes_df['DESCRIPTION'].str[:50] + '...'
        sns.barplot(data=condition_codes_df, y='PLOT_LABEL', x='count', orient='h')
        plt.title(f'Top {top_n} Most Frequent Condition Descriptions')
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Condition Description (Truncated)')
        save_plot(plt.gcf(), 'top_condition_codes_distribution.png')

    except Exception as e:
        print(f"[ERROR] Analyzing condition codes: {e}")

def analyze_observations(connection, top_n=20):
    """Analyze observations: top descriptions and value distribution for one key observation.
       Uses columns: DESCRIPTION, VALUE, UNITS from observations view.
    """
    print(f"\n--- Analyzing Top {top_n} Observation Descriptions ---")
    try:
        # Top N Observation Descriptions
        query_desc = f"""
            SELECT DESCRIPTION, COUNT(*) as count
            FROM observations
            WHERE DESCRIPTION IS NOT NULL
            GROUP BY DESCRIPTION
            ORDER BY count DESC
            LIMIT {top_n}
        """
        obs_desc_df = connection.execute(query_desc).df()

        if not obs_desc_df.empty:
            plt.figure(figsize=(12, 8))
            obs_desc_df['PLOT_LABEL'] = obs_desc_df['DESCRIPTION'].str[:50] + '...'
            sns.barplot(data=obs_desc_df, y='PLOT_LABEL', x='count', orient='h')
            plt.title(f'Top {top_n} Most Frequent Observation Descriptions')
            plt.xlabel('Number of Records')
            plt.ylabel('Observation Description (Truncated)')
            save_plot(plt.gcf(), 'top_observation_descriptions.png')
        else:
            print("  No observation descriptions found.")

        # Analyze distribution of a specific numeric observation (e.g., Body Height)
        obs_name = 'Body Height'
        print(f"\n--- Analyzing Value Distribution for '{obs_name}' Observation ---")
        query_value = f"""
            SELECT VALUE
            FROM observations
            WHERE DESCRIPTION = '{obs_name}'
              AND VALUE IS NOT NULL
        """
        obs_value_df = connection.execute(query_value).df()
        
        # Attempt to convert VALUE to numeric, coercing errors
        obs_value_df['VALUE_NUMERIC'] = pd.to_numeric(obs_value_df['VALUE'], errors='coerce')
        obs_value_df = obs_value_df.dropna(subset=['VALUE_NUMERIC']) # Remove non-numeric

        if not obs_value_df.empty:
            print(f"  Number of numeric '{obs_name}' records: {len(obs_value_df):,}")
            print(f"  Value Summary:")
            print(obs_value_df['VALUE_NUMERIC'].describe())
            
            plt.figure(figsize=(10, 6))
            sns.histplot(obs_value_df['VALUE_NUMERIC'], bins=50, kde=False)
            plt.title(f'Distribution of \'{obs_name}\' Values')
            plt.xlabel('Value') # Units might vary, check UNITS column if needed
            plt.ylabel('Frequency')
            save_plot(plt.gcf(), f'{obs_name.lower().replace(" ", "_")}_distribution.png')
        else:
            print(f"  No numeric data found for observation: '{obs_name}'")
            
    except Exception as e:
        print(f"[ERROR] Analyzing observations: {e}")

def analyze_medications(connection, top_n=20):
    """Analyze the most frequent medications.
       Uses columns: DESCRIPTION from medications view.
    """
    print(f"\n--- Analyzing Top {top_n} Medications ---")
    try:
        query = f"""
            SELECT DESCRIPTION, COUNT(*) AS count
            FROM medications
            WHERE DESCRIPTION IS NOT NULL
            GROUP BY DESCRIPTION
            ORDER BY count DESC
            LIMIT {top_n}
        """
        medications_df = connection.execute(query).df()

        if medications_df.empty:
            print("  No medication data found.")
            return

        plt.figure(figsize=(12, 8))
        medications_df['PLOT_LABEL'] = medications_df['DESCRIPTION'].str[:50] + '...'
        sns.barplot(data=medications_df, y='PLOT_LABEL', x='count', orient='h')
        plt.title(f'Top {top_n} Most Frequent Medication Descriptions')
        plt.xlabel('Number of Records')
        plt.ylabel('Medication Description (Truncated)')
        save_plot(plt.gcf(), 'top_medications_distribution.png')

    except Exception as e:
        print(f"[ERROR] Analyzing medications: {e}")

def analyze_procedures(connection, top_n=20):
    """Analyze the most frequent procedures.
       Uses columns: DESCRIPTION from procedures view.
    """
    print(f"\n--- Analyzing Top {top_n} Procedures ---")
    try:
        query = f"""
            SELECT DESCRIPTION, COUNT(*) AS count
            FROM procedures
            WHERE DESCRIPTION IS NOT NULL
            GROUP BY DESCRIPTION
            ORDER BY count DESC
            LIMIT {top_n}
        """
        procedures_df = connection.execute(query).df()

        if procedures_df.empty:
            print("  No procedure data found.")
            return

        plt.figure(figsize=(12, 8))
        procedures_df['PLOT_LABEL'] = procedures_df['DESCRIPTION'].str[:50] + '...'
        sns.barplot(data=procedures_df, y='PLOT_LABEL', x='count', orient='h')
        plt.title(f'Top {top_n} Most Frequent Procedure Descriptions')
        plt.xlabel('Number of Records')
        plt.ylabel('Procedure Description (Truncated)')
        save_plot(plt.gcf(), 'top_procedures_distribution.png')

    except Exception as e:
        print(f"[ERROR] Analyzing procedures: {e}")
        
def analyze_allergies(connection, top_n=20):
    """Analyze the most frequent allergies.
       Uses columns: DESCRIPTION from allergies view.
    """
    print(f"\n--- Analyzing Top {top_n} Allergies ---")
    try:
        query = f"""
            SELECT DESCRIPTION, COUNT(*) as count
            FROM allergies
            WHERE DESCRIPTION IS NOT NULL
            GROUP BY DESCRIPTION
            ORDER BY count DESC
            LIMIT {top_n}
        """
        allergies_df = connection.execute(query).df()

        if allergies_df.empty:
            print("  No allergy data found.")
            return

        plt.figure(figsize=(12, 8))
        allergies_df['PLOT_LABEL'] = allergies_df['DESCRIPTION'].str[:50] + '...'
        sns.barplot(data=allergies_df, y='PLOT_LABEL', x='count', orient='h')
        plt.title(f'Top {top_n} Most Frequent Allergy Descriptions')
        plt.xlabel('Number of Records')
        plt.ylabel('Allergy Description (Truncated)')
        save_plot(plt.gcf(), 'top_allergies_distribution.png')

    except Exception as e:
        print(f"[ERROR] Analyzing allergies: {e}")

def analyze_immunizations(connection, top_n=20):
    """Analyze the most frequent immunizations (by CODE).
       Uses columns: CODE from immunizations view.
    """
    print(f"\n--- Analyzing Top {top_n} Immunization Codes ---")
    try:
        query = f"""
            SELECT CODE, COUNT(*) as count
            FROM immunizations
            WHERE CODE IS NOT NULL
            GROUP BY CODE
            ORDER BY count DESC
            LIMIT {top_n}
        """
        immunizations_df = connection.execute(query).df()

        if immunizations_df.empty:
            print("  No immunization data found.")
            return

        plt.figure(figsize=(12, 8))
        # Convert code to string for plotting
        immunizations_df['CODE_STR'] = immunizations_df['CODE'].astype(str)
        sns.barplot(data=immunizations_df, y='CODE_STR', x='count', orient='h')
        plt.title(f'Top {top_n} Most Frequent Immunization Codes')
        plt.xlabel('Number of Records')
        plt.ylabel('Immunization Code')
        save_plot(plt.gcf(), 'top_immunization_codes.png')

    except Exception as e:
        print(f"[ERROR] Analyzing immunizations: {e}")

def analyze_age_by_encounter(connection, top_n=10):
    """Analyze patient age distribution by encounter description.
       Uses columns: ID, BIRTHDATE from patients; PATIENT, DATE, DESCRIPTION from encounters.
    """
    print(f"\n--- Analyzing Age by Top {top_n} Encounter Descriptions ---")
    try:
        # Get top N encounter descriptions first for efficiency
        top_enc_query = f"""
            SELECT DESCRIPTION 
            FROM encounters 
            WHERE DESCRIPTION IS NOT NULL
            GROUP BY DESCRIPTION 
            ORDER BY count(*) DESC 
            LIMIT {top_n}
        """
        top_enc_df = connection.execute(top_enc_query).df()
        if top_enc_df.empty:
            print("  No top encounter descriptions found to analyze.")
            return
        top_descriptions = top_enc_df['DESCRIPTION'].tolist()
        
        # Parameterized query is safer and handles quoting
        # Create placeholders like ($1, $2, ...)
        placeholders = ", ".join([f'${i+1}' for i in range(len(top_descriptions))])
        # *** Alias table in filter clause ***
        filter_clause = f"ec.DESCRIPTION IN ({placeholders})" 
        params = top_descriptions # Pass the list as parameters

        # Calculate age at the time of encounter for top encounters
        # *** Use try_cast for dates and filter NULLs using CTEs ***
        query = f"""
        WITH PatientCasts AS (
            SELECT ID, try_cast(BIRTHDATE AS DATE) as CastBirthDate
            FROM patients
            WHERE try_cast(BIRTHDATE AS DATE) IS NOT NULL -- Pre-filter invalid patient dates
        ), EncounterCasts AS (
            SELECT PATIENT, try_cast(DATE AS DATE) as CastEncounterDate, DESCRIPTION
            FROM encounters
            WHERE try_cast(DATE AS DATE) IS NOT NULL -- Pre-filter invalid encounter dates
        )
        SELECT 
            ec.DESCRIPTION,
            DATE_DIFF('year', pc.CastBirthDate, ec.CastEncounterDate) AS AgeAtEncounter
        FROM EncounterCasts ec
        JOIN PatientCasts pc ON ec.PATIENT = pc.ID
        WHERE ec.CastEncounterDate >= pc.CastBirthDate -- Ensure encounter is on or after birth
          AND {filter_clause} -- Apply description filter
        """
        # Execute with parameters
        age_encounter_df = connection.execute(query, params).df()
        
        # Filter out negative ages if any date issues (redundant now but safe)
        age_encounter_df = age_encounter_df[age_encounter_df['AgeAtEncounter'] >= 0]

        if age_encounter_df.empty:
             print("  No data found for age by encounter description analysis (after filtering).")
             return
            
        print(f"  Analyzing {len(age_encounter_df):,} encounter records.")

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=age_encounter_df, y='DESCRIPTION', x='AgeAtEncounter', 
                    orient='h', order=top_descriptions)
        plt.title(f'Patient Age Distribution by Top {top_n} Encounter Descriptions')
        plt.xlabel('Age at Encounter (Years)')
        plt.ylabel('Encounter Description')
        save_plot(plt.gcf(), 'age_by_encounter_description.png')

    except Exception as e: # Ensure except block exists
        print(f"[ERROR] Analyzing age by encounter description: {e}")

def analyze_conditions_vs_encounters(connection):
    """Analyze the relationship between number of conditions and encounters per patient.
       Uses columns: PATIENT from conditions, PATIENT from encounters.
    """
    print("\n--- Analyzing Conditions vs Encounters per Patient ---")
    try:
        # Use CTEs for clarity, perform aggregation in DuckDB
        query = """
        WITH ConditionsPerPatient AS (
            SELECT PATIENT, COUNT(*) AS num_conditions
            FROM conditions
            GROUP BY PATIENT
        ),
        EncountersPerPatient AS (
            SELECT PATIENT, COUNT(*) AS num_encounters
            FROM encounters
            GROUP BY PATIENT
        )
        SELECT
            cpp.num_conditions,
            epp.num_encounters
        FROM ConditionsPerPatient cpp
        JOIN EncountersPerPatient epp ON cpp.PATIENT = epp.PATIENT;
        """
        # Fetch only the counts
        combined_df = connection.execute(query).df()

        if combined_df.empty:
            print("  No data found for conditions vs encounters analysis (possibly due to join needing patients with both). Consider FULL OUTER JOIN if needed.")
            return
            
        print(f"  Analyzing {len(combined_df):,} patients with both conditions and encounters.")

        # Scatter plot (consider hexbin for large N)
        plt.figure(figsize=(10, 7))
        # Use alpha for transparency due to potential overplotting
        sns.scatterplot(data=combined_df, x='num_conditions', y='num_encounters', alpha=0.1, s=10) 
        plt.title('Number of Encounters vs. Number of Conditions per Patient')
        plt.xlabel('Number of Unique Conditions')
        plt.ylabel('Number of Encounters')
        # Optional: Use log scale if distributions are heavily skewed
        # plt.xscale('log')
        # plt.yscale('log')
        save_plot(plt.gcf(), 'conditions_vs_encounters_scatter.png')

    except Exception as e:
        print(f"[ERROR] Analyzing conditions vs encounters: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"SIMPLE EDA SCRIPT".center(80))
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80 + "\n")

    # --- Run Analyses ---
    get_table_counts(con)
    analyze_patient_demographics(con)
    analyze_encounters(con)
    analyze_conditions_per_patient(con)
    analyze_condition_codes(con) # Focus on target variable (y)
    analyze_observations(con)    # Potential features (X)
    analyze_medications(con)     # Potential features (X)
    analyze_procedures(con)      # Potential features (X)
    analyze_allergies(con)       # Potential features (X)
    analyze_immunizations(con)   # Potential features (X)
    analyze_age_by_encounter(con)
    analyze_conditions_vs_encounters(con)

    # --- Close Connection ---
    try:
        con.close()
        print("\nDuckDB connection closed.")
    except Exception as e:
        print(f"[ERROR] Closing DuckDB connection: {e}")

    print("\n" + "="*80)
    print(f"EDA SCRIPT FINISHED".center(80))
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80 + "\n") 