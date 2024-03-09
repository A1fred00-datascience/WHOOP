# WHOOP Personal Health Analysis Project

## Overview
This project leverages data collected from a WHOOP device to perform a comprehensive analysis of personal health and wellness trends. By integrating and analyzing data on sleep patterns, workouts, journal entries, and physiological responses, the project aims to provide actionable insights into improving personal health outcomes.

## Data Description
The analysis is based on four primary datasets extracted from the WHOOP platform, covering journal entries, sleep analysis, workout logs, and physiological cycles. These datasets include metrics crucial for understanding overall wellness, such as sleep performance, recovery scores, and activity strain.

### Data Files
- **Journal Entries:** Insights into daily habits and subjective wellness assessments.
- **Sleep:** Metrics on sleep quality, disturbances, and recovery.
- **Workouts:** Details on exercise types, duration, and intensity.
- **Physiological Cycles:** Data on heart rate variability, resting heart rate, and more.

## Project Objectives and Steps
- Integrate data from various sources into a unique dataframe.
- Conduct exploratory data analysis to identify key health and wellness trend.
- Develop machine learning models to predict recovery scores based on lifestyle factors.
- Offer insights and recommendations for my own wellness improvement.
- Trespass the data into a Tableau dashboard in order to have an interactive board with insights


## Methodology
The project follows a structured approach, starting with data preprocessing and integration, followed by exploratory data analysis to uncover underlying patterns. Feature engineering identifies relevant predictors, which are then used to forecast recovery scores. The analysis concludes with an evaluation of feature importance to highlight the most influential factors on recovery.

### Key Insights
- **Sleep Patterns:** I explored how sleep quality correlates with recovery scores, identifying trends that suggest better sleep equals better recovery.
- **Workout Impact:** Analyzing workout intensity against recovery scores highlighted the fine balance between training hard and adequate rest.
- **Physiological Responses:** I examined how daily stressors and activities influence physiological markers like heart rate variability (HRV) and resting heart rate (RHR).

## Results
The analysis provides a multi-faceted view of my personal health, linking daily activities (Walking, lifting weights, gymnastics) and physiological metrics (Exposure to sunlight, hydration levels, REM sleep) to analyze recovery outcomes and pose recommendations on the mos important factors to take into account when trying to obtain a "Green" recovery.

## Technologies Used
- **Pandas & Numpy:** For data manipulation and analysis.
- **Matplotlib & Seaborn:** For data visualization.
- **Scikit-learn:** For machine learning model development.

## Running the Project
To run this project:
1. Ensure Python and all required libraries are installed.
2. Clone the repository to your local environment.
3. Execute `python WHOOP_Project.py` in your terminal.

## I am working on a Tableau dashboard at the moment to showcase my insights 

## Future Directions
Future enhancements may include integrating more diverse data sources, exploring additional predictive modeling techniques, and developing interactive dashboards for real-time health monitoring and insights.

## Acknowledgements
Thanks to WHOOP for the data that made this project possible. This project is for educational and self-improvement purposes.
