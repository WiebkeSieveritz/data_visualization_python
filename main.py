import pandas as pd
import numpy as np
import statistics


# import data set
path = (r'C:\Users\wsiev\OneDrive\Desktop\IU - Data Analyst Python\3. Modul - Explorative Data Analysis and Visualization\Task - neu\Salary Data.csv')
salary_data = pd.read_csv(path)
salary_data = salary_data.dropna()


# Shares of total
share_bachelor = round(len(salary_data[salary_data['Education Level'] == "Bachelor's"])/len(salary_data['Education Level'])*100, 2)
share_master = round(len(salary_data[salary_data['Education Level'] == "Master's"])/len(salary_data['Education Level'])*100, 2)
share_phd = round(len(salary_data[salary_data['Education Level'] == "PhD"])/len(salary_data['Education Level'])*100,2)
share_male = round(len(salary_data[salary_data['Gender'] == "Male"])/len(salary_data['Gender'])*100, 2)
share_female = round(len(salary_data[salary_data['Gender'] == "Female"])/len(salary_data['Gender'])*100,2)

# arithmetic mean (total, male and female):
arith_mean_total = round(sum(salary_data['Salary'])/(len(salary_data['Salary'])), 2)
# using library pandas:
# average_salary = salary_data['Salary'].mean()

arith_mean_male_sum = 0
arith_mean_male_len = 0
for index, row in salary_data.iterrows():
    if row['Gender'] == 'Male':
        arith_mean_male_len += 1
        arith_mean_male_sum += row['Salary']
if arith_mean_male_len > 0:
    arith_mean_male = arith_mean_male_sum / arith_mean_male_len
# using library pandas:
# male_salary_data = salary_data[salary_data['Gender'] == 'Male']
# average_salary_male = male_salary_data['Salary'].mean()

arith_mean_female_sum = 0
arith_mean_female_len = 0
for index, row in salary_data.iterrows():
    if row['Gender'] == 'Female':
        arith_mean_female_len += 1
        arith_mean_female_sum += row['Salary']
if arith_mean_female_len > 0:
    arith_mean_female = arith_mean_female_sum / arith_mean_female_len
# using library pandas:
# female_salary_data = salary_data[salary_data['Gender'] == 'Female']
# average_salary_female = female_salary_data['Salary'].mean()


# arithmetic mean (education level - total, male and female):
bachelor_male_salaries = salary_data[(salary_data['Gender'] == 'Male') & (salary_data['Education Level'] == "Bachelor's")]['Salary']
bachelor_female_salaries = salary_data[(salary_data['Gender'] == 'Female') & (salary_data['Education Level'] == "Bachelor's")]['Salary']
bachelor_total_salaries = salary_data[salary_data['Education Level'] == "Bachelor's"]['Salary']
master_male_salaries = salary_data[(salary_data['Gender'] == 'Male') & (salary_data['Education Level'] == "Master's")]['Salary']
master_female_salaries = salary_data[(salary_data['Gender'] == 'Female') & (salary_data['Education Level'] == "Master's")]['Salary']
master_total_salaries = salary_data[salary_data['Education Level'] == "Master's"]['Salary']
phd_male_salaries = salary_data[(salary_data['Gender'] == 'Male') & (salary_data['Education Level'] == "PhD")]['Salary']
phd_female_salaries = salary_data[(salary_data['Gender'] == 'Female') & (salary_data['Education Level'] == "PhD")]['Salary']
phd_total_salaries = salary_data[salary_data['Education Level'] == "PhD"]['Salary']


arith_mean_bachelor_male = np.mean(bachelor_male_salaries)
arith_mean_bachelor_female = np.mean(bachelor_female_salaries)
arith_mean_bachelor_total = np.mean(bachelor_total_salaries)
arith_mean_master_male = np.mean(master_male_salaries)
arith_mean_master_female = np.mean(master_female_salaries)
arith_mean_master_total = np.mean(master_total_salaries)
arith_mean_phd_male = np.mean(phd_male_salaries)
arith_mean_phd_female = np.mean(phd_female_salaries)
arith_mean_phd_total = np.mean(phd_total_salaries)

# print(arith_mean_male, arith_mean_female, arith_mean_total, arith_mean_bachelor_male, arith_mean_master_male, arith_mean_phd_male, arith_mean_bachelor_female, arith_mean_master_female, arith_mean_phd_female, arith_mean_bachelor_total, arith_mean_master_total, arith_mean_phd_total)

# median (total, male and female):
salaries = salary_data['Salary'].values
sorted_salaries = np.sort(salaries)
n_total = len(sorted_salaries)
if n_total % 2 == 0:
    median1 = sorted_salaries[n_total//2]
    median2 = sorted_salaries[n_total//2 - 1]
    median_total = (median1 + median2)/2
else:
    median_total = sorted_salaries[n_total//2]
# using library numpy:
# median_total = np.median(salaries)

male_salaries = salary_data[salary_data['Gender'] == 'Male']['Salary']
sorted_male_salaries = male_salaries.sort_values()
n_male = len(sorted_male_salaries)
if n_male % 2 == 0:
    median1 = sorted_male_salaries.iloc[n_male // 2]
    median2 = sorted_male_salaries.iloc[n_male // 2 - 1]
    median_male = (median1 + median2) / 2
else:
    median_male = sorted_male_salaries.iloc[n_male // 2]
# using library numpy:
# male_salaries = salary_data[salary_data['Gender'] == 'Male']['Salary']
# median_male = np.median(male_salaries)

female_salaries = salary_data[salary_data['Gender'] == 'Female']['Salary']
sorted_female_salaries = female_salaries.sort_values()
n_female = len(sorted_female_salaries)
if n_female % 2 == 0:
    median1 = sorted_female_salaries.iloc[n_female // 2]
    median2 = sorted_female_salaries.iloc[n_female // 2 - 1]
    median_female = (median1 + median2) / 2
else:
    median_female = sorted_female_salaries.iloc[n_female // 2]
# using library numpy:
# female_salaries = salary_data[salary_data['Gender'] == 'Female']['Salary']
# median_female = np.median(female_salaries)


# median (education level - total, male and female):
median_bachelor_male = np.median(bachelor_male_salaries)
median_bachelor_female = np.median(bachelor_female_salaries)
median_bachelor_total = np.median(bachelor_total_salaries)
median_master_male = np.median(master_male_salaries)
median_master_female = np.median(master_female_salaries)
median_master_total = np.median(master_total_salaries)
median_phd_male = np.median(phd_male_salaries)
median_phd_female = np.median(phd_female_salaries)
median_phd_total = np.median(phd_total_salaries)

# print(median_male, median_female, median_total, median_bachelor_male, median_master_male, median_phd_male, median_bachelor_female, median_master_female, median_phd_female, median_bachelor_total, median_master_total, median_phd_total)

# trimmed mean (total, male and female):
from scipy.stats import trim_mean
trimmed_mean_total = trim_mean(salaries, proportiontocut=0.1)

trimmed_mean_male = trim_mean(male_salaries, proportiontocut=0.1)

trimmed_mean_female = trim_mean(female_salaries, proportiontocut=0.1)
# without using a library:
# sorted_salaries = sorted(salaries)
# trim_count = int(len(sorted_salaries) * 0.2)
# trimmed_data = sorted_salaries[trim_count:-trim_count]
# trimmed_mean_total = sum(trimmed_data) / len(trimmed_data)


# trimmed mean (education level - total, male and female):
trimmed_mean_bachelor_male = trim_mean(bachelor_male_salaries, proportiontocut=0.1)
trimmed_mean_bachelor_female = trim_mean(bachelor_female_salaries, proportiontocut=0.1)
trimmed_mean_bachelor_total = trim_mean(bachelor_total_salaries, proportiontocut=0.1)
trimmed_mean_master_male = trim_mean(master_male_salaries, proportiontocut=0.1)
trimmed_mean_master_female = trim_mean(master_female_salaries, proportiontocut=0.1)
trimmed_mean_master_total = trim_mean(master_total_salaries, proportiontocut=0.1)
trimmed_mean_phd_male = trim_mean(phd_male_salaries, proportiontocut=0.1)
trimmed_mean_phd_female = trim_mean(phd_female_salaries, proportiontocut=0.1)
trimmed_mean_phd_total = trim_mean(phd_total_salaries, proportiontocut=0.1)

# print(trimmed_mean_male, trimmed_mean_female, trimmed_mean_total, trimmed_mean_bachelor_male, trimmed_mean_master_male, trimmed_mean_phd_male, trimmed_mean_bachelor_female, trimmed_mean_master_female, trimmed_mean_phd_female, trimmed_mean_bachelor_total, trimmed_mean_master_total, trimmed_mean_phd_total)


# Variance (total, male and female):
variance_total = round(statistics.variance(salaries), 1)

variance_male = round(statistics.variance(male_salaries), 1)

variance_female = round(statistics.variance(female_salaries), 1)
# without using library statistics:
# difference_squared_total = [(i-arith_mean_total)**2 for i in salaries]
# sum_difference_squared_total = sum(difference_squared_total)
# variance_total = sum_difference_squared_total/n_total


# Standard deviation (total, male and female):
standard_deviation_total = round((variance_total**0.5), 2)

standard_deviation_male = round((variance_male**0.5), 2)

standard_deviation_female = round((variance_female**0.5), 2)
# using library statistics:
# standard_deviation_total = statistics.stdev(salaries)


# sample standard deviation (total, male and female):
sample_std_dev_total = statistics.pstdev(salaries)

sample_std_dev_male = statistics.pstdev(male_salaries)

sample_std_dev_female = statistics.pstdev(female_salaries)
# without using library statistics:
# difference_squared_total = [(i-arith_mean_total)**2 for i in salaries]
# sum_difference_squared_total = sum(difference_squared_total)
# variance_sample_std_dev_total = sum_difference_squared_total/(n_total-1)
# sample_std_dev_m = variance_sample_std_dev_total**0.5


# standard error (total, male and female):
standard_error_total = sample_std_dev_total/((n_total-1)**0.5)

standard_error_male = sample_std_dev_male/((n_male-1)**0.5)

standard_error_female = sample_std_dev_female/((n_female-1)**0.5)
# using library scipy:
# from scipy.stats import sem
# standard_error_total = sem(salaries)


# variance (education level):
bachelor_salaries = salary_data[salary_data['Education Level'] == "Bachelor's"]['Salary']
variance_bachelor = statistics.variance(bachelor_salaries)

master_salaries = salary_data[salary_data['Education Level'] == "Master's"]['Salary']
variance_master = statistics.variance(master_salaries)

phd_salaries = salary_data[salary_data['Education Level'] == "PhD"]['Salary']
variance_phd = statistics.variance(phd_salaries)


# standard deviation (education level):
standard_deviation_bachelor = statistics.stdev(bachelor_salaries)
sample_std_dev_bachelor = statistics.pstdev(bachelor_salaries)

standard_deviation_master = statistics.stdev(master_salaries)
sample_std_dev_master = statistics.pstdev(master_salaries)

standard_deviation_phd = statistics.stdev(phd_salaries)
sample_std_dev_phd = statistics.pstdev(phd_salaries)


# standard error (education level):
from scipy.stats import sem
standard_error_bachelor = sem(bachelor_salaries)

standard_error_master = sem(master_salaries)

standard_error_phd = sem(phd_salaries)


# confidence interval (total, male and female):
from scipy.stats import t
confidence_level = 0.95
df_total = n_total - 1
confidence_intervall_total = t.interval(confidence_level, df_total, loc=arith_mean_total, scale=standard_deviation_total/np.sqrt(n_total))

df_male = n_male - 1
confidence_intervall_male = t.interval(confidence_level, df_male, loc=arith_mean_male, scale=standard_deviation_male/np.sqrt(n_male))

df_female = n_female - 1
confidence_intervall_female = t.interval(confidence_level, df_female, loc=arith_mean_female, scale=standard_deviation_female/np.sqrt(n_female))
# without using library scipy:
# confidence_intervall_total = 1.96*standard_error_total
# ci_upper_limit = arith_mean_total + confidence_intervall_total
# ci_lower_limit = arith_mean_total - confidence_intervall_total


# interquartile range (total, male and female):
q1_total = np.percentile(salaries, 25)
q3_total = np.percentile(salaries, 75)
iqr_total = q3_total-q1_total

q1_male = np.percentile(male_salaries, 25)
q3_male = np.percentile(male_salaries, 75)
iqr_male = q3_male-q1_male

q1_female = np.percentile(female_salaries, 25)
q3_female = np.percentile(female_salaries, 75)
iqr_female = q3_female-q1_female


# covariance (years of experience vs. salary):
experience = salary_data['Years of Experience'].values
arith_mean_experience = sum(experience)/len(experience)
covariance_exp_sal = np.cov(experience, salaries)[0, 1]

# using library numpy:
# diff_experience = experience - arith_mean_experience
# diff_salary = salaries - arith_mean_total
# covariance_experience_salary = np.mean(diff_experience * diff_salary)


# correlation (years of experience vs. salary):
standard_deviation_experience = statistics.stdev(experience)
correlation_exp_sal = covariance_exp_sal/(standard_deviation_total*standard_deviation_experience)
# using library scipy:
# from scipy.stats import pearsonr
# correlation_exp_sal = pearsonr(salaries, experience)
print(covariance_exp_sal, correlation_exp_sal)

# probability density function (total):
from scipy.stats import norm
pdf_values = norm.pdf(salaries, loc=arith_mean_total, scale=standard_deviation_total)
