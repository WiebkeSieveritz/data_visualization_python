import main
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, ScalarFormatter


# creation of new DataFrame of all means:
data = {
    'Gender': ['Male', 'Female', 'Total'],
    'Arithmetic mean': [main.arith_mean_male, main.arith_mean_female, main.arith_mean_total],
    'Median': [main.median_male, main.median_female, main.median_total],
    'Trimmed mean': [main.trimmed_mean_male, main.trimmed_mean_female, main.trimmed_mean_total]
}
df_means = pd.DataFrame(data)
df_melted = pd.melt(df_means, id_vars='Gender', var_name='Means', value_name='Value')

arith_mean_male_el = [main.arith_mean_bachelor_male, main.arith_mean_master_male, main.arith_mean_phd_male]
arith_mean_female_el = [main.arith_mean_bachelor_female, main.arith_mean_master_female, main.arith_mean_phd_female]
arith_mean_total_el = [main.arith_mean_bachelor_total, main.arith_mean_master_total, main.arith_mean_phd_total]

median_male_el = [main.median_bachelor_male, main.median_master_male, main.median_phd_male]
median_female_el = [main.median_bachelor_female, main.median_master_female, main.median_phd_female]
median_total_el = [main.median_bachelor_total, main.median_master_total, main.median_phd_total]

trimmed_mean_male_el = [main.trimmed_mean_bachelor_male, main.trimmed_mean_master_male, main.trimmed_mean_phd_male]
trimmed_mean_female_el = [main.trimmed_mean_bachelor_female, main.trimmed_mean_master_female,
                          main.trimmed_mean_phd_female]
trimmed_mean_total_el = [main.trimmed_mean_bachelor_total, main.trimmed_mean_master_total, main.trimmed_mean_phd_total]

avg_less = round((1-(main.arith_mean_female/main.arith_mean_male))*100, 2)

# lists for labeling
education_level = ["Bachelor's", "Master's", "PhD"]
education_level_s = ["PhD", "Master's", "Bachelor's"]
labels_legend = ['Male', 'Female', 'Total']
means = ['Arithmetic mean', 'Median', 'Trimmed mean']


# plot creation of means

# color assignment
colors = sns.color_palette('muted')
colors[1] = 'lightcoral'
colors_adjusted = sns.color_palette('muted')
colors_adjusted[0] = 'lightgrey'
colors_adjusted[1] = 'lightcoral'
colors_adjusted[2] = 'grey'

# subplots
fig1 = plt.figure(figsize=(10, 6))

ax1 = fig1.add_subplot(2, 1, 1)
barplot = sns.barplot(x='Means', y='Value', hue='Gender', data=df_melted, palette=colors_adjusted, ax=ax1)

ax2 = fig1.add_subplot(2, 3, 4)
ax2.bar(education_level, arith_mean_male_el, color=colors_adjusted[0], label="Bachelor's")
ax2.bar(education_level, arith_mean_female_el, bottom=arith_mean_male_el, color=colors_adjusted[1], label="Master's")
ax2.bar(education_level, arith_mean_total_el, bottom=[sum(x) for x in zip(arith_mean_male_el, arith_mean_female_el)],
        color=colors_adjusted[2], label="PhD")

ax3 = fig1.add_subplot(2, 3, 5, sharey=ax2)
ax3.bar(education_level, median_male_el, color=colors_adjusted[0], label="Bachelor's")
ax3.bar(education_level, median_female_el, bottom=median_male_el, color=colors_adjusted[1], label="Master's")
ax3.bar(education_level, median_total_el, bottom=[sum(x) for x in zip(median_male_el, median_female_el)],
        color=colors_adjusted[2], label="PhD")

ax4 = fig1.add_subplot(2, 3, 6, sharey=ax2)
ax4.bar(education_level, trimmed_mean_male_el, color=colors_adjusted[0], label="Bachelor's")
ax4.bar(education_level, trimmed_mean_female_el, bottom=trimmed_mean_male_el, color=colors_adjusted[1],
        label="Master's")
ax4.bar(education_level, trimmed_mean_total_el, bottom=[sum(x) for x in zip(trimmed_mean_male_el,
                                                                            trimmed_mean_female_el)],
        color=colors_adjusted[2], label="PhD")

# function for a thousand separation
def thousand_separator(x, pos):
    return '{:,.0f}'.format(x)

# setting of titles and labels
ax1.set_title('Different salary means by gender', fontsize=14, y=1.2, fontweight='bold')
plt.suptitle(f'Women earn {avg_less}% less', fontsize=12, y=0.94)

ax1.set_xlabel('')
ax1.set_xticklabels(labels=means, fontsize=9, fontweight='bold')
ax1.set_ylabel('Salary ($)', fontsize=9, fontweight='bold')
ax2.set_xlabel('Arithmetic mean', fontsize=9, fontweight='bold')
ax2.set_ylabel('Salary ($)', fontsize=9, fontweight='bold')
ax3.set_xlabel('Median', fontsize=9, fontweight='bold')
ax4.set_xlabel('Trimmed mean', fontsize=9, fontweight='bold')

# formatting of axes
formatter = FuncFormatter(thousand_separator)
ax1.yaxis.set_major_formatter(formatter)
ax2.yaxis.set_major_formatter(formatter)
ax3.yaxis.set_visible(False)
ax4.yaxis.set_visible(False)
for ax in [ax2, ax3, ax4]:
    ax.tick_params(axis='both', which='major', labelsize=8)
for ax in [ax1]:
    ax.tick_params(axis='y', which='major', labelsize=8)

# implementation of legends
ax1.legend(title='', labels=['Total', 'Male', 'Female'])
handles, labels = barplot.get_legend_handles_labels()  # method to create legends manually, handles = list of objects,
# e.g. bars, labels = list of labels for bars
ax1.legend(handles[0:], labels[0:], title='', loc='upper right', fontsize='small')
ax4.legend(labels=labels_legend, loc='upper right', fontsize='small')

# formatting of the background
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor('#F5F5F5')
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
for ax in [ax3, ax4]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

plt.show()


# plot creation of probability density function

# color assignment
colors = sns.color_palette('muted')
yellow_muted = sns.desaturate('#FFFF00', 0.7)
orange_muted = sns.desaturate('#FFA500', 0.7)
turquoise_muted = sns.desaturate('#00CED1', 0.7)
degree_colors = [yellow_muted, orange_muted, turquoise_muted]
palette_g = {'Male': sns.color_palette("muted")[0], 'Female': 'lightcoral'}
male_color = colors[0]
female_color_grey = 'lightgrey'
total_color = 'grey'

# subplots
fig2, (ax1, ax2) = plt. subplots(1, 2, figsize=(10, 6))

sns.violinplot(data=main.salary_data, x='Gender', y='Salary', ax=ax1, palette={'Male': male_color,
                                                                               'Female': female_color_grey})
sns.violinplot(data=main.salary_data, x=(len(main.salary_data['Salary'])), y='Salary', ax=ax1, color=total_color)

sns.histplot(data=main.salary_data, x='Salary', hue='Education Level', multiple='stack', ax=ax2, palette=degree_colors)

# setting of titles and labels
plt.suptitle('Distribution of salaries', fontsize=14, fontweight='bold')
ax1.set_title('Men have a higher salary range', fontsize=12)
ax2.set_title('A PhD degree results in a higher salary', fontsize=12)

ax1.set_xticklabels([''], color='white')
ax1.set_xlabel('Gender', fontsize=9, fontweight='bold')
ax1.set_ylabel('Salary ($)', fontsize=9, fontweight='bold')
ax2.set_xlabel('Salary ($)', fontsize=9, fontweight='bold')
ax2.set_ylabel('Amount', fontsize=9, fontweight='bold')

# formatting of axes
for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=8)

# implementation of legend
handles_1 = [
    mpatches.Patch(color=male_color, label='Male'),
    mpatches.Patch(color=female_color_grey, label='Female'),
    mpatches.Patch(color=total_color, label='Total')
]
ax1.legend(handles=handles_1, labels=labels_legend, loc='upper center', fontsize='small', title=None)
ax2.legend(labels=education_level_s, loc='upper right', fontsize='small', title=None)

# adding annotations
ax1.text(0.25, -0.10, f"Share of male: {main.share_male}%  -  Share of female: {main.share_female}%",
         transform=ax1.transAxes, fontsize=8)
ax2.text(0.08, -0.10, f"Share of Bachelor's: {main.share_bachelor}%  -  Share of Master's: {main.share_master}%  "
                      f"-  Share of PhD: {main.share_phd}%", transform=ax2.transAxes, fontsize=8)

# formatting of the background
for ax in [ax1, ax2]:
    ax.set_facecolor('#F5F5F5')
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.show()


# plot creation confidence intervall and interquartile range incl. distribution

# color assignment
female_color = 'lightcoral'

# subplots
fig3, (ax2, ax1) = plt. subplots(1, 2, figsize=(10, 6))

sns.boxplot(data=main.salary_data, x='Gender', y='Salary', ax=ax2, palette={'Male': male_color, 'Female': female_color})
sns.swarmplot(data=main.salary_data, x='Gender', y='Salary', ax=ax2, color='black', alpha=0.5)
sns.histplot(main.salaries, kde=True, color=colors[2], ax=ax1)
ax1.axvline(main.confidence_intervall_total[0], linestyle='--', color='darkred', label='Lower CI - 5%')
ax1.axvline(main.confidence_intervall_total[1], linestyle='--', color='darkred', label='Upper CI - 95%')
ax1.axvline(main.q1_total, linestyle='-.', color='black', label='Lower IQR - 25%')
ax1.axvline(main.q3_total, linestyle='-.', color='black', label='Upper IQR - 75%')

# setting of titles and labels
plt.suptitle('Distribution of salaries', fontsize=14, fontweight='bold')
ax1.set_title(f'Salaries have a wide range,\nbut the mean is more likely {main.arith_mean_total}$', fontsize=12)
ax2.set_title('Men have larger outliers than women', fontsize=12,)

ax1.set_xlabel('Salary ($)', fontsize=9, fontweight='bold')
ax1.set_ylabel('Amount', fontsize=9, fontweight='bold')
ax2.set_xlabel('Gender', fontsize=9, fontweight='bold')
ax2.set_ylabel('Salary ($)', fontsize=9, fontweight='bold')

# formatting of axes
ax1.xaxis.set_major_formatter(formatter)
ax2.yaxis.set_major_formatter(formatter)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=8)

# implementation of a legend
ax1.legend(fontsize='small')

# formatting of the background
for ax in [ax1, ax2]:
    ax.set_facecolor('#F5F5F5')
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.show()

# plot creation overlapping density plot

# subplots
fig4, (ax1, ax2) = plt. subplots(1, 2, figsize=(10, 6))

sns.kdeplot(data=main.salary_data, x='Salary', hue='Gender', fill=True, ax=ax1, palette=palette_g)
sns.kdeplot(data=main.salary_data, x='Salary', hue='Education Level', fill=True, ax=ax2, palette=degree_colors)

# setting of titles and labels
plt.suptitle('Overlapping density plot of salary', fontsize=14, fontweight='bold')
ax1.set_title('Salary for men and women: Mostly in agreement', fontsize=12)
ax2.set_title('Income largely depends on the degree', fontsize=12)

ax1.set_xlabel('Salary ($)', fontsize=9, fontweight='bold')
ax1.set_ylabel('Density', fontsize=9, fontweight='bold')
ax2.set_xlabel('Salary ($)', fontsize=9, fontweight='bold')
ax2.set_ylabel('Density', fontsize=9, fontweight='bold')

# formatting of axes
for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
for ax in [ax1, ax2]:
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.yaxis.get_major_formatter().set_scientific(False)

# implementation of legends
ax1.legend(labels=labels_legend, loc='upper right', fontsize='small', title=None)
ax2.legend(labels=education_level_s, loc='upper right', fontsize='small', title=None)

# formatting of the background
for ax in [ax1, ax2]:
    ax.set_facecolor('#F5F5F5')
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=8)
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.show()

# plot creation of covariance and correlation
from matplotlib.patches import Ellipse

# creation of covariance matrix to illustrate the variance of both variables
cov_matrix = np.array([[np.var(main.experience), main.covariance_exp_sal],
                       [main.covariance_exp_sal, np.var(main.salaries)]])

# calculation of Eigenvalues and eigenvectors of covariance matrix to determine the length and direction of the ellipse
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))  # orientation of the ellipse

# creation of the covariance ellipse
ellipse = Ellipse(xy=(main.arith_mean_experience, main.arith_mean_total),
                  width=np.sqrt(5.991 * eigenvalues[0]),
                  height=np.sqrt(5.991 * eigenvalues[1]),
                  angle=angle,
                  edgecolor='r',
                  fc='None')

# subplots
fig5, (ax1, ax2) = plt. subplots(1, 2, figsize=(10, 6))

ax1.add_patch(ellipse)
ax1.scatter(main.experience, main.salaries, alpha=0.5, c=[palette_g[gender] for gender in main.salary_data['Gender']])
ax2.scatter(main.experience, main.salaries, alpha=0.5, c=[palette_g[gender] for gender in main.salary_data['Gender']])
slope, intercept = np.polyfit(main.experience, main.salaries, 1)
plt.plot(main.experience, slope * np.array(main.experience) + intercept, color='red', label='Linear Fit')
# creation of regression line including slope of every element for each intercept

# setting of titles and labels
plt.suptitle('Connection between years of experience and salary', fontsize=14, fontweight='bold')
ax1.set_title('High covariance', fontsize=12)
ax2.set_title('High correlation', fontsize=12)

ax1.set_xlabel('Years of Experience', fontsize=9, fontweight='bold')
ax2.set_xlabel('Years of Experience', fontsize=9, fontweight='bold')
ax1.set_ylabel('Salary ($)', fontsize=9, fontweight='bold')
ax2.set_ylabel('Salary ($)', fontsize=9, fontweight='bold')

# formatting of axes
ax1.yaxis.set_major_formatter(formatter)
ax2.yaxis.set_major_formatter(formatter)
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=8)

# formatting of the background
ax1.grid(True)
ax2.grid(True)

for ax in [ax1, ax2]:
    ax.set_facecolor('#F5F5F5')
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# implementation of a legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label)
                   for label, color in palette_g.items()]
ax2.legend(handles=legend_elements, loc='upper right', fontsize='small')

# adding annotations
ax1.text(0.08, -0.10, f'Variance male: {main.variance_male}  -  female: {main.variance_female}  -  total: '
                      f'{main.variance_total}', transform=ax1.transAxes, fontsize=8)
ax1.text(0.1, -0.13, f'Standard deviation male: {main.standard_deviation_male}  -  female: '
                     f'{main.standard_deviation_female}  -  total: {main.standard_deviation_total}',
         transform=ax1.transAxes, fontsize=8)

plt.show()
