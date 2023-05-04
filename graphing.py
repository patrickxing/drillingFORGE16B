# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:48:14 2021

@author: zphelan
"""
import numpy as np
import scipy
import seaborn
from pandas import Series
from statsmodels.graphics.gofplots import qqplot


# Create boxplots of each variable in df
def visualize_data(df, title=None, kind='box', layout=(4, 4), folder=""):
    plt.figure(figsize=(24, 12))

    if kind == 'hist':
        df.hist()

    else:
        df.plot(kind=kind, subplots=True, layout=layout,
                sharex=False, sharey=False)

    if title is not None:
        plt.suptitle(title)

    plt.savefig('./figures/' + folder + '/box plots/boxes')


def graph_changes_histogram(df_prefilter, df, folder):
    for var in df.columns:
        if var not in df_prefilter.columns:
            continue

        fig = compare_data(df_prefilter, df, var=var, firstlabel='Pre-Filter', secondlabel="Base filter", cmp=False)

        var = var.replace('/', '_')
        temp = var.split()
        name = '_'.join(temp)
        plt.savefig('./figures/' + folder + '/new/compare_' + name)
        plt.close(fig)


# Function to compare two different data frames. Here, it is used to see the before and after distributions of a filter.
def compare_data(df, df_filtered, figsize=(12, 12), var='Flow In', firstlabel='Before dropping', secondlabel='After dropping', cmp=True):
    # Set figure size for distribution
    fig = plt.figure(figsize=figsize)

    if cmp:
        ax = fig.add_subplot(131)
        df[var].hist(bins=100, ax=ax, color='orange', alpha=1, label=firstlabel)
        plt.title(firstlabel.split()[0])
        plt.legend()

        ax = fig.add_subplot(132)
        df_filtered[var].hist(bins=100, ax=ax, color='green', alpha=0.8, label=secondlabel)

        plt.title(secondlabel.split()[0])
        plt.legend()

        ax = fig.add_subplot(133)
        df[var].hist(bins=100, ax=ax, color='orange', alpha=1, label=firstlabel)
        df_filtered[var].hist(bins=100, ax=ax, color='green',
                              alpha=0.5, label=secondlabel)
        plt.title("Distribution change")
        plt.legend()

    else:
        start = min(df_filtered[var]) * 1.1
        end = max(df_filtered[var]) * 1.1

        ax = fig.add_subplot(111)
        df[var].hist(bins=100, ax=ax, color='orange', alpha=1, label=firstlabel, range=(start, end))
        df_filtered[var].hist(bins=100, ax=ax, color='green', alpha=0.5, label=secondlabel)
        plt.title("Distribution change")
        plt.legend()

    plt.suptitle(str(var), fontsize=24)
    fig.tight_layout()
    seaborn.set_style("whitegrid")

    return fig

# graph_corr -- Graph the correlation between all variables in the dataframe 'df'. Set 'title' as the suptitle in graph.
# dependencies: seaborn, matplotlib.pyplot
def graphCorrelation(df, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    correlation = df.corr(method='spearman')
    correlation.style.background_gradient(cmap='jet')

    seaborn.heatmap(correlation, annot=True, cmap='jet')

    # Rotate x axis for legibility
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
    plt.suptitle(title, fontsize=24)

    plt.savefig(title)
    # plt.show() # Uncomment if want to show plot. For me, I save it after the method is complete.

# Graph differences in the KDE plots of df1 and df2.
# Note: if the dataframes are too large, it will try to allocate too much memory.
def graph_differences(df1, df2, title):
    kdea = scipy.stats.gaussian_kde(df1)
    kdeb = scipy.stats.gaussian_kde(df2)

    grid = np.linspace(0, 150, 151)

    plt.plot(grid, kdea(grid), label="kde A")
    plt.plot(grid, kdeb(grid), label="kde B")
    plt.plot(grid, kdea(grid) - kdeb(grid), label="difference")

    plt.legend()
    plt.show()


# Graph rolling average of Calculated ROP and depth.
def graph_avg(df, folder, save=False, var=['ROP', 'Depth'], windows=[2, 6, 60, 600]):
    types = ['mean', 'median']

    for num in windows:
        for variant in types:
            for t in var:
                if t == 'ROP':
                    col = 1  # ROP Calculated is column 2
                
                elif t == 'Depth':
                    col = 0  # Total Depth is column 0

                # Use rolling average of calculated ROP to smooth results
                if types == 'mean':
                    avg = df.iloc[:, col].rolling(window=num).mean()
                     
                else:
                    avg = df.iloc[:, col].rolling(window=num).median()

                fig, ax = plt.subplots(figsize=(16, 8))
            
                if t == 'ROP':
                    x = df['Total Depth']
                    y = avg
                    # m, b = np.polyfit(x, y, 1)
                    # plt.plot(x,  m * x + b, c='orange')

                elif t == 'Depth':
                    continue
                    x = df['Time']
                    y = df['Total Depth']

                ax.scatter(x, y)
                if t == 'ROP':
                    ax.set_xlabel('Total Depth')
                    ax.set_ylabel('Calculated ' + str(t) + ' (rolling ' + str(variant) + ')')
                
                elif t == 'Depth':
                    ax.set_xlabel('Time')
                    ax.set_ylabel(str(t) + ' (rolling ' + str(variant) + ')')
                    plt.tick_params(
                        axis='x',
                        bottom=False,
                        labelbottom=False
                        )
                    
                title = str(t) + ' (rows = ' + str(num) + ')'
                plt.suptitle(title)

                if save:
                    plt.savefig('./figures/' + folder + '/average_' + str(t).lower() + '/' + str(variant) + '_rows=' + str(num))

                else:
                    plt.show()

                plt.close(fig)


# Graph QQ plots for each variable to test for normal distribution
def graph_qq(df, folder, line='s', save=False, ):
    df = df.dropna()
    for var in df.columns:
        if var == 'Time':
            continue
        qqplot(df[var], line=line)
        plt.suptitle(str(var))

        if save:
            var = var.replace('/', '_')
            temp = var.split()
            name = '_'.join(temp)
            plt.savefig('./figures/' + folder + '/normal/qq_' + str(name))

        else:
            plt.show()


# Plots the loss over epochs from the ANN learning phase
#   If a validation set is provided to the model, can plot that as well
def graph_loss(model, folder):
    plt.figure()
    plt.plot(model.history.history['loss'])
    # plt.plot(trained_model.history.history['val_loss'])
    plt.legend(['Train'])
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.title('Training log')
    plt.suptitle(folder, fontsize=24)
    plt.savefig('./figures/' + folder + '/ANN/loss.png')
    plt.close()

# graph_coeff -- Graph the feature importance coefficients for the selected features. This assumes that the
#                   correlation and features arrays / series are in the same order.
# Variables:
#   - coeffs        : coefficients of ML method, in an array or array-like object.
#   - features      : Feature / column names corresponding to coefficients, in an array or array-like object.
#   - folder / type : Used for folder structure, can be changed or discarded if not needed.
#   - relative      : Boolean, set to true to normalize 'coeffs' data between 0.0 and 1.0. Otherwise, use given values.
# Dependencies:
#   - matplotlib
#   - pandas
def graph_coeff(coeffs, features, folder, type, relative=False):
    coeffs = abs(coeffs)
    if relative:
        coeffs = (coeffs - min(coeffs)) / (max(coeffs) - min(coeffs))

    coeff_df = Series(data=coeffs, index=features)
    coeff_df = coeff_df.sort_values()
    coeff_df.plot.bar()
    plt.tight_layout()
    #plt.savefig('./figures/' + folder + '/coeff/' + type + '.png')
    plt.show()
    plt.close()


# Graph a seaborn pairplot, with colors corresponding to the depth of the data row.
def pairplot(df_new):
    conditions = [
        (df_new["Total Depth"] >= 0) & (df_new["Total Depth"] < 5892),
        (df_new["Total Depth"] >= 5892) & (df_new["Total Depth"] < 7262),
        (df_new["Total Depth"] >= 7262)
    ]
    values = ["Initial", "Tilting", "Final"]
    categorized = df_new.copy()
    categorized["Depth Group"] = np.select(conditions, values)
    seaborn.pairplot(categorized, hue="Depth Group", corner=True)#, markers="$\circ$")
    plt.show()


# Graph multiple plots to compare data change for each variable, against y (usually depth)
def graph_originals_vs_new(df_old, y_old, df_new, y_new):
    df_old.drop('ROP Depth/Hour', inplace=True, axis=1)

    fig, ax = plt.subplots(2, len(df_new.columns), figsize=(16,12))
    for i, col in enumerate(df_old.columns):
        axis1 = seaborn.regplot(x=df_old[col], y=y_old, ax=ax[0, i])
        axis1.invert_yaxis()

        axis2 = seaborn.regplot(x=df_new[col], y=y_new, ax=ax[1, i])
        axis2.sharex(axis1)
        axis2.sharey(axis1)

    fig.tight_layout()

    plt.show()


def graph_results(train_output, test_output, train_y, test_y):
    df = pd.DataFrame()
    temp_output = train_output.copy()
    temp_output.extend(test_output)
    df['Predicted'] = temp_output

    categories = ['Training Data' for x in train_output]
    temp = ['Testing Data' for x in test_output]
    categories.extend(temp)
    df['Category'] = categories

    temp_y = train_y.copy()
    temp_y.extend(test_y)
    df['Actual'] = temp_y
    seaborn.set_context("paper", font_scale=2.0)
    seaborn.jointplot(data=df, x="Predicted", y="Actual", hue="Category")
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 00:01:50 2021

@author: ppanja
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sb
import statistics
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.split(script_path)[0]


# %% Heat map, written by Palash Panja
def scatter_hist_heatmap(x, y, data_name, unit):  # , plot_type,plot_number):
    n_data = x.shape[0]  # total number of wells

    g1 = sb.jointplot(x, y, kind="kde", color='b', shade=True, cmap=cm.jet, shade_lowest=True, fill=True, )

    # title of scatter plot
    plt.title(str(data_name) + "Count : " + str(n_data), x=-3.0, y=0.95, color='White', fontstyle='normal',
              fontweight='bold', fontsize=10, fontname='Times New Roman')

    # % transfor the position of text depending on axis value
    min_x, max_x = plt.xlim()  # return the current xlim
    min_y, max_y = plt.ylim()  # return the current xlim
    del_x = max_x - min_x
    del_y = max_y - min_y
    ################# Top histogram ########################
    plt.text(min_x - del_x * 1.5, max_y + del_y * 0.08,
             "Mean      : " + str(round(statistics.mean(x))) + "\n" "Median   : " + str(
                 round(statistics.median(x))) + "\n" "Std. Dev. : " + str(round(statistics.stdev(x))),
             fontstyle='normal', fontweight='bold', fontsize=10, fontname='Times New Roman')

    ################ right histogram ########################
    plt.text(min_x + del_x * 0.2, min_y + del_y * 0.5,
             "Mean      : " + str(round(statistics.mean(y))) + "\n" "Median   : " + str(
                 round(statistics.median(y))) + "\n" "Std. Dev. : " + str(round(statistics.stdev(y))),
             fontstyle='normal', fontweight='bold', fontsize=10, fontname='Times New Roman')

    ################# Axes names ########################
    g1.set_axis_labels('Actual ' + str(data_name) + ' ( ' + str(unit) + ' )',
                       'Predicted ' + str(data_name) + ' ( ' + str(unit) + ' )', color='Black', fontsize=12,
                       fontname='Times New Roman')
    # g1.fig.axes[0].invert_yaxis()

    ################# Bottom Colorbar ########################
    norm = plt.Normalize(0, n_data)
    sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=norm)  # other cmap "Spectral",  "RdYlGn"
    sm.set_array([])
    cbar_ax = g1.fig.add_axes([0.15, -0.05, 0.65, 0.03])  # [left, bottom, width, height] # for vertical bar
    g1.fig.colorbar(sm, label="Data count", cax=cbar_ax, orientation="horizontal")#, color='Black', fontsize=12,
                    #fontname='Times New Roman')

    ################# save the figure ########################
    file_name = str(data_name) + '.png'
    #plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


# %% main program
# df = pd.read_excel(os.path.join(script_dir, ".\Data_Formations_v2.xlsx"), 1)
# x = df[df.columns[1]]
# y = df[df.columns[2]]
# scatter_hist_heatmap(x, y, "Weight On Bit", "kg")