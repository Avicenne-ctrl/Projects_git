import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib
import seaborn as sns
import webcolors
from typing import List, Tuple, Optional



def display_classes_repartition(data, var1: str, var2: str):
    """
    Display a density graph to see the repartition of one label's value
    depending on the value we want to predict.

    Args:
        data (pd.DataFrame): the dataframe we want to visualize
        var1 (str): the label we want to see the distribution
        var2 (str): the label we will predict in the future
    """
    
    # We take the unique labels of var2
    classes = data[var2].unique()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    # We create the figure and the axes
    fig, axes = plt.subplots(nrows=1, ncols=len(classes), figsize=(14, 6), sharey=True)
    
    if len(classes) == 1:
        axes = [axes]
    
    for ax, clas, color in zip(axes, classes, colors):
        # We extract the data depending on the distinct label of var2
        subset = data[data[var2] == clas]
        
        # Calculate the distributions of var1
        counts_total = data[var1].value_counts().sort_index()
        counts_class = subset[var1].value_counts().sort_index()
        
        # Plot the total distribution
        ax.fill_between(counts_total.index, counts_total.values, color='grey', alpha=0.5, label='all people surveyed')
        
        # Plot the class-specific distribution
        ax.fill_between(counts_class.index, counts_class.values, color=color, alpha=0.5, label=f'highlighted group {var2}={clas}')
        
        # Mean value of the class
        mean_value = subset[var1].mean()
        
        # Add the mean to the graph
        ax.axvline(mean_value, color='g', linestyle='--', label=f'Mean: {mean_value:.2f}')
        
        ax.legend()
        
        ax.set_xlabel(var1)
        ax.set_ylabel('Count')
        ax.set_title(f'{var1} Distribution for {var2} {clas}')
    
    plt.tight_layout()
    plt.show()
    
    
def display_density_comparison(data, var1: str, var2: str):
    """The goal of this function is to display a density graph to see the repartition of one label's value
       depending of the value we want to predict.

    Args:
        data (pd.DataFrame): the dataframe we want to visualize
        var1 (str): the label we want to see the distribution
        var2 (_type_): the label we will predict in the futur
    """
    
    
    # We take the unique label of var2
    classes = data[var2].unique()
    
    fig, axes = plt.subplots(nrows=1, ncols=len(classes), figsize=(14, 6), sharey=True)
    
    for ax, clas in zip(axes, classes):
        
        # we extract the data depending on the distinct label of var2
        subset = data[data[var2] == clas]
            
        # We trace the density 
        sns.kdeplot(data=subset[var1], label=f'{clas} specific', ax=ax, fill=True, common_norm=False)
        sns.kdeplot(data=data[var1], label='total', ax=ax, fill=True, common_norm=False)
        ax.legend()
            
        # Mean value for the subset
        mean_value = subset[var1].mean()
            
        # We add the mean to the graph
        ax.axvline(mean_value, color='g', linestyle='--', label=f'Mean {clas}: {mean_value:.2f}')
        
        # Adding mean value text on the plot
        ax.text(mean_value*1.2, ax.get_ylim()[1] * 0.8, f'Age Mean\n{mean_value:.2f}', color='g', ha='center')
        ax.set_title(f'Density of {var1} for {var2} = {clas}')
        ax.set_xlabel(var1)
        ax.set_ylabel('Density')
        
    return ax
        
# Bar graph visualization  : 

def plot_barh_graph(values: List[float], 
                   labels: List[str], 
                   name_value: str, 
                   name_labels: str, 
                   color: str, 
                   ax=Optional[plt.Axes])-> Tuple[plt.Figure, plt.Axes]:

    """ The goal of this function is to display a bar graph, with the option
    Args:
        values (list): list of value we want to display
        labels (list): list of the label's values
        name_value (str): label for the y axis
        name_labels (_type_): label for the x axis
        color (list): color description
        ax (plt): subplot where we want to display the fig

    Returns:
        fig : plt fig
        ax : plt axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 40))

    else:
        fig = ax.figure
        
    fig_width, fig_height = fig.get_size_inches()
        
    # Determine the colors for the bars
    if isinstance(color, str):
        try:
            # Try to convert web color name to hex
            color = webcolors.name_to_hex(color)
            colors = [color for _ in range(len(labels))]
            
        except ValueError:
            
            colormap = plt.get_cmap(color)
            colors = [colormap(i / len(values)) for i in range(len(values))]

    elif isinstance(color, (matplotlib.colors.LinearSegmentedColormap, matplotlib.colors.ListedColormap)):
        colors = [color(i / len(values)) for i in range(len(values))]

    else:
        colors = color

    data = pd.DataFrame({'values': values, 'labels': labels})
    
    # Calculate font size based on figure size
    base_font_size = 10
    font_size = (base_font_size * (fig_height / 10))
    bar_width = 0.8 * (fig_width / 10)
    linewidth = 0.5 * (fig_width / 10)  # Adjust the linewidth
    
    # Sort the data
    data = data.sort_values('values', ascending=False)  # Ascending for horizontal bars
    
    # Plot with seaborn
    sns.barplot(x='values', y='labels', data=data, palette=colors, ax=ax, edgecolor='black', linewidth=linewidth)

    # Add values at the end of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_width(), '.2f'), 
                   (p.get_width(), p.get_y() + p.get_height() / 2.), 
                   ha='left', va='center', fontsize = 12,
                   xytext=(5, 0), 
                   textcoords='offset points')

    # Plot a dashed line for the maximum value
    max_value = max(values)
    ax.axvline(x=max_value, color='red', linestyle='--', label='Max Value')
    
    ax.legend(bbox_to_anchor=(0.95, 0.07), fontsize=font_size)
    
    # Set xlabel and ylabel
    ax.set_xlabel(name_value, fontsize=font_size)
    ax.set_ylabel(name_labels, fontsize=font_size)
    
    # Set the title
    ax.set_title('Bar graph with max value', fontsize=font_size)

    # Adjust the ticks' font size
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    
    return ax

def plot_bar_graph(values: List[float], 
                   labels: List[str], 
                   name_value: str, 
                   name_labels: str, 
                   color: str, 
                   ax=Optional[plt.Axes])-> Tuple[plt.Figure, plt.Axes]:
    """ The goal of this function is to display a vertical bar graph using sns.barplot.
    
    Args:
        values (list): list of values we want to display
        labels (list): list of the label's values
        name_value (str): label for the y axis
        name_labels (str): label for the x axis
        color (str or list): color description, either a seaborn palette name or a list of colors
        ax (plt.Axes, optional): subplot where we want to display the fig. Defaults to None.

    Returns:
        plt.Axes: The axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(40, 20))

    else:
        fig = ax.figure
        
    # Determine the colors for the bars
    if isinstance(color, str):
        try:
            # Try to convert web color name to hex
            color = webcolors.name_to_hex(color)
            colors = [color for _ in range(len(labels))]
            
        except ValueError:
            
            colormap = plt.get_cmap(color)
            colors = [colormap(i / len(values)) for i in range(len(values))]

    elif isinstance(color, (matplotlib.colors.LinearSegmentedColormap, matplotlib.colors.ListedColormap)):
        colors = [color(i / len(values)) for i in range(len(values))]

    else:
        colors = color
        
    # Sort the values and labels in descending order
    sorted_indices = np.argsort(values)[::1]
    values = [values[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    
    # Create a DataFrame for Seaborn
    data = pd.DataFrame({'values': values, 'labels': labels})
    
    # Sort the data
    data = data.sort_values('values', ascending=False)  # Sort in descending order
    
    # Calculate font size based on figure size
    fig_width, fig_height = fig.get_size_inches()
    base_font_size = 10
    font_size = base_font_size * (fig_height / 10)
    bar_width = 0.8 * (fig_width / 10)
    linewidth = 0.5 * (fig_width / 10)  # Adjust the linewidth
    
    # Plot with seaborn
    sns.barplot(x='labels', y='values', data=data, palette=colors, ax=ax, edgecolor='black', linewidth=linewidth)

    # Add values on top of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 9), fontsize = font_size,
                   textcoords='offset points')

    # Plot a dashed line for the maximum value
    max_value = max(values)
    ax.axhline(y=max_value, color='red', linestyle='--', label='Max Value')
    
    ax.legend(bbox_to_anchor=(0.95, 0.07), fontsize=font_size)
    
    # Set xlabel and ylabel
    ax.set_xlabel(name_value, fontsize=font_size)
    ax.set_ylabel(name_labels, fontsize=font_size)

    # Set the title
    ax.set_title('Bar graph with max value', fontsize=font_size)

    # Adjust the ticks' font size
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    
    return ax

def plot_multiple_barh(datas: List[List[float]], 
                        columns: List[str], 
                        labels: List[str], 
                        colors: List[str], 
                        ax: Optional[plt.Axes])-> Tuple[plt.Figure, plt.Axes]:
    
    """
    The goal of this function is to plot a double barh. Data 1, 2 and label need to have the same length.

    Args:
        datas (List[List[float]]) : list of list values
        columns (List[str]) : name of the 2 vars compared
        labels (List[str]) : label list
        colors (List[str]) : list of color for the first list of values
        ax (Optional[plt.Axes]) : figure we want to plot this graph

    Returns:
        ax : the plt figure

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(20,20))

    else:
        fig = ax.figure

    # bar position
    indices = np.arange(len(labels))
    bar_width = 0.5
    bar_width = 0.8 / len(datas)

    for i in range(len(datas)):
        data = datas[i]
        name = columns[i]
        color = colors[i]

        bars = ax.barh(indices + (i - len(datas)/2) * bar_width, data, bar_width, label=name, color=color)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.2f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')

    ax.set_yticks(indices)
    ax.set_yticklabels(labels)
    ax.legend()
    plt.show()
    return fig, ax


def plot_multiple_barv(datas: List[List[float]], 
                        columns: List[str], 
                        labels: List[str], 
                        colors: List[str], 
                        ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    The goal of this function is to plot a double bar. Data 1, 2 and label need to have the same length.

    Args:
        datas (list) : list of list values
        columns (list of str) : name of the vars compared
        labels (list) : label list
        colors (list of str) : list of color for the list of values
        ax (plt.Axes) : axes we want to plot this graph

    Returns:
        ax : the plt axes
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 20))
    else:
        fig = ax.figure

    # Bar position
    indices = np.arange(len(labels))
    bar_width = 0.8 / len(datas)

    for i in range(len(datas)):
        data = datas[i]
        name = columns[i]
        color = colors[i]

        bars = ax.bar(indices + (i - len(datas) / 2) * bar_width, data, bar_width, label=name, color=color)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()
    return fig, ax

