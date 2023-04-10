"""
Author: Song Gao

Description:
This python script will generate output eda.html. eda.html contains 
the report of correlation heatmaps, scatter plots, histogram, boxplots, and feature selections.

Usage: 
python eda.py; python eda.py ./path/filename; ./eda.py; ./eda.py ./path/filename
With out filename will open an file chooser.
If path doesn't exist will open an file chooser.

Dependency:
sklearn, tkinter, scipy, numpy, pands, bokeh.

Time Complexity:
At least O(n^2). Will Grow quadratically as the input size grows. 

Inuput: 
an .xlsx file or .csv file, with columns of features and row of observations. 
Must have 'host' column and 'uuid' column as features.
'host' column contains strings of benchmark name.
'uuid' column contains strings of platform name.

Output: 
eda.html, which is html report with plots and selected features.
"""

import sys, os, importlib, warnings, time
import pandas as pd
import numpy as np
from scipy import stats
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, chi2, f_classif
from bokeh.io import show
from bokeh.palettes import Viridis256, inferno
from bokeh.plotting import figure, output_file, show
from bokeh.models import CustomJS, ColumnDataSource, BasicTicker, ColorBar, LinearColorMapper, Whisker, TabPanel, Tabs
from bokeh.transform import cumsum, factor_cmap, transform
from bokeh.models.widgets import Select, DataTable, DateFormatter, TableColumn, Div, Paragraph, PreText
from bokeh.layouts import column, row



def removeCorrelatedFeatures(dataframe):
    sel = VarianceThreshold(threshold=0.01)
    sel.fit_transform(dataframe)
    quasi_constant = [col for col in dataframe.columns if col not in sel.get_feature_names_out()]
    train = dataframe[sel.get_feature_names_out()]
    corr_matrix = train.corr()
    corr_features = [feature for feature in corr_matrix.columns if (corr_matrix[feature].iloc[:corr_matrix.columns.get_loc(feature)] > 0.8).any()]
    dataframe = dataframe.drop(quasi_constant + corr_features, axis=1)
    features = dataframe.columns
    nparray = MinMaxScaler().fit_transform(dataframe)
    return nparray, features
    

    
def selectFeaturesChart(nparray, y, features, selectedFeature, function_name, size = 400):
    module = importlib.import_module('sklearn.feature_selection')
    func = getattr(module, function_name)
    # k features with highest f-score statistics are selected
    func_features = SelectKBest(func, k = len(features) // 2 + 2)
    X_kbest_features = func_features.fit(nparray, y)
    selectedFeature += X_kbest_features.get_feature_names_out(features).tolist()
    
    importances = np.array(func(nparray, y))
    
    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help, hover"
    p = figure(x_range=list(features), 
               width=size, 
               height=size+200, 
               title=function_name,
               toolbar_location=None, 
               tools=TOOLS)

    p.vbar(x=features, top=importances.flatten()[0:len(features)], width=0.9)
    p.xaxis.major_label_orientation = "vertical"
    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    return p



def interactive_correlation_heatmap(dataframes, benchmarks, features, output_file_name = "", size=1000):
    data = []
    titles = benchmarks
    corrmat = []
    
    for i in range(len(dataframes)):
        df = dataframes[i]
        corrmat = df.corr()
        corrmat.index.name = 'AllColumns1'
        corrmat.columns.name = 'AllColumns2'
        corrmat = corrmat.stack().rename("value").reset_index()
        data.append(corrmat)

    sources = ColumnDataSource(data[0])

    # I am using 'Viridis256' to map colors with value
    mapper = LinearColorMapper(
        palette=Viridis256, low=-1, high=1)

    # Define a figure and tools
    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help, hover"
    p = figure(
        tools=TOOLS,
        tooltips="@value",
        width=size,
        height=size - 200,
        title="Correlation plot",
        x_range=list(corrmat.AllColumns1.drop_duplicates()),
        y_range=list(corrmat.AllColumns2.drop_duplicates()),
        toolbar_location="right",
        x_axis_location="below")

    # Create rectangle for heatmap
    p.rect(
        x="AllColumns1",
        y="AllColumns2",
        width=1,
        height=1,
        source=sources,
        line_color=None,
        fill_color=transform('value', mapper))

    # Add legend
    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=10))

    p.xaxis.major_label_orientation = "vertical"
    p.add_layout(color_bar, 'right')

    data = [i.to_dict('list') for i in data]
    data = dict(zip(titles, data))

    select = Select(title="Benchmarks: ", value=titles[0], options=titles)
    select.js_on_change("value", CustomJS(args=dict(sources=sources, data=data), code="""
        sources.data = data[cb_obj.value];
    """))

    if output_file_name != "":
        show(column(p, select))
        output_file(output_file_name + ".html")
    
    return TabPanel(child=column(p, select), title="Correlations")



def interactive_scatterplot(dataframes, benchmarks, features, output_file_name = "", size=1000):
    titles = benchmarks
    data = []
    
    for i in range(len(dataframes)):
        df_zscore = dataframes[i].apply(stats.zscore)        #same as df_zscore = (df - df.mean())/df.std()
        sample = df_zscore.sample(1000)
        feature_data = []
        for j in range(len(features)):
            feature_data.append(sample[features[j]])
        data.append(dict(zip(features, feature_data)))
    data = dict(zip(titles, data))

    sources = ColumnDataSource(data=dict(x = data[titles[0]][features[0]], y = data[titles[0]][features[0]]))

    p = figure(width=size, 
               height=size-200, 
               title = "Standard Deviation Plot")
    p.scatter('x', 'y', source=sources)

    select_benchmark = Select(title="Benchmarks: ", value=titles[0], options=titles)
    select_feature_x = Select(title="Featrues x: ", value=features[0], options=features) 
    select_feature_y = Select(title="Featrues y: ", value=features[0], options=features) 
    
    select_benchmark.js_on_change("value", CustomJS(args=dict(select_feature_x=select_feature_x, select_feature_y=select_feature_y, sources=sources, data=data), code="""
        const obj = { x: data[cb_obj.value][select_feature_x.value], y: data[cb_obj.value][select_feature_y.value] };
        sources.data = obj;
    """))
    
    select_feature_x.js_on_change("value", CustomJS(args=dict(select_benchmark=select_benchmark, select_feature_y=select_feature_y, sources=sources, data=data), code="""
        const obj = { x: data[select_benchmark.value][cb_obj.value], y: data[select_benchmark.value][select_feature_y.value] };
        sources.data = obj;
    """))
    
    select_feature_y.js_on_change("value", CustomJS(args=dict(select_benchmark=select_benchmark, select_feature_x=select_feature_x, sources=sources, data=data), code="""
        const obj = { x: data[select_benchmark.value][select_feature_x.value], y: data[select_benchmark.value][cb_obj.value] };
        sources.data = obj;
    """))

    if output_file_name != "":
        show(column(p, select_benchmark, select_feature))
        output_file(output_file_name + ".html")
    return TabPanel(child=column(p, select_benchmark, row(select_feature_x, select_feature_y)), title="Scatters")



def interactive_histgram(dataframes, benchmarks, features, output_file_name = "", size=1000):
    titles = benchmarks
    features = features
    data = []

    for i in range(len(dataframes)):
        data_features = []
        df = dataframes[i]
        for j in features:
            # Histogram
            x = df[j]
            bins = np.linspace(x.min(), x.max(), 40)
            hist, edges = np.histogram(x, density=True, bins=bins)
            data_features.append(dict(top=hist, bottom=[0]* len(hist), left=edges[:-1], right=edges[1:]))
        data.append(dict(zip(features, data_features)))

    data = dict(zip(titles, data))
    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help, hover"
    p = figure(width=size, 
               height=size-200, 
               tools=TOOLS,
               title="Histogram")

    x = data[titles[0]][features[0]]
    x = ColumnDataSource(x)
    p.quad(source=x,
           fill_color="skyblue", 
           line_color="white",
           legend_label="samples")
    p.y_range.start = 0
    p.xaxis.axis_label = "x"
    p.yaxis.axis_label = "Counts(x)"

    select_benchmark = Select(title="Benchmarks: ", value=titles[0], options=titles)
    select_feature = Select(title="Featrues : ", value=features[0], options=features) 
    select_benchmark.js_on_change("value", CustomJS(args=dict(select_feature=select_feature, x=x, data=data), code="""
        x.data = data[cb_obj.value][select_feature.value];
    """))
    select_feature.js_on_change("value", CustomJS(args=dict(select_benchmark=select_benchmark, x=x, data=data), code="""
        x.data = data[select_benchmark.value][cb_obj.value];
    """))
    
    if output_file_name != "":
        show(column(p, select_benchmark, select_feature))
        output_file(output_file_name + ".html")
    return TabPanel(child=column(p, select_benchmark, select_feature), title="Histgrams")



def interactive_boxplot(dataframes, benchmarks, features, output_file_name = "", size=1000, log_transformation = False):
    titles = benchmarks
    data = []
    data_outliers = []
    df = pd.DataFrame()
    features = features
    for i in range(len(dataframes)):
        if log_transformation:
            df = np.log10(dataframes[i])
        else:
            df = dataframes[i]

        # compute quantiles
        qs = df.quantile([0.25, 0.5, 0.75], axis=0).T
        qs.columns = ["q1", "q2", "q3"]
        
        # compute IQR outlier bounds
        iqr = qs.q3 - qs.q1
        qs["upper"] = qs.q3 + 1.5*iqr
        qs["lower"] = qs.q1 - 1.5*iqr

        data_out_iqr = df[df.lt(qs["lower"].T, axis=1) | df.ge(qs["upper"].T, axis=1)]
        data_outliers_in_feature = pd.DataFrame(columns=['value', 'feature'])
        df_temp = pd.DataFrame()

        for j in features:
            df_temp['value'] = data_out_iqr[j]
            df_temp['feature'] = j
            data_outliers_in_feature = pd.concat([data_outliers_in_feature, df_temp.dropna()])

        data_outliers.append(data_outliers_in_feature.to_dict('list'))
        df = df.reset_index(drop=True)
        df = pd.concat([df, qs.T])
        df = df.T
        df = df.reset_index()
        df.columns.values[0] = "feature"
        df.columns = df.columns.map(str)
        data.append(df.to_dict('list'))

    outliers = dict(zip(titles, data_outliers))
    data = dict(zip(titles, data))

    sources_outlier = ColumnDataSource(data=outliers[titles[0]])
    sources = ColumnDataSource(data=data[titles[0]])
    
    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help, hover"
    p = figure(x_range=features, 
               tools=TOOLS,
               title="BoxPlot", 
               width=size, 
               height=size-200,
               background_fill_color="#eaefef", 
               y_axis_label=  "Log-10 Range" if log_transformation else "Range")
    
    # outlier range
    whisker = Whisker(base="feature", upper="upper", lower="lower", source=sources, line_width=1)
    whisker.upper_head.size = whisker.lower_head.size = 20
    p.add_layout(whisker)

    # quantile boxes
    cmap = factor_cmap("feature", inferno(len(features)), features)
    p.vbar("feature", 0.7, "q2", "q3", source=sources, color=cmap, line_color="black")
    p.vbar("feature", 0.7, "q1", "q2", source=sources, color=cmap, line_color="black")

    # outliers
    p.scatter("feature", "value", source=sources_outlier, size=6, color="black", alpha=0.3)
    
    p.xaxis.major_label_orientation = "vertical"
    select_benchmark = Select(title="Benchmarks: ", value=titles[0], options=titles)
    select_benchmark.js_on_change("value", CustomJS(args=dict(sources=sources, data=data, sources_outlier=sources_outlier, outliers=outliers), code="""
        sources.data = data[cb_obj.value];
        sources_outlier.data = outliers[cb_obj.value];
    """))

    if output_file_name != "":
        show(column(p, select_benchmark))
        output_file(output_file_name + ".html")
    return TabPanel(child=column(p, select_benchmark), title="Boxplots")



def make_all_plots(dataframe):
    print("25% done... \t")
    benchmarks = list(dataframe['host'].unique())
    platforms = list(dataframe['uuid'].unique())
    features = list(dataframe.columns.values)
    features.remove('host')
    features.remove('uuid')
    dataframes = []
    plts = []
    
    for s in platforms:
    #for l in benchmarks:
        df = pd.DataFrame()
        df = dataframe[dataframe['uuid'].astype('str') == s]
        if len(df) != 0:
            nunique = df.nunique()
            cols_to_update = nunique[nunique == 1].index
            for col in cols_to_update:
                if col != 'host' and col != 'uuid': # need change this to check whether feature is str type
                    random_vector = 0.000000001 * np.random.randint(low = 1, high = 3, size = len(df[col]))
                    df[col] += random_vector
            df = df.drop(['host', 'uuid'], axis=1)
            dataframes.append(df)  
    
    plts.append(interactive_correlation_heatmap(dataframes, benchmarks, features))
    plts.append(interactive_scatterplot(dataframes, benchmarks, features))
    plts.append(interactive_histgram(dataframes, benchmarks, features))
    plts.append(interactive_boxplot(dataframes, benchmarks, features, log_transformation = True))       
    print("50% done... \t")  
    
    y = dataframe['host']
    dataframe = dataframe.drop(['host', 'uuid'], axis=1)
    uncorr_arr, uncorr_features = removeCorrelatedFeatures(dataframe)
    selectedFeatures = []
    
    fig1 = selectFeaturesChart(uncorr_arr, y, uncorr_features, selectedFeatures, "mutual_info_classif")
    fig2 = selectFeaturesChart(uncorr_arr, y, uncorr_features, selectedFeatures, "chi2")
    fig3 = selectFeaturesChart(uncorr_arr, y, uncorr_features, selectedFeatures, "f_classif")
    selectedFeatures = np.unique(selectedFeatures)
    
    pre = PreText(text = """Final Selected Features:""",width=1000, height=10)
    p1 = Paragraph(text = str(selectedFeatures), width=1000, height=30)
    p2 = Paragraph(text = f"In total {len(selectedFeatures)} features have being selected", width=1000, height=60)
    div = Div(text = f"""
    The best {len(selectedFeatures)} features have being selected after filter out high correlation features, low variance features, and combination of information metrics from <i>mutual_info_classif</i>, <i>chi2</i>, <i>f_classif</i>.
    <ul>
    <li><i>mutual_info_classif</i> is mutual information estimation for a discrete target variable.</li>
    <li><i>chi2</i> compute chi-squared stats between each non-negative feature and class.</li>
    <li><i>f_classif</i> compute the ANOVA F-value for the provided sample.</li>
    </ul>
    Refer the <a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection">Sklearn API</a> for more information""", width=1000, height=80)
    
    print("75% done... \t")
    plts.append(TabPanel(child = column(children = [row(fig1, fig2, fig3), pre, p1, p2, div]), title="Feature Selection"))
    show(Tabs(tabs=plts))
    print("100% done! \t")
    output_file("EDA.html")
    
    

def load_file_to_dataframe(filename = ""):
    if filename == "":
        Tk().withdraw()
        filename = askopenfilename()
    try:
        df = pd.read_excel(filename)
    except:
        try:
            df = pd.read_csv(filename)
        except:
            print("Wrong filename, script exited.")
    return df



def drop_string_datetime_na(dataframe):
    dataframe = dataframe[list(dataframe.T[(dataframe.dtypes==np.float64) | (dataframe.dtypes==np.int64)].index) + ["host", "uuid"]]
    dataframe[dataframe.T[dataframe.dtypes==np.int64].index] = dataframe[dataframe.T[dataframe.dtypes==np.int64].index].astype(float)
    dataframe = dataframe.dropna(axis=1, how='all')
    nunique = dataframe.nunique()
    cols_to_drop = nunique[nunique == 1].index
    dataframe = dataframe.drop(list(cols_to_drop) + ["Unnamed: 0"], axis=1)
    return dataframe



def main():
    if len(sys.argv) <= 1:
        filePath = ""
    else:
        args = sys.argv[1]
        if os.path.isfile(args):
            filePath = args
        elif os.path.isfile(os.getcwd() + args):
            filePath = os.getcwd() + "/" + args
        else:
            print("Cannot find the file. Please choose in file folder.")
            filePath = ""
    start = time.time()
    df = load_file_to_dataframe(filePath)
    df = drop_string_datetime_na(df)
    make_all_plots(df)
    end = time.time()
    print(f"{end - start} seconds has been take to complete.")
    

    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()