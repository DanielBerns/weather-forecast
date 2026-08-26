from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot style for dark mode HTML
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e293b", "figure.facecolor": "#0f172a", "text.color": "#f8fafc", "axes.labelcolor": "#f8fafc", "xtick.color": "#f8fafc", "ytick.color": "#f8fafc"})

# Paths
DATA_DIR = Path("./data")
REPORT_DIR = Path("./report")
PLOTS_DIR = REPORT_DIR / "plots"
HTML_PATH = REPORT_DIR / "index.html"

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

class HTMLReport:
    def __init__(self, path):
        self.path = path
        self.sections = []

    def add_section(self, title, text=""):
        section_id = title.replace(" ", "-").lower().replace(".", "")
        html = f'<section id="{section_id}" class="glass-panel report-section">\n'
        html += f'  <h2 class="gradient-text">{title}</h2>\n'
        if text:
            html += f'  <p class="section-desc">{text}</p>\n'
        self.sections.append(html)

    def add_plot(self, title, image_filename, description=""):
        html = f'  <div class="plot-container">\n'
        html += f'    <h3>{title}</h3>\n'
        html += f'    <img class="report-img" src="plots/{image_filename}" alt="{title}">\n'
        if description:
            html += f'    <p class="plot-desc">{description}</p>\n'
        html += f'  </div>\n'
        self.sections[-1] += html

    def add_text(self, text, html_format=False):
        if html_format:
            self.sections[-1] += f'{text}\n'
        else:
            self.sections[-1] += f'  <p>{text}</p>\n'
            
    def close_section(self):
        self.sections[-1] += '</section>\n'

    def save(self):
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comodoro Rivadavia - Weather Data Quality</title>
    <link rel="stylesheet" href="style.css">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
</head>
<body>
    <div class="app-container">
        <aside class="sidebar glass-panel">
            <h1 class="gradient-text">Weather Report</h1>
            <p class="sidebar-subtitle">Comodoro Rivadavia Data Quality</p>
            <nav id="toc" class="toc"></nav>
        </aside>
        <main class="content">
"""
        html_content += "\n".join(self.sections)
        html_content += """
        </main>
    </div>
    <script src="script.js"></script>
</body>
</html>
"""
        with open(self.path, "w") as f:
            f.write(html_content)

def parse_hourly_temp(val_str):
    if pd.isna(val_str):
        return np.nan
    try:
        val_part, q_flag = val_str.split(',')
        val = int(val_part)
        if val == 9999 or val == -9999:
            return np.nan
        if q_flag in ['3', '7', '9']:
            return np.nan
        return val / 10.0
    except Exception:
        return np.nan

def df_to_html_table(df):
    return df.to_html(index=False, classes="data-table", border=0)

def main():
    report = HTMLReport(HTML_PATH)

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print("Loading data...")
    try:
        df_daily = pd.read_csv(DATA_DIR / "typ_daily_CR.csv", parse_dates=["DATE"])
        df_monthly = pd.read_csv(DATA_DIR / "typ_monthly_CR.csv", parse_dates=["DATE"])
        df_yearly = pd.read_csv(DATA_DIR / "typ_year_CR.csv", parse_dates=["DATE"])
        df_hourly = pd.read_csv(DATA_DIR / "tytd_hourly_CR.csv", parse_dates=["DATE"], dtype={"CALL_SIGN": str}, low_memory=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Clean Hourly Data
    print("Cleaning hourly data...")
    df_hourly['TMP_C'] = df_hourly['TMP'].apply(parse_hourly_temp)
    df_hourly['DEW_C'] = df_hourly['DEW'].apply(parse_hourly_temp)

    # ---------------------------------------------------------
    # Data Documentation
    # ---------------------------------------------------------
    print("Generating data documentation...")
    report.add_section("1. Data Documentation", "Overview of columns, extreme values for numeric variables, and unique values for categorical variables.")
    
    def document_dataframe(df, name):
        report.add_text(f'<h3 class="dataset-title">{name}</h3>', html_format=True)
        doc_data = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                c_min = df[col].min()
                c_max = df[col].max()
                val_str = f"Min: {c_min}, Max: {c_max}"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                c_min = df[col].min().strftime('%Y-%m-%d')
                c_max = df[col].max().strftime('%Y-%m-%d')
                val_str = f"Min: {c_min}, Max: {c_max}"
            else:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 10:
                    val_str = "Values: " + ", ".join(map(str, unique_vals))
                else:
                    val_str = f"{len(unique_vals)} unique categorical values"
            doc_data.append({"Column": col, "Type": col_type, "Details": val_str})
        
        doc_df = pd.DataFrame(doc_data)
        report.add_text(df_to_html_table(doc_df), html_format=True)

    document_dataframe(df_daily, "Daily Data")
    document_dataframe(df_monthly, "Monthly Data")
    document_dataframe(df_yearly, "Yearly Data")
    document_dataframe(df_hourly, "Hourly Data")
    report.close_section()

    # ---------------------------------------------------------
    # Visualizations of Numeric Columns
    # ---------------------------------------------------------
    print("Generating visualizations of numeric columns...")
    report.add_section("2. Numeric Distributions", "Histograms showing the distribution of numeric columns across the datasets.")

    def plot_numeric_distributions(df, name, prefix):
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return
        n_cols = len(num_cols)
        cols = 3
        rows = (n_cols + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = np.array(axes).flatten()
        
        for i, col in enumerate(num_cols):
            sns.histplot(df[col].dropna(), bins=30, ax=axes[i], kde=False, color="#38bdf8")
            axes[i].set_title(col, color="#f8fafc")
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            
        for j in range(len(num_cols), len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        filename = f"{prefix}_distributions.png"
        plt.savefig(PLOTS_DIR / filename, transparent=True)
        plt.close()
        
        report.add_plot(f"{name} Numeric Distributions", filename, f"Histograms of numeric columns in the {name} dataset.")

    plot_numeric_distributions(df_daily, "Daily Data", "daily")
    plot_numeric_distributions(df_monthly, "Monthly Data", "monthly")
    plot_numeric_distributions(df_yearly, "Yearly Data", "yearly")
    plot_numeric_distributions(df_hourly, "Hourly Data", "hourly")
    report.close_section()

    # ---------------------------------------------------------
    # Missing Values Analysis
    # ---------------------------------------------------------
    print("Analyzing missing values...")
    report.add_section("3. Missing Data", "Analyzing the proportion of missing (NaN) values across datasets.")

    def plot_missing(df, name, filename):
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if len(missing_pct) == 0:
            report.add_text(f"<strong>{name}</strong>: No missing values found.<br>", html_format=True)
            return

        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing_pct.values, y=missing_pct.index, color="#f43f5e")
        plt.xlabel('Percentage of Missing Values (%)', color="#f8fafc")
        plt.title(f'Missing Values in {name}', color="#f8fafc")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, transparent=True)
        plt.close()
        
        report.add_plot(f"Missing Values: {name}", filename, f"Columns in {name} with missing percentages.")

    plot_missing(df_daily, "Daily Data", "missing_daily.png")
    plot_missing(df_monthly, "Monthly Data", "missing_monthly.png")
    plot_missing(df_yearly, "Yearly Data", "missing_yearly.png")
    plot_missing(df_hourly, "Hourly Data", "missing_hourly.png")
    report.close_section()

    # ---------------------------------------------------------
    # Outlier Analysis (Boxplots)
    # ---------------------------------------------------------
    print("Analyzing outliers...")
    report.add_section("4. Outlier Detection", "Visualizing temperature distributions to identify extreme outliers.")

    plt.figure(figsize=(10, 6))
    temp_cols = ['TMAX', 'TMIN', 'TAVG']
    df_daily_melt = df_daily.melt(value_vars=temp_cols, var_name='Variable', value_name='Temperature (C)')
    sns.boxplot(x='Variable', y='Temperature (C)', data=df_daily_melt, hue='Variable', legend=False, palette="cool")
    plt.title('Daily Temperature Distributions (Boxplot)', color="#f8fafc")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "outliers_daily_temp.png", transparent=True)
    plt.close()
    
    report.add_plot("Daily Temperatures Outliers", "outliers_daily_temp.png", "Boxplots for TMAX, TMIN, and TAVG. Points beyond the whiskers are statistical outliers.")

    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df_hourly['TMP_C'], color="#a855f7")
    plt.title('Hourly Temperature Distribution (Boxplot)', color="#f8fafc")
    plt.xlabel('Temperature (C)', color="#f8fafc")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "outliers_hourly_temp.png", transparent=True)
    plt.close()

    report.add_plot("Hourly Temperature Outliers", "outliers_hourly_temp.png", "Boxplot showing the distribution of hourly temperatures.")
    report.close_section()

    # ---------------------------------------------------------
    # Suspicious Patterns Detection
    # ---------------------------------------------------------
    print("Checking suspicious patterns...")
    report.add_section("5. Suspicious Patterns", "Verifying the physical logic of the temperature readings.")

    invalid_temp_rows = df_daily[df_daily['TMAX'] < df_daily['TMIN']]
    report.add_text(f"<strong>TMAX &lt; TMIN check:</strong> Found {len(invalid_temp_rows)} days where the maximum temperature is lower than the minimum.", html_format=True)
    if len(invalid_temp_rows) > 0:
        report.add_text("Examples of TMAX &lt; TMIN anomalies:", html_format=True)
        report.add_text(df_to_html_table(invalid_temp_rows[['DATE', 'TMAX', 'TMIN']].head(5)), html_format=True)

    invalid_tavg = df_daily[(df_daily['TAVG'] < df_daily['TMIN']) | (df_daily['TAVG'] > df_daily['TMAX'])]
    report.add_text(f"<strong>TAVG bounds check:</strong> Found {len(invalid_tavg)} days where the average temperature is outside the [TMIN, TMAX] range.", html_format=True)

    extreme_high = df_daily[df_daily['TMAX'] > 45]
    extreme_low = df_daily[df_daily['TMIN'] < -20]
    report.add_text(f"<strong>Extreme Values Check:</strong> Found {len(extreme_high)} days with TMAX &gt; 45°C, and {len(extreme_low)} days with TMIN &lt; -20°C.", html_format=True)

    import plotly.graph_objects as go

    # To ensure smooth browser interactivity, we calculate daily averages from the hourly data
    df_hourly_daily = df_hourly.set_index('DATE').resample('D').mean(numeric_only=True).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=df_hourly_daily['DATE'], y=df_hourly_daily['TMP_C'], mode='lines', name='Avg Temperature (C)', opacity=0.8, line=dict(color='#38bdf8', width=1)))
    fig.add_trace(go.Scattergl(x=df_hourly_daily['DATE'], y=df_hourly_daily['DEW_C'], mode='lines', name='Avg Dew Point (C)', opacity=0.8, line=dict(color='#a855f7', width=1)))

    fig.update_layout(
        title='Daily Average Temperature and Dew Point (Interactive)',
        xaxis_title='Date',
        yaxis_title='Degrees Celsius',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    interactive_path = PLOTS_DIR / 'hourly_interactive.html'
    fig.write_html(interactive_path, include_plotlyjs='cdn')

    report.add_text('<h3 class="dataset-title">Interactive Time Series (TMP_C & DEW_C)</h3>', html_format=True)
    report.add_text('<p class="plot-desc">This interactive plot shows the long-term trends of Temperature and Dew Point. The hourly data was aggregated into daily averages to ensure smooth interactive performance in the browser. You can zoom in and pan across the timeline.</p>', html_format=True)
    report.add_text(f'<iframe src="plots/hourly_interactive.html" width="100%" height="500px" frameborder="0" style="border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05); margin-top: 20px;"></iframe>', html_format=True)

    report.close_section()

    report.save()
    print(f"Data quality analysis complete. Web report generated at {HTML_PATH}")

if __name__ == "__main__":
    main()
