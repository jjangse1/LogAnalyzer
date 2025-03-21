import streamlit as st
import pandas as pd
import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Line, Grid
from streamlit_echarts import st_pyecharts
from fpdf import FPDF
from dtaidistance import dtw
import os
import tempfile
import traceback
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import seaborn as sns

# Set font and backend configuration
plt.rcParams['font.family'] = 'Arial'
matplotlib.use('Agg')  # Backend configuration

# Font settings
FONT_PATH = "Arial.ttf"
st.set_page_config(layout="wide")

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self._has_korean_font = False
        # Page margin settings - wider margins
        self.set_margins(20, 20, 20)

    def header(self):
        if self._has_korean_font:
            self.set_font("Arial", size=12)
        else:
            self.set_font("Arial", size=12)
        self.cell(170, 10, txt="Log Analysis Report", ln=True, align="C")
        self.ln(5)

    def add_korean_font(self):
        if os.path.exists(FONT_PATH):
            try:
                self.add_font("Arial", "", FONT_PATH, uni=True)
                self._has_korean_font = True
                return True
            except Exception as e:
                st.warning(f"Font add failed: {str(e)}")
                return False

    def add_image(self, image_path, x=None, y=None, w=0, h=0, title=None):
        if title:
            if self._has_korean_font:
                self.set_font("Arial", 'B', 10)
            else:
                self.set_font('Arial', 'B', 10)
            # Use multi_cell for long titles to prevent truncation
            self.multi_cell(0, 8, title, 0, 'L')

        if os.path.exists(image_path):
            # Image size adjustment - prevent oversizing
            if x is not None and y is not None:
                self.image(image_path, x=x, y=y, w=w, h=h)
            else:
                # Adjust image width considering margins
                effective_width = self.w - 2 * self.l_margin
                if w > effective_width or w == 0:
                    w = effective_width
                self.image(image_path, w=w, h=h)
            self.ln(5)
            return True
        return False

    def cell(self, w, h=0, txt='', border=0, ln=0, align='', fill=False, link=''):
        try:
            # Text processing - truncate overly long text
            if w > 0 and txt and self.get_string_width(txt) > w:
                txt = self._truncate_text(txt, w)
            safe_txt = ''.join(c if ord(c) < 128 else '_' for c in txt)
            super().cell(w, h, safe_txt, border, ln, align, fill, link)
        except Exception:
            super().cell(w, h, '', border, ln, align, fill, link)

    def _truncate_text(self, text, width):
        # Truncate text that's too long and add "..."
        if not text:
            return ''
        while self.get_string_width(text + '...') > width and len(text) > 0:
            text = text[:-1]
        return text + '...' if len(text) < len(text) else text

    def multi_cell(self, w, h, txt, border=0, align='J', fill=False):
        """Improved multi_cell with automatic line breaks"""
        try:
            safe_txt = ''.join(c if ord(c) < 128 else '_' for c in txt)
            super().multi_cell(w, h, safe_txt, border, align, fill)
        except Exception as e:
            super().multi_cell(w, h, '', border, align, fill)

    def add_table(self, headers, data, col_widths=None):
        """Table function with text wrapping support"""
        line_height = 7

        # Calculate effective page width
        effective_width = self.w - 2 * self.l_margin

        # Calculate column widths - dynamic adjustment
        if col_widths is None:
            # Adjust width based on header length
            col_widths = []
            for header in headers:
                w = self.get_string_width(str(header)) + 6  # Add padding
                col_widths.append(w)

            # Check data length and adjust width
            for row in data:
                for i, cell in enumerate(row):
                    if i < len(col_widths):
                        w = self.get_string_width(str(cell)) + 6
                        col_widths[i] = max(col_widths[i], w)

            # Adjust ratios if total width exceeds page width
            total_width = sum(col_widths)
            if total_width > effective_width:
                ratio = effective_width / total_width
                col_widths = [w * ratio for w in col_widths]

        # Add headers
        self.set_font('Arial', 'B', 10)
        self.set_fill_color(200, 200, 200)

        # Save current position
        x_pos = self.get_x()
        y_pos = self.get_y()

        # Variable for max height per cell
        max_height = line_height

        # Store header text and calculate max height
        for i, header in enumerate(headers):
            header_str = str(header)
            self.set_xy(x_pos, y_pos)
            self.multi_cell(col_widths[i], line_height, header_str, 1, 'C', True)
            max_height = max(max_height, self.get_y() - y_pos)
            x_pos += col_widths[i]

        # Move to next line
        self.ln(max_height)

        # Add data
        self.set_font('Arial', '', 10)
        self.set_fill_color(255, 255, 255)

        for row in data:
            x_pos = self.get_x()
            y_pos = self.get_y()
            max_height = line_height

            # Add new page if near bottom
            if self.get_y() > self.h - 30:
                self.add_page()
                x_pos = self.get_x()
                y_pos = self.get_y()

            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cell_str = str(cell)
                    self.set_xy(x_pos, y_pos)
                    self.multi_cell(col_widths[i], line_height, cell_str, 1, 'C')
                    max_height = max(max_height, self.get_y() - y_pos)
                    x_pos += col_widths[i]

            # Move to next row
            self.set_y(y_pos + max_height)

        self.ln(5)

def create_line_chart(df1, df2, sensor, title, ref_name, compare_name, highlight_anomalies=False, step_range=None):
    """Function to create line charts with anomaly highlighting"""
    plt.figure(figsize=(10, 6))
    
    # Filter by step range if provided
    if step_range:
        df1 = df1[(df1['step'] >= step_range[0]) & (df1['step'] <= step_range[1])]
        df2 = df2[(df2['step'] >= step_range[0]) & (df2['step'] <= step_range[1])]
    
    plt.plot(df1['step'], df1[sensor], label=ref_name, linewidth=2)
    plt.plot(df2['step'], df2[sensor], label=compare_name, linewidth=2, linestyle='--')
    
    # Highlight anomalies if enabled
    if highlight_anomalies:
        # Calculate threshold
        threshold = df1[sensor].std() * 2  # Using 2 standard deviations
        
        # Find points where difference exceeds threshold
        combined_df = pd.merge(
            df1[['step', sensor]], 
            df2[['step', sensor]], 
            on='step', 
            suffixes=('_ref', '_comp')
        )
        combined_df['diff'] = abs(combined_df[f'{sensor}_ref'] - combined_df[f'{sensor}_comp'])
        anomalies = combined_df[combined_df['diff'] > threshold]
        
        if not anomalies.empty:
            plt.scatter(
                anomalies['step'], 
                anomalies[f'{sensor}_ref'],
                color='red', 
                s=50, 
                zorder=5,
                label='Anomalies'
            )
            
            # Add red vertical lines at anomaly points
            for step in anomalies['step']:
                plt.axvline(x=step, color='red', alpha=0.2)
    
    plt.title(f"{title}", fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save image to memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return buf

def detect_anomalies(df1, df2, threshold_factor=0.3):
    """Anomaly detection function"""
    diff = df1.set_index('step') - df2.set_index('step')
    abs_diff = diff.abs()
    threshold = df1.set_index('step').abs().mean() * threshold_factor
    outliers = abs_diff[abs_diff > threshold].dropna(how="all")
    return outliers, diff

def calculate_dtw_similarity(df1, df2):
    """Calculate DTW similarity function"""
    scores = {}
    for col in df1.columns:
        if col != 'step' and col in df2.columns:
            try:
                series1 = df1[col].fillna(0).tolist()
                series2 = df2[col].fillna(0).tolist()

                min_len = min(len(series1), len(series2))
                series1 = series1[:min_len]
                series2 = series2[:min_len]

                if min_len > 0:
                    score = dtw.distance(series1, series2)
                    scores[col] = score
            except Exception as e:
                pass
    return scores

# Streamlit app start
st.title("📊 Log File Comparison Analysis")
uploaded_files = st.file_uploader("Upload CSV or TXT log files", type=["csv", "txt"],
                                 accept_multiple_files=True)

if uploaded_files:
    dataframes = {}
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding="cp949")
            except Exception as e:
                st.error(f"Failed to read file '{uploaded_file.name}': {str(e)}")
                continue

        if 'step' not in df.columns:
            st.error(f"File '{uploaded_file.name}' doesn't have a 'step' column which is required for analysis.")
            continue

        dataframes[uploaded_file.name] = df

    if len(dataframes) < 2:
        st.warning("At least 2 valid files are needed for analysis.")
    else:
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("기준 로그 선택")
            # 정렬 옵션 추가
            sort_option = st.radio("로그 정렬 방식:", ["이름 순", "날짜 순 (파일명에 날짜 포함 시)"])
            
            # 검색 필터 추가
            search_term = st.text_input("로그 파일 검색:", "")
            
            # 파일 정렬 및 필터링
            log_files = list(dataframes.keys())
            
            # 검색어로 필터링
            if search_term:
                log_files = [log for log in log_files if search_term.lower() in log.lower()]
            
            # 정렬
            if sort_option == "이름 순":
                log_files.sort()
            else:
                # 날짜 포함된 파일명 정렬 (예: log_20230101.csv)
                # 실제 환경에 맞게 날짜 형식 조정 필요
                import re
                def extract_date(filename):
                    date_match = re.search(r'(\d{6,8})', filename)
                    return date_match.group(1) if date_match else filename
                log_files.sort(key=extract_date)
            
            ref_log = st.selectbox("기준 로그 파일:", log_files)
            ref_df = dataframes[ref_log]

        with col2:
            st.subheader("비교 로그 선택")
            
            other_logs = [key for key in dataframes.keys() if key != ref_log]
            
            # 그룹 선택 옵션 추가
            selection_mode = st.radio("선택 모드:", ["개별 선택", "범위 선택", "패턴 선택"])
            
            compare_logs = []
            
            if selection_mode == "개별 선택":
                compare_logs = st.multiselect("비교할 로그 파일:", other_logs)
            
            elif selection_mode == "범위 선택":
                if len(other_logs) > 0:
                    start_idx = st.selectbox("시작 로그:", other_logs, index=0)
                    end_idx = st.selectbox("종료 로그:", other_logs, index=min(5, len(other_logs)-1))
                    
                    start_pos = other_logs.index(start_idx)
                    end_pos = other_logs.index(end_idx)
                    
                    if start_pos <= end_pos:
                        compare_logs = other_logs[start_pos:end_pos+1]
                    else:
                        compare_logs = other_logs[end_pos:start_pos+1]
                    
                    st.write(f"선택된 로그 파일 수: {len(compare_logs)}")
                    
                    # 선택된 로그 파일 프리뷰 (처음 3개와 마지막 3개만 표시)
                    if len(compare_logs) > 6:
                        st.write(f"선택된 파일: {', '.join(compare_logs[:3])} ... {', '.join(compare_logs[-3:])}")
                    else:
                        st.write(f"선택된 파일: {', '.join(compare_logs)}")
            
            elif selection_mode == "패턴 선택":
                pattern = st.text_input("파일명 패턴 (정규식 지원):", "")
                
                if pattern:
                    import re
                    try:
                        regex = re.compile(pattern)
                        compare_logs = [log for log in other_logs if regex.search(log)]
                        
                        st.write(f"패턴과 일치하는 파일 수: {len(compare_logs)}")
                        
                        # 선택된 로그 파일 프리뷰
                        if len(compare_logs) > 6:
                            st.write(f"선택된 파일: {', '.join(compare_logs[:3])} ... {', '.join(compare_logs[-3:])}")
                        elif len(compare_logs) > 0:
                            st.write(f"선택된 파일: {', '.join(compare_logs)}")
                        else:
                            st.warning("일치하는 파일이 없습니다.")
                    except re.error:
                        st.error("잘못된 정규식 패턴입니다.")
            
            # 선택된 파일 수 표시
            st.metric("선택된 비교 로그 수", len(compare_logs))
            
            # 전체 선택/해제 버튼
            col2_1, col2_2 = st.columns(2)
            
            # 세션 상태를 사용하여 선택 상태 관리
            if 'select_all' not in st.session_state:
                st.session_state.select_all = False
            
            with col2_1:
                if st.button("모두 선택"):
                    compare_logs = other_logs.copy()
                    st.session_state.select_all = True
                    st.rerun()  # experimental_rerun() 대신 rerun() 사용
            
            with col2_2:
                if st.button("모두 해제"):
                    compare_logs = []
                    st.session_state.select_all = False
                    st.rerun()  # experimental_rerun() 대신 rerun() 사용
            
            # 세션 상태에 따라 자동 선택
            if st.session_state.select_all and selection_mode == "개별 선택":
                compare_logs = other_logs.copy()

        if compare_logs:
            df1_mean = ref_df.groupby("step").mean(numeric_only=True).reset_index()

            common_columns = list(set(df1_mean.columns) & set(dataframes[compare_logs[0]].columns))
            if "step" in common_columns:
                common_columns = sorted(common_columns, key=lambda x: 0 if x == "step" else 1)

            if len(common_columns) <= 1:
                st.error("No common numeric columns between files.")
            else:
                df1_mean = df1_mean[common_columns]

                numeric_columns = [col for col in common_columns if col != 'step']
                selected_sensors = st.multiselect("Select sensors to display", numeric_columns,
                                                default=numeric_columns[:min(5, len(numeric_columns))])
                # Chart display options
                chart_options_col1, chart_options_col2, chart_options_col3 = st.columns(3)
                with chart_options_col1:
                    chart_display_mode = st.radio("차트 표시 모드:", ["겹쳐서 보기", "따로 보기"], index=0)
                with chart_options_col2:
                    highlight_anomalies = st.checkbox("이상 구간 하이라이트", value=True)
                with chart_options_col3:
                    step_range = st.slider("Step 범위 선택:", 
                                        min_value=int(df1_mean['step'].min() if not df1_mean.empty else 0),
                                        max_value=int(df1_mean['step'].max() if not df1_mean.empty else 100),
                                        value=(int(df1_mean['step'].min() if not df1_mean.empty else 0),
                                                int(df1_mean['step'].max() if not df1_mean.empty else 100)))
                if selected_sensors:
                    st.subheader("📊 Sensor Comparison Charts")

                    for idx, sensor in enumerate(selected_sensors):
                        line = Line()
                        x_values = df1_mean['step'].tolist()
                        if not x_values:
                            st.error(f"X-axis values (step from {ref_log}) are empty.")
                            continue

                        line.add_xaxis(x_values)

                        # Add reference log data
                        series_ref = df1_mean[sensor].fillna(0).tolist()
                        line.add_yaxis(
                            f"{ref_log}",
                            series_ref,
                            is_smooth=True,
                            linestyle_opts=opts.LineStyleOpts(width=2),
                            label_opts=opts.LabelOpts(position="bottom")
                        )

                        # Add comparison log data
                        for compare_log in compare_logs:
                            compare_df = dataframes[compare_log]
                            df_compare_mean = compare_df.groupby("step").mean(numeric_only=True).reset_index()
                            series_compare = df_compare_mean[sensor].fillna(0).tolist()

                            line.add_yaxis(
                                f"{compare_log}",
                                series_compare,
                                is_smooth=True,
                                linestyle_opts=opts.LineStyleOpts(width=2),
                                label_opts=opts.LabelOpts(position="bottom")
                            )

                        line.set_global_opts(
                            title_opts=opts.TitleOpts(
                                title=f"{sensor} Compare",
                                pos_left="center",
                                pos_top="5%"
                            ),
                            legend_opts=opts.LegendOpts(
                                pos_top="15%",
                                orient="horizontal"
                            ),
                            xaxis_opts=opts.AxisOpts(
                                name="Step",
                                name_location="center",
                                name_gap=25
                            ),
                            yaxis_opts=opts.AxisOpts(
                                name="Value",
                                name_location="center",
                                name_gap=40
                            ),
                            tooltip_opts=opts.TooltipOpts(trigger="axis")
                        )

                        st_pyecharts(line, height="400px", key=f"chart_{idx}")
                        # Create Matplotlib charts based on display mode
                        if chart_display_mode == "따로 보기":
                            st.subheader("📊 개별 센서 차트 (Matplotlib)")
                            for idx, sensor in enumerate(selected_sensors):
                                cols = st.columns(len(compare_logs))
                                for i, compare_log in enumerate(compare_logs):
                                    with cols[i]:
                                        compare_df = dataframes[compare_log]
                                        df2_mean = compare_df.groupby("step").mean(numeric_only=True).reset_index()[common_columns]
                                        
                                        # Filter by step range
                                        filtered_df1 = df1_mean[(df1_mean['step'] >= step_range[0]) & (df1_mean['step'] <= step_range[1])]
                                        filtered_df2 = df2_mean[(df2_mean['step'] >= step_range[0]) & (df2_mean['step'] <= step_range[1])]
                                        
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        ax.plot(filtered_df1['step'], filtered_df1[sensor], label=ref_log, linewidth=2)
                                        ax.plot(filtered_df2['step'], filtered_df2[sensor], label=compare_log, linewidth=2, linestyle='--')
                                        
                                        # Highlight anomalies if enabled
                                        if highlight_anomalies:
                                            threshold = filtered_df1[sensor].std() * 2
                                            combined_df = pd.merge(
                                                filtered_df1[['step', sensor]], 
                                                filtered_df2[['step', sensor]], 
                                                on='step', 
                                                suffixes=('_ref', '_comp')
                                            )
                                            combined_df['diff'] = abs(combined_df[f'{sensor}_ref'] - combined_df[f'{sensor}_comp'])
                                            anomalies = combined_df[combined_df['diff'] > threshold]
                                            
                                            if not anomalies.empty:
                                                ax.scatter(
                                                    anomalies['step'], 
                                                    anomalies[f'{sensor}_ref'],
                                                    color='red', 
                                                    s=50, 
                                                    zorder=5,
                                                    label='Anomalies'
                                                )
                                                
                                                # Add red vertical lines at anomaly points
                                                for step in anomalies['step']:
                                                    ax.axvline(x=step, color='red', alpha=0.2)
                                        
                                        ax.set_title(f"{sensor}: {ref_log} vs {compare_log}")
                                        ax.set_xlabel('Step')
                                        ax.set_ylabel('Value')
                                        ax.grid(True, linestyle='--', alpha=0.7)
                                        ax.legend()
                                        plt.tight_layout()
                                        st.pyplot(fig)
                    # Add styling for Korean font support
                    st.markdown(
                        """
                        <style>
                        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
                        html, body, [class*="css"] {
                            font-family: 'Noto Sans KR', sans-serif;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Store analysis results for PDF report
                    analysis_results = {}

                    for compare_log in compare_logs:
                        st.subheader(f"{ref_log} vs {compare_log} Analysis Results")
                        compare_df = dataframes[compare_log]
                        df2_mean = compare_df.groupby("step").mean(numeric_only=True).reset_index()[common_columns]

                        # Initialize results dictionary
                        analysis_results[compare_log] = {
                            'charts': [],
                            'outliers': None,
                            'dtw_scores': None
                        }

                        # Create charts for each sensor (using Matplotlib)
                        for sensor in selected_sensors:
                            buf = create_line_chart(
                                df1_mean, df2_mean, sensor, f"{sensor} Compare", ref_log, compare_log
                            )
                             # Create charts for each sensor with options
                            buf = create_line_chart(
                                df1_mean, df2_mean, sensor, f"{sensor} Compare", 
                                ref_log, compare_log,
                                highlight_anomalies=highlight_anomalies,
                                step_range=step_range
                            )
                            analysis_results[compare_log]['charts'].append({
                                'sensor': sensor,
                                'image_data': buf
                            })

                        # Anomaly analysis
                        outliers, diff_df = detect_anomalies(df1_mean, df2_mean)
                        analysis_results[compare_log]['outliers'] = outliers
                        analysis_results[compare_log]['diff_df'] = diff_df

                        # Calculate DTW similarity
                        scores = calculate_dtw_similarity(df1_mean, df2_mean)
                        analysis_results[compare_log]['dtw_scores'] = scores

                        tab1, tab2 = st.tabs(["Anomaly Analysis", "DTW Similarity"])
                        with tab1:
                            if not outliers.empty:
                                st.write("⚠️ Anomalous patterns detected!")
                                st.dataframe(outliers.style.highlight_max(axis=1, color='#ffcccc'))
                            else:
                                st.info("✅ No anomalous patterns found.")

                        with tab2:
                            if scores:
                                score_df = pd.DataFrame(list(scores.items()), columns=["Sensor", "DTW Score"])
                                st.write("📏 DTW Similarity Scores (lower means higher similarity):")
                                st.dataframe(score_df.style.bar(subset=["DTW Score"], color='#d3d3d3'))

                                # Visualize DTW results
                                fig, ax = plt.figure(figsize=(10, 6)), plt.subplot()
                                sns.barplot(x="Sensor", y="DTW Score", data=score_df)
                                plt.title("DTW Similarity by Sensor")
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("DTW similarity scores were not calculated.")

                        # Analyze anomaly pattern steps
                        if not outliers.empty:
                            st.subheader("Analysis of Anomaly Pattern Steps")
                            anomaly_steps = outliers.index.tolist()

                            # Display steps with anomaly patterns
                            st.write(f"Steps with detected anomaly patterns: {', '.join(map(str, anomaly_steps))}")

                            # Compare values at anomaly steps
                            for step in anomaly_steps[:5]:  # Show only top 5
                                st.write(f"Sensor value comparison at Step {step}:")
                                step_df1 = df1_mean[df1_mean['step'] == step]
                                step_df2 = df2_mean[df2_mean['step'] == step]

                                if not step_df1.empty and not step_df2.empty:
                                    comparison = pd.DataFrame({
                                        'Sensor': selected_sensors,
                                        f'{ref_log} Value': [step_df1[sensor].values[0] if not step_df1[sensor].empty else 0 for sensor in selected_sensors],
                                        f'{compare_log} Value': [step_df2[sensor].values[0] if not step_df2[sensor].empty else 0 for sensor in selected_sensors],
                                        'Difference': [step_df1[sensor].values[0] - step_df2[sensor].values[0] if not step_df1[sensor].empty and not step_df2.empty else 0 for sensor in selected_sensors]
                                    })
                                    st.dataframe(comparison.style.bar(subset=['Difference'], color='#ffcccc'))

                    # Add PDF report generation button
                    if st.button("📄 Download Complete PDF Report"):
                        try:
                            with st.spinner("Generating PDF..."):
                                # Create temporary directory
                                temp_dir = tempfile.mkdtemp()

                                # Create PDF
                                pdf = PDF()
                                pdf.add_korean_font()  # Try to add Korean font
                                pdf.add_page()

                                # Title page
                                pdf.set_font("Arial", "B", 16)
                                pdf.cell(0, 10, "Log Analysis Report", ln=True, align='C')
                                pdf.ln(1)

                                pdf.set_font("Arial", "", 12)
                                # Use multi_cell for long filenames
                                pdf.multi_cell(0, 10, f"Reference Log: {ref_log}")
                                # Handle comparison log list with multi_cell
                                pdf.multi_cell(0, 10, f"Comparison Logs: {', '.join(compare_logs)}")
                                pdf.ln(1)

                                # Add analysis results for each comparison log
                                for compare_log, results in analysis_results.items():
                                    pdf.add_page()
                                    pdf.set_font("Arial", "B", 14)
                                    # Use multi_cell for auto line breaks with long filenames
                                    pdf.multi_cell(0, 10, f"Analysis:\n{ref_log} vs {compare_log}")
                                    pdf.ln(5)

                                    # Add sensor charts
                                    pdf.set_font("Arial", "B", 12)
                                    pdf.multi_cell(0, 10, "Sensor Comparisons")

                                    for chart_data in results['charts']:
                                        # Read image from buffer
                                        chart_data['image_data'].seek(0)
                                        img_path = os.path.join(temp_dir, f"{chart_data['sensor']}.png")
                                        with open(img_path, 'wb') as f:
                                            f.write(chart_data['image_data'].getvalue())

                                        # Add image with appropriate width
                                        img_width = 160  # Width considering margins
                                        pdf.add_image(img_path, w=img_width, title=f"{chart_data['sensor']} Sensor Comparison")

                                    # Add anomaly detection results
                                    pdf.ln(5)
                                    pdf.set_font("Arial", "B", 12)
                                    pdf.multi_cell(0, 10, "Anomaly Detection Results")

                                    if results['outliers'] is not None and not results['outliers'].empty:
                                        pdf.set_font("Arial", "", 10)
                                        pdf.multi_cell(0, 10, "Anomalies Detected!")

                                        # Convert outlier data to table
                                        outliers_reset = results['outliers'].reset_index()
                                        # Select only important columns if too many
                                        if len(results['outliers'].columns) > 6:
                                            # Select most important columns
                                            important_cols = list(results['outliers'].abs().mean().nlargest(5).index)
                                            headers = ["Step"] + important_cols
                                            pdf.multi_cell(0, 10, "(Showing top 5 most significant columns)")
                                        else:
                                            headers = ["Step"] + list(results['outliers'].columns)

                                        # Prepare outlier data (limited to 10 rows)
                                        data = []
                                        for _, row in outliers_reset.head(10).iterrows():
                                            data_row = [str(row['step'])]
                                            for col in headers[1:]:
                                                if col in row:
                                                    data_row.append(f"{row[col]:.2f}")
                                            data.append(data_row)

                                        # Add outlier table
                                        pdf.add_table(headers, data)

                                        # Analyze anomaly steps
                                        anomaly_steps = results['outliers'].index.tolist()
                                        pdf.set_font("Arial", "B", 12)
                                        pdf.multi_cell(0, 10, "Anomaly Steps Analysis")
                                        pdf.set_font("Arial", "", 10)

                                        # Limit steps if too many
                                        if len(anomaly_steps) > 10:
                                            step_text = f"Anomalies detected at steps: {', '.join(map(str, anomaly_steps[:10]))}... (and {len(anomaly_steps)-10} more)"
                                        else:
                                            step_text = f"Anomalies detected at steps: {', '.join(map(str, anomaly_steps))}"

                                        # Use auto line breaking for long text
                                        pdf.multi_cell(0, 10, step_text)

                                    else:
                                        pdf.set_font("Arial", "", 10)
                                        pdf.multi_cell(0, 10, "No anomalies detected.")

                                    # Add DTW similarity results
                                    pdf.ln(5)
                                    pdf.set_font("Arial", "B", 12)
                                    pdf.multi_cell(0, 10, "DTW Similarity Scores")

                                    if results['dtw_scores']:
                                        # Convert DTW scores to table
                                        headers = ["Sensor", "DTW Score"]

                                        # Limit data if too many (top 10)
                                        if len(results['dtw_scores']) > 10:
                                            sorted_scores = sorted(results['dtw_scores'].items(), key=lambda x: x[1])[:10]
                                            data = [[sensor, f"{score:.4f}"] for sensor, score in sorted_scores]
                                            pdf.multi_cell(0, 10, "(Showing top 10 most similar sensors)")
                                        else:
                                            data = [[sensor, f"{score:.4f}"] for sensor, score in results['dtw_scores'].items()]

                                        # Add table
                                        pdf.add_table(headers, data)

                                        # Add DTW visualization
                                        dtw_df = pd.DataFrame(list(results['dtw_scores'].items()),
                                                            columns=["Sensor", "DTW Score"])

                                        # Show only top/bottom few if too many sensors
                                        if len(dtw_df) > 10:
                                            top_sensors = dtw_df.nsmallest(10, "DTW Score")
                                            plt.figure(figsize=(10, 6))
                                            sns.barplot(x="Sensor", y="DTW Score", data=top_sensors)
                                            plt.title("Top 10 Most Similar Sensors by DTW")
                                        else:
                                            plt.figure(figsize=(10, 6))
                                            sns.barplot(x="Sensor", y="DTW Score", data=dtw_df)
                                            plt.title("DTW Similarity by Sensor")

                                        plt.xticks(rotation=45)
                                        plt.tight_layout()

                                        dtw_path = os.path.join(temp_dir, f"dtw_{compare_log}.png")
                                        plt.savefig(dtw_path, format='png', dpi=100, bbox_inches='tight')
                                        plt.close()

                                        # Add image with appropriate width
                                        pdf.add_image(dtw_path, w=160)
                                    else:
                                        pdf.set_font("Arial", "", 10)
                                        pdf.multi_cell(0, 10, "DTW scores not calculated.")

                                # Add summary page
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 14)
                                pdf.cell(0, 10, "Summary", ln=True)
                                pdf.ln(5)

                                pdf.set_font("Arial", "", 12)

                                # Summary of overall results
                                for compare_log, results in analysis_results.items():
                                    pdf.set_font("Arial", "B", 12)
                                    pdf.cell(0, 10, f"{ref_log} vs", ln=True)
                                    pdf.cell(0, 10, f"{compare_log}:", ln=True)
                                    pdf.set_font("Arial", "", 10)

                                    # Anomaly summary
                                    if results['outliers'] is not None and not results['outliers'].empty:
                                        pdf.cell(0, 10, f"- Anomalies detected in {len(results['outliers'])} steps", ln=True)
                                        most_anomalous = results['outliers'].abs().mean().sort_values(ascending=False)
                                        if not most_anomalous.empty:
                                            pdf.cell(0, 10, f"- Most anomalous sensor: {most_anomalous.index[0]}", ln=True)
                                    else:
                                        pdf.cell(0, 10, "- No anomalies detected", ln=True)

                                    # DTW summary
                                    if results['dtw_scores']:
                                        dtw_df = pd.DataFrame(list(results['dtw_scores'].items()),
                                                            columns=["Sensor", "DTW Score"])
                                        avg_dtw = dtw_df["DTW Score"].mean()
                                        min_dtw_sensor = dtw_df.loc[dtw_df["DTW Score"].idxmin()]["Sensor"]
                                        pdf.cell(0, 10, f"- Average DTW score: {avg_dtw:.4f}", ln=True)
                                        pdf.cell(0, 10, f"- Most similar sensor: {min_dtw_sensor}", ln=True)

                                    pdf.ln(5)

                                # Generate PDF file
                                pdf_output = pdf.output(dest="S").encode("latin-1")

                                # Create download button
                                st.download_button(
                                    label="Download PDF",
                                    data=pdf_output,
                                    file_name="log_analysis_report.pdf",
                                    mime="application/pdf"
                                )
                        except Exception as e:
                            st.error(f"Error occurred while generating report: {str(e)}")
                            st.error(f"Error details: {e.__class__.__name__}")
                            st.error(f"Error traceback:\n{traceback.format_exc()}")

                else:
                    st.warning("Please select at least one sensor to display.")
        else:
            st.warning("Please select at least one log to compare.")