import streamlit as st
import pandas as pd
import requests

# 1. Page Configuration
st.set_page_config(page_title="AI Threat Detection Dashboard", layout="wide", page_icon="🛡️")

# 2. Load CSS Styling
def load_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

# ================= SIDEBAR =================
st.sidebar.header("📁 1. Upload New Log File")
st.sidebar.caption("Supports files up to 100MB.")
uploaded_file = st.sidebar.file_uploader("Select Linux log file (.log, .txt)", type=["log", "txt"])

if uploaded_file is not None:
    if uploaded_file.size > 100 * 1024 * 1024:
        st.sidebar.error("❌ File size exceeds 100MB limit!")
    else:
        if st.sidebar.button("📤 Upload to Database"):
            with st.spinner("Uploading to Elasticsearch..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
                    upload_res = requests.post("http://127.0.0.1:8000/upload", files=files)
                    if upload_res.status_code == 200:
                        st.sidebar.success(f"✅ Received file: {uploaded_file.name}")
                        st.session_state['auto_load'] = True
                    else:
                        st.sidebar.error(f"Upload failed: {upload_res.status_code}")
                except requests.exceptions.ConnectionError:
                    st.sidebar.error("Error: Could not connect to port 8000.")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 2. Control Panel")

if st.sidebar.button("🚀 Analyze Logs Now", use_container_width=True):
    with st.spinner("⚡ Running AI Threat Detection (Optimized Pipeline)..."):
        try:
            res = requests.post(
                "http://127.0.0.1:8000/analyze", 
                json={"max_docs": 100000, "retrain": True, "write_back": True}
            )
            if res.status_code == 200:
                st.sidebar.success("✅ Analysis completed successfully!")
                summary_data = res.json()
                st.session_state['summary_data'] = summary_data
                st.session_state['auto_load'] = True
            else:
                st.sidebar.error(f"Backend Error: {res.status_code}")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("Error: Could not connect to port 8000.")

if 'summary_data' in st.session_state:
    s = st.session_state['summary_data']
    st.sidebar.markdown(f"""
    <div class="summary-container">
        <div style="font-weight:700; color:#38bdf8; margin-bottom:8px;">📊 Analysis Executive Summary</div>
        <div class="summary-item"><span>Total Processed:</span><span class="badge-tag tag-blue">{s.get('total_logs', 0):,}</span></div>
        <div class="summary-item"><span>Anomalies Detected:</span><span class="badge-tag tag-amber">{s.get('anomalies', 0):,}</span></div>
        <div class="summary-item"><span>High Threats:</span><span class="badge-tag tag-red">{s.get('by_severity', {}).get('High', 0):,}</span></div>
        <div class="summary-item"><span>Duration:</span><span class="badge-tag tag-green">{s.get('duration_seconds', 0)}s</span></div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.write("")
if st.sidebar.button("🔄 Refresh Dashboard Data", use_container_width=True):
    st.session_state['auto_load'] = True

# ================= MAIN HERO HEADER =================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title"> AI Linux Threat Detection Dashboard</div>
    <div class="hero-subtitle">Real-time Anomaly Detection & Threat Classification Powered by Isolation Forest & Elasticsearch</div>
</div>
""", unsafe_allow_html=True)

# Fetch results from Backend API
@st.cache_data(ttl=10, show_spinner=False)
def fetch_results(limit=10000):
    try:
        response = requests.get(f"http://127.0.0.1:8000/results?limit={limit}")
        if response.status_code == 200:
            return response.json().get("items", [])
    except Exception:
        pass
    return []

# Only load and display data when user triggers analysis or refresh
if st.session_state.get('auto_load', False) or 'summary_data' in st.session_state:
    data = fetch_results(limit=5000)
    
    if data and len(data) > 0:
        df = pd.DataFrame(data)
        
        # Clean & Normalize columns
        if 'severity' in df.columns:
            df['severity'] = df['severity'].fillna('Normal').astype(str).str.capitalize()
        if 'threat_type' in df.columns:
            df['threat_type'] = df['threat_type'].fillna('Unknown / Normal')
        if 'status' in df.columns:
            df['status'] = df['status'].fillna('Normal').astype(str).str.capitalize()
        if 'anomaly_score' in df.columns:
            df['anomaly_score'] = pd.to_numeric(df['anomaly_score'], errors='coerce').fillna(0).round(4)
            
        # SORT ORDER: High -> Medium -> Low -> Normal
        severity_order = {'High': 0, 'Medium': 1, 'Low': 2, 'Normal': 3}
        if 'severity' in df.columns:
            df['sort_key'] = df['severity'].map(severity_order).fillna(4)
            df = df.sort_values(by=['sort_key', 'anomaly_score'], ascending=[True, True]).drop(columns=['sort_key'])
        
        # ================= PART 1: OVERVIEW & CHARTS =================
        st.header("📊 Threat Overview")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        total_logs = len(df)
        high_alerts = len(df[df['severity'] == 'High']) if 'severity' in df.columns else 0
        anomalies = len(df[df['status'] == 'Anomaly']) if 'status' in df.columns else 0
        
        col_m1.metric("Total Processed Logs", f"{total_logs:,}")
        col_m2.metric("Critical Threats (High)", f"{high_alerts:,}", delta="-Requires Attention", delta_color="inverse")
        col_m3.metric("Detected Anomalies", f"{anomalies:,}")
        
        st.markdown("---")
        
        chart_col1, chart_col2 = st.columns([1.2, 1])
        
        with chart_col1:
            import plotly.express as px
            st.subheader("🍩 Severity Distribution")
            
            filter_mode = st.radio(
                "View Mode:", 
                ["All Logs (Including Normal)", "Only Anomaly Threats"], 
                horizontal=True, 
                key="pie_chart_mode"
            )
            
            chart_df = df.copy()
            if filter_mode == "Only Anomaly Threats" and 'severity' in chart_df.columns:
                chart_df = chart_df[chart_df['severity'] != 'Normal']
            
            if not chart_df.empty and 'severity' in chart_df.columns:
                chart_data = chart_df['severity'].value_counts().reset_index()
                chart_data.columns = ['Severity', 'Count']
                color_map = {'High': '#EF4444', 'Medium': '#F59E0B', 'Low': '#3B82F6', 'Normal': '#10B981'}
                
                fig_pie = px.pie(
                    chart_data, values='Count', names='Severity', color='Severity',
                    color_discrete_map=color_map, hole=0.5
                )
                fig_pie.update_layout(
                    height=320,
                    margin=dict(t=20, b=30, l=0, r=0),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No anomaly threats found to display in pie chart.")

        with chart_col2:
            st.subheader("🎯 Top Threat Categories")
            if 'threat_type' in df.columns:
                anomaly_only = df[df['status'] == 'Anomaly']
                if not anomaly_only.empty:
                    threat_counts = anomaly_only['threat_type'].value_counts().reset_index()
                    threat_counts.columns = ['Threat Type', 'Occurrences']
                    
                    fig_bar = px.bar(
                        threat_counts.head(5), 
                        x='Occurrences', 
                        y='Threat Type', 
                        orientation='h',
                        color='Occurrences',
                        color_continuous_scale='Reds'
                    )
                    fig_bar.update_traces(width=0.4)
                    fig_bar.update_layout(
                        height=320,
                        margin=dict(t=20, b=20, l=0, r=0),
                        yaxis=dict(autorange="reversed"),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.success("🎉 No threat categories detected! System logs appear clean.")

        st.markdown("---")
        
        # ================= PART 2: DYNAMIC LOG TABLE =================
        st.header("📋 Detailed Log Analysis")
        
        filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns([1.5, 1.2, 1.2, 0.8, 1])
        
        with filter_col1:
            search_term = st.text_input("🔍 Search logs (IP, Message, etc.):", "")
        with filter_col2:
            severity_options = ["All"] + list(df['severity'].unique()) if 'severity' in df.columns else ["All"]
            selected_severity = st.selectbox("🎯 Filter by Severity:", severity_options)
        with filter_col3:
            threat_options = ["All"] + list(df['threat_type'].unique()) if 'threat_type' in df.columns else ["All"]
            selected_threat = st.selectbox("⚠️ Filter by Threat Type:", threat_options)
        with filter_col4:
            page_size = st.number_input("Rows:", min_value=10, max_value=total_logs, value=min(200, total_logs), step=50)
            
        filtered_df = df.copy()
        if selected_severity != "All" and 'severity' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['severity'] == selected_severity]
        if selected_threat != "All" and 'threat_type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['threat_type'] == selected_threat]
            
        if search_term:
            mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
            filtered_df = filtered_df[mask]
        
        # Reset row index starting from 1 cleanly
        filtered_df = filtered_df.reset_index(drop=True)
        filtered_df.index = filtered_df.index + 1
        
        display_df = filtered_df.head(page_size)
        
        st.caption(f"Showing **{len(display_df):,}** of **{len(filtered_df):,}** matching records (Total in DB: {total_logs:,}).")
        
        st.dataframe(
            display_df, 
            use_container_width=True, 
            height=480,
            column_config={
                "anomaly_score": st.column_config.NumberColumn("Anomaly Score", format="%.4f"),
                "status": st.column_config.TextColumn("Status"),
                "severity": st.column_config.TextColumn("Severity"),
            }
        )
        
        with filter_col5:
            st.write("")
            st.write("")
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Export CSV",
                data=csv,
                file_name='threat_analysis_report.csv',
                mime='text/csv',
                use_container_width=True
            )
    else:
        st.info("👋 Log file successfully uploaded to database. Please click the **🚀 Analyze Logs Now** button in the left sidebar to run AI analysis and view results.")
else:
    st.info("👋 Welcome! Please upload a log file in the sidebar, then click the **🚀 Analyze Logs Now** button to begin analysis.")
