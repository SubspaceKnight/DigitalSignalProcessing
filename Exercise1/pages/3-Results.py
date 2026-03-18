import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import helper 

#Page config 
st.set_page_config(page_title="Results", layout="wide")
st.title("Results")

#Load data 
@st.cache_data
def load():
    df = helper.load_raw()
    return helper.get_driver(df)

driver_df = load()

#Quick sanity check - show what we have
with st.expander("Raw data preview", expanded=False):
    st.dataframe(driver_df.head(200), use_container_width=True)
    st.write(f"**Shape:** {driver_df.shape}  |  **Columns:** {list(driver_df.columns)}")
    st.write(f"**Laps found:** {helper.list_laps(driver_df)}")

#Sidebar controls 
st.sidebar.header("Plot settings")
available_signals = [c for c in helper.TELEMETRY_COLS if c in driver_df.columns]
signal = st.sidebar.selectbox("Telemetry variable", available_signals, index=0)
show_outliers = st.sidebar.checkbox("Highlight outlier laps", value=True)
outlier_laps = helper.flag_outlier_laps(driver_df)
#z_thresh = st.sidebar.slider("Outlier Z-score threshold", min_value=0.0, max_value=5.0, value=2.0, step=0.1)


#Detect outlier laps
outlier_laps = helper.flag_outlier_laps(driver_df, signal="Speed") #, z_threshold=z_thresh
all_laps     = helper.list_laps(driver_df)
normal_laps  = [l for l in all_laps if l not in outlier_laps]

st.sidebar.markdown(
    f"**Laps flagged as outliers:** {outlier_laps if outlier_laps else 'None'}"
)

#Tabs 
tab1, tab2, tab3, tab4 = st.tabs(
    ["Lap overlay", "Per-lap statistics", "Track map (X/Y)", "Speed & brake zones"]
)

#TAB 1: Lap overlay 
with tab1:
    st.subheader(f"{signal} - all laps overlaid (0-100% lap completion)")

    mat, valid_laps = helper.build_lap_matrix(driver_df, signal)
    x = np.linspace(0, 100, 500)

    fig = go.Figure()

    #Normal laps
    for i, lap in enumerate(valid_laps):
        if lap in outlier_laps and show_outliers:
            color = "rgba(255,80,80,0.25)"
        elif lap in outlier_laps:
            continue   #don't draw if checkbox is off
        else:
            color = "rgba(100,149,237,0.12)"

        fig.add_trace(go.Scatter(
            x=x, y=mat[i],
            mode="lines",
            line=dict(color=color, width=1),
            name=f"Lap {lap}",
            hovertemplate=f"Lap {lap}<br>%{{x:.1f}}%: %{{y:.1f}}<extra></extra>",
            showlegend=False,
        ))

    #Mean +/- std band
    normal_idx = [i for i, l in enumerate(valid_laps) if l not in outlier_laps]
    if normal_idx:
        normal_mat = mat[normal_idx]
        mean_y = normal_mat.mean(axis=0)
        std_y  = normal_mat.std(axis=0)

        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([mean_y + std_y, (mean_y - std_y)[::-1]]),
            fill="toself",
            fillcolor="rgba(100,149,237,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±1 σ band",
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=x, y=mean_y,
            mode="lines",
            line=dict(color="cornflowerblue", width=2.5),
            name="Mean (clean laps)",
        ))

    fig.update_layout(
        xaxis_title="Lap completion (%)",
        yaxis_title=signal,
        legend=dict(orientation="h", y=1.02),
        height=480,
        margin=dict(l=60, r=20, t=40, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Blue traces = clean laps · Red traces = flagged outlier laps · "
        "Bold line = mean of clean laps · Shaded band = ±1 standard deviation"
    )

#TAB 2: Per-lap stats 
with tab2:
    st.subheader(f"{signal} - statistics per lap (trend over the stint)")
    summary = helper.lap_summary(driver_df, signal)
    summary["outlier"] = summary["Lap"].isin(outlier_laps)

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Mean per lap", "Std per lap"])

    colors = ["red" if o else "cornflowerblue" for o in summary["outlier"]]

    fig2.add_trace(go.Bar(
        x=summary["Lap"], y=summary["mean"],
        marker_color=colors, name="Mean",
        hovertemplate="Lap %{x}: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig2.add_trace(go.Bar(
        x=summary["Lap"], y=summary["std"],
        marker_color=colors, name="Std dev",
        hovertemplate="Lap %{x}: %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    fig2.update_layout(height=500, showlegend=False,
                       margin=dict(l=60, r=20, t=60, b=60))
    fig2.update_xaxes(title_text="Lap number", row=2, col=1)
    fig2.update_yaxes(title_text=signal, row=1, col=1)
    fig2.update_yaxes(title_text="Std dev", row=2, col=1)

    st.plotly_chart(fig2, use_container_width=True)

    st.write("**Coefficient of Variation (CV = std/mean × 100)**")
    cv = (summary["std"] / summary["mean"].abs() * 100).round(2)
    cv_df = pd.concat([summary[["Lap", "mean", "std"]], cv.rename("CV (%)")], axis=1)
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

#TAB 3: Track map 
with tab3:
    st.subheader("Bahrain track layout - colored by speed")
    if "X" in driver_df.columns and "Y" in driver_df.columns:
        lap_choice = st.selectbox("Select lap for track map", all_laps, index=0)
        lap_data   = helper.get_lap(driver_df, lap_choice)

        color_col = st.selectbox("Color by", available_signals, index=0)

        fig3 = go.Figure(go.Scatter(
            x=lap_data["X"], y=lap_data["Y"],
            mode="markers",
            marker=dict(
                color=lap_data[color_col],
                colorscale="RdYlGn",
                size=3,
                colorbar=dict(title=color_col),
            ),
            hovertemplate=f"X: %{{x:.0f}}<br>Y: %{{y:.0f}}<br>{color_col}: %{{marker.color:.1f}}<extra></extra>",
        ))
        fig3.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            height=550,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("X/Y columns not found in the dataset — track map unavailable.")


#TAB 4: speed+brake plot
with tab4:
    from analysis import get_brake_section_summary

    st.subheader("Speed profile with braking zones")

    lap_select = st.selectbox("Select lap", helper.list_laps(driver_df), index=0)
    lap_data   = helper.get_lap(driver_df, lap_select).copy()
    lap_data["time_in_lap_s"] = lap_data[helper.COL_TIME].dt.total_seconds()
    lap_data   = lap_data.dropna(subset=["time_in_lap_s", "Speed"]).sort_values("time_in_lap_s")

    fig_brake = go.Figure()

    #Red shaded brake zones
    x_arr = lap_data["time_in_lap_s"].values
    b_arr = lap_data["Brake_active"].values
    in_seg, start = False, None

    for i in range(len(x_arr)):
        if b_arr[i] == 1 and not in_seg:
            start, in_seg = x_arr[i], True
        elif b_arr[i] == 0 and in_seg:
            fig_brake.add_vrect(
                x0=start, x1=x_arr[i],
                fillcolor="red", opacity=0.12, line_width=0,
            )
            in_seg = False
    if in_seg:
        fig_brake.add_vrect(x0=start, x1=x_arr[-1],
                            fillcolor="red", opacity=0.12, line_width=0)

    #Speed trace
    fig_brake.add_trace(go.Scatter(
        x=x_arr,
        y=lap_data["Speed"].values,
        mode="lines",
        line=dict(color="cornflowerblue", width=2),
        name="Speed",
    ))

    fig_brake.update_layout(
        xaxis_title="Time within lap (s)",
        yaxis_title="Speed (km/h)",
        title=f"Lap {lap_select} — red zones = braking",
        height=380,
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(fig_brake, use_container_width=True)

    st.markdown("**Braking zones detail**")
    brake_table = get_brake_section_summary(lap_data)
    if brake_table.empty:
        st.info("No braking zones detected in this lap.")
    else:
        st.dataframe(brake_table, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total zones",       len(brake_table))
        c2.metric("Total brake time",  f"{brake_table['Duration (s)'].sum():.2f} s")
        c3.metric("Hardest stop",      f"{brake_table['Speed drop (km/h)'].max():.1f} km/h drop")