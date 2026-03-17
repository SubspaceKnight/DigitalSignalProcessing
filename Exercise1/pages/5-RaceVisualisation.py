import helper
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from analysis import get_clean_laps
from helper import COL_LAP

st.set_page_config(
    page_title="Race Visualisation",
    page_icon="🏎️",
    layout="wide",
)
st.title("Race Visualisation - Verstappen, Bahrain 2024")

#Load 
@st.cache_data
def load():
    df = helper.load_raw()
    return helper.get_driver(df)

driver_df  = load()
clean_laps = get_clean_laps(driver_df)

#Sidebar 
with st.sidebar:
    st.header("Settings")
    lap_choice   = st.selectbox("Lap", clean_laps, index=4)
    color_signal = st.selectbox(
        "Color track by",
        ["Speed", "Throttle", "Brake", "nGear", "RPM"],
    )
    trail_length = st.slider("Trail length (samples)", 10, 150, 60)

#Data for selected lap 
lap_df  = helper.get_lap(driver_df, lap_choice).reset_index(drop=True)
n       = len(lap_df)

#Full track outline from all clean laps combined
track_outline = (
    driver_df[driver_df[COL_LAP].isin(clean_laps)][["X", "Y"]]
    .dropna()
)

#Tabs 
tab_replay, tab_map, tab_compare = st.tabs(
    ["Lap replay", "Full lap map", "Lap comparison"]
)

#TAB 1 = Lap replay with scrubber
with tab_replay:
    st.markdown(
        "Drag the **position slider** to scrub through the lap. "
        "The marker moves on the track and all telemetry readouts update live."
    )

    #Position slider
    pos = st.slider(
        "Position in lap (sample index)",
        min_value=0,
        max_value=n - 1,
        value=0,
        step=1,
        label_visibility="collapsed",
    )

    #Current sample values
    row          = lap_df.iloc[pos]
    trail_start  = max(0, pos - trail_length)
    trail        = lap_df.iloc[trail_start : pos + 1]

    #Live metric strip 
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Speed",    f"{row['Speed']:.0f} km/h")
    c2.metric("Throttle", f"{row['Throttle']:.0f} %")
    c3.metric("Brake",    f"{row['Brake']:.2f}")
    c4.metric("Gear",     f"{int(row['nGear'])}")
    c5.metric("RPM",      f"{row['RPM']:.0f}")

    #Main layout: track map (left) + telemetry stack (right) 
    left, right = st.columns([1, 1])

    with left:
        col_vals = lap_df[color_signal].values
        vmin, vmax = col_vals.min(), col_vals.max()

        fig_track = go.Figure()

        #Track outline (all clean laps, faint)
        fig_track.add_trace(go.Scatter(
            x=track_outline["X"],
            y=track_outline["Y"],
            mode="markers",
            marker=dict(size=1, color="rgba(180,180,180,0.15)"),
            hoverinfo="skip",
            showlegend=False,
        ))

        #Driven trail colored by signal
        fig_track.add_trace(go.Scatter(
            x=trail["X"],
            y=trail["Y"],
            mode="markers",
            marker=dict(
                size=4,
                color=trail[color_signal].values,
                colorscale="RdYlGn",
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(
                    title=color_signal,
                    thickness=12,
                    len=0.6,
                ),
                showscale=True,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

        #Not-yet-driven part (ghost)
        future = lap_df.iloc[pos:]
        fig_track.add_trace(go.Scatter(
            x=future["X"],
            y=future["Y"],
            mode="markers",
            marker=dict(size=2, color="rgba(120,120,120,0.2)"),
            hoverinfo="skip",
            showlegend=False,
        ))

        #Car marker
        fig_track.add_trace(go.Scatter(
            x=[row["X"]],
            y=[row["Y"]],
            mode="markers+text",
            marker=dict(
                size=14,
                color="red",
                symbol="circle",
                line=dict(color="white", width=2),
            ),
            text=["VER"],
            textposition="top center",
            textfont=dict(size=11, color="white"),
            hovertemplate=(
                f"Speed: {row['Speed']:.0f} km/h<br>"
                f"Throttle: {row['Throttle']:.0f}%<br>"
                f"Gear: {int(row['nGear'])}<extra></extra>"
            ),
            showlegend=False,
        ))

        fig_track.update_layout(
            xaxis=dict(
                scaleanchor="y", scaleratio=1,
                showgrid=False, zeroline=False, showticklabels=False,
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=480,
            title=dict(text=f"Track position — Lap {lap_choice}", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_track, use_container_width=True)

    with right:
        #Telemetry stack: Speed / Throttle+Brake / Gear
        x_axis    = np.arange(n)
        x_current = pos

        fig_telem = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=("Speed (km/h)", "Throttle & Brake (%)", "Gear"),
            vertical_spacing=0.08,
        )

        def add_signal_trace(fig, signal, row_idx, color, name=None):
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=lap_df[signal].values,
                mode="lines",
                line=dict(color=color, width=1.5),
                name=name or signal,
                hovertemplate=f"{signal}: %{{y:.1f}}<extra></extra>",
            ), row=row_idx, col=1)

        add_signal_trace(fig_telem, "Speed",    1, "cornflowerblue")
        add_signal_trace(fig_telem, "Throttle", 2, "mediumseagreen", "Throttle")
        add_signal_trace(fig_telem, "Brake",    2, "tomato",         "Brake (×100)")

        #Scale brake to same range as throttle for visual clarity
        fig_telem.add_trace(go.Scatter(
            x=x_axis,
            y=lap_df["Brake"].values * 100,
            mode="lines",
            line=dict(color="tomato", width=1.5),
            name="Brake ×100",
            showlegend=True,
        ), row=2, col=1)

        add_signal_trace(fig_telem, "nGear", 3, "plum", "Gear")

        #Vertical line = current position (one per subplot)
        for r in [1, 2, 3]:
            fig_telem.add_vline(
                x=x_current,
                line=dict(color="red", width=1.5, dash="dash"),
                row=r, col=1,
            )

        fig_telem.update_layout(
            height=480,
            showlegend=True,
            legend=dict(orientation="h", y=1.02, font=dict(size=11)),
            margin=dict(l=60, r=20, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        fig_telem.update_xaxes(
            title_text="Sample index", row=3, col=1,
            showgrid=True, gridcolor="rgba(128,128,128,0.15)",
        )
        for r in [1, 2, 3]:
            fig_telem.update_yaxes(
                showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                row=r, col=1,
            )

        st.plotly_chart(fig_telem, use_container_width=True)


#TAB 2 — Full lap map colored by signal
with tab_map:
    st.markdown(
        f"Full lap **{lap_choice}** plotted on the Bahrain circuit, "
        f"colored by **{color_signal}**. "
        "Use the sidebar to change lap or signal."
    )

    fig_map = go.Figure()

    #Track outline
    fig_map.add_trace(go.Scatter(
        x=track_outline["X"],
        y=track_outline["Y"],
        mode="markers",
        marker=dict(size=1.5, color="rgba(180,180,180,0.2)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    #Full lap colored by signal
    fig_map.add_trace(go.Scatter(
        x=lap_df["X"],
        y=lap_df["Y"],
        mode="markers",
        marker=dict(
            size=5,
            color=lap_df[color_signal].values,
            colorscale="RdYlGn",
            colorbar=dict(title=color_signal, thickness=14),
            showscale=True,
        ),
        hovertemplate=(
            f"{color_signal}: %{{marker.color:.1f}}<br>"
            "X: %{x:.0f}  Y: %{y:.0f}<extra></extra>"
        ),
        showlegend=False,
    ))

    #Mark start/finish
    fig_map.add_trace(go.Scatter(
        x=[lap_df["X"].iloc[0]],
        y=[lap_df["Y"].iloc[0]],
        mode="markers+text",
        marker=dict(size=14, color="gold", symbol="star",
                    line=dict(color="black", width=1)),
        text=["S/F"],
        textposition="top center",
        textfont=dict(size=11),
        name="Start / Finish",
    ))

    fig_map.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1,
                   showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text=f"Bahrain circuit — Lap {lap_choice} colored by {color_signal}",
            x=0.5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_map, use_container_width=True)


#TAB 3 — Lap-to-lap comparison on track
with tab_compare:
    st.markdown(
        "Compare how **the same track position** looks across different laps. "
        "Differences in color intensity reveal where Verstappen was quicker "
        "or slower lap-to-lap — no frequency analysis needed."
    )

    compare_laps = st.multiselect(
        "Select laps to compare (max 4)",
        clean_laps,
        default=clean_laps[4:8],
        max_selections=4,
    )

    if compare_laps:
        n_laps   = len(compare_laps)
        cols     = st.columns(n_laps)

        for col, lap in zip(cols, compare_laps):
            ldf = helper.get_lap(driver_df, lap)
            sig_vals = ldf[color_signal].values
            vmin_l, vmax_l = sig_vals.min(), sig_vals.max()

            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=track_outline["X"],
                y=track_outline["Y"],
                mode="markers",
                marker=dict(size=1, color="rgba(180,180,180,0.15)"),
                hoverinfo="skip",
                showlegend=False,
            ))
            fig_c.add_trace(go.Scatter(
                x=ldf["X"],
                y=ldf["Y"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=sig_vals,
                    colorscale="RdYlGn",
                    cmin=vmin_l,
                    cmax=vmax_l,
                    showscale=False,
                ),
                hovertemplate=f"{color_signal}: %{{marker.color:.1f}}<extra></extra>",
                showlegend=False,
            ))
            fig_c.update_layout(
                xaxis=dict(scaleanchor="y", scaleratio=1,
                           showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text=f"Lap {lap}", x=0.5, font=dict(size=13)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            col.plotly_chart(fig_c, use_container_width=True)

            #Summary metric under each map
            col.metric(
                f"Mean {color_signal}",
                f"{sig_vals.mean():.1f}",
                delta=f"{sig_vals.mean() - lap_df[color_signal].mean():.1f} vs lap {lap_choice}",
            )
    else:
        st.info("Select at least one lap from the list above.")