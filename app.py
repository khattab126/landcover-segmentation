"""Land Cover Segmentation — Streamlit Web App."""

import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from model import load_model, predict_image, colorize, CLASS_NAMES, PALETTE
from results.experiment_data import (
    FACTOR1_HISTORY, FACTOR1_SUMMARY,
    FACTOR2_HISTORY, FACTOR2_SUMMARY,
    FACTOR2_PER_CLASS_IOU,
)

st.set_page_config(page_title="Land Cover Segmentation", page_icon=":earth_americas:", layout="wide")

st.title("Land Cover Segmentation with Partial Focal CE Loss")
st.markdown(
    "A U-Net trained on the **DeepGlobe Land Cover** dataset using sparse point labels "
    "and a partial focal cross-entropy loss. Upload a satellite image or browse the results below."
)

# ---------------------------------------------------------------------------
# Cached model loader — loads once per session
# ---------------------------------------------------------------------------

@st.cache_resource
def get_model():
    return load_model("model/best_model.pth")

# ---------------------------------------------------------------------------
# Helper: class legend
# ---------------------------------------------------------------------------

def show_legend():
    cols = st.columns(len(CLASS_NAMES))
    for i, (name, col) in enumerate(zip(CLASS_NAMES, cols)):
        r, g, b = PALETTE[i]
        color_hex = f"#{r:02x}{g:02x}{b:02x}"
        col.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<span style="width:18px;height:18px;border-radius:3px;'
            f'background:{color_hex};display:inline-block;border:1px solid #555;"></span>'
            f'<span style="font-size:14px;">{name.title()}</span></div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_predict, tab_experiments, tab_gallery = st.tabs([
    "\U0001f50d Predict", "\U0001f4ca Experiments", "\U0001f5bc Gallery"
])

# ===========================================================================
# TAB 1 — Predict
# ===========================================================================

with tab_predict:
    st.header("Upload a Satellite Image")
    st.write("Upload a satellite or aerial image to see land cover segmentation predictions.")

    uploaded = st.file_uploader(
        "Choose an image...", type=["png", "jpg", "jpeg"], key="uploader"
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded)
        with st.spinner("Running prediction..."):
            model = get_model()
            pred_rgb = predict_image(model, pil_img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(pred_rgb, caption="Land Cover Prediction", use_container_width=True)

        st.subheader("Class Legend")
        show_legend()
    else:
        st.info("Upload a PNG or JPEG image above to get started.")

# ===========================================================================
# TAB 2 — Experiments
# ===========================================================================

with tab_experiments:
    # --- Factor 1 --------------------------------------------------------
    st.header("Experiment Results")

    st.subheader("Factor 1: Annotation Budget Sweep")
    st.write(
        "How many labeled points per class are needed? We sweep `points_per_class` "
        "from 1 to 50 with balanced sampling and focal loss (gamma=2), plus a full-supervision baseline."
    )

    st.dataframe(FACTOR1_SUMMARY, use_container_width=True, hide_index=True)

    # Factor 1 line chart: mIoU vs points-per-class
    ppc_vals = [1, 5, 10, 20, 50]
    best_mious = [s["best_mIoU"] for s in FACTOR1_SUMMARY[:5]]
    full_miou = FACTOR1_SUMMARY[5]["best_mIoU"]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=ppc_vals, y=best_mious, mode="lines+markers",
        name="Partial pfCE (best epoch)",
        marker=dict(size=10), line=dict(width=2),
    ))
    fig1.add_hline(
        y=full_miou, line_dash="dash", line_color="red",
        annotation_text=f"Full supervision ({full_miou:.3f})",
    )
    fig1.update_layout(
        xaxis_title="Points per class (log scale)",
        yaxis_title="Best-epoch mIoU",
        title="mIoU vs Annotation Budget",
        xaxis_type="log",
        height=420,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Factor 2 --------------------------------------------------------
    st.subheader("Factor 2: Loss x Sampling Strategy")
    st.write(
        "At a fixed budget of 100 points/image, we compare plain CE vs focal CE "
        "with balanced vs uniform sampling."
    )

    st.dataframe(FACTOR2_SUMMARY, use_container_width=True, hide_index=True)

    fig2 = go.Figure()
    names = [s["config"] for s in FACTOR2_SUMMARY]
    mious = [s["best_mIoU"] for s in FACTOR2_SUMMARY]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    fig2.add_trace(go.Bar(x=names, y=mious, marker_color=colors, text=[
        f"{v:.3f}" for v in mious
    ], textposition="outside"))
    fig2.update_layout(
        yaxis_title="Best-epoch mIoU",
        title="Best mIoU by Loss x Sampling",
        height=420, yaxis=dict(rangemode="tozero"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Per-class IoU grouped bar chart
    st.subheader("Per-Class IoU Comparison")
    fig3 = go.Figure()
    for i, (cfg_name, class_ious) in enumerate(FACTOR2_PER_CLASS_IOU.items()):
        fig3.add_trace(go.Bar(
            name=cfg_name,
            x=CLASS_NAMES,
            y=[class_ious[c] for c in CLASS_NAMES],
            marker_color=colors[i],
        ))
    fig3.update_layout(
        barmode="group",
        yaxis_title="IoU",
        title="Per-Class IoU — Factor 2",
        height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Training curves (collapsible)
    with st.expander("Training Curves — All Runs"):
        fig4 = go.Figure()
        all_configs = {**FACTOR1_HISTORY, **FACTOR2_HISTORY}
        palette_lines = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
            "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
            "#FF97FF", "#FECB52",
        ]
        for i, (cfg_name, history) in enumerate(all_configs.items()):
            fig4.add_trace(go.Scatter(
                x=[h["epoch"] for h in history],
                y=[h["mIoU"] for h in history],
                mode="lines+markers",
                name=cfg_name,
                line=dict(color=palette_lines[i % len(palette_lines)]),
            ))
        fig4.update_layout(
            xaxis_title="Epoch", yaxis_title="Validation mIoU",
            title="Training Curves — All Configurations",
            height=500,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Key findings
    st.subheader("Key Findings")
    st.markdown("""
    1. **Diminishing returns on annotation budget.** Best-epoch mIoU rises monotonically with
       points/class but flattens above ~20 points/class. Roughly 20 balanced clicks per class
       capture most of the achievable performance.
    2. **Imbalance corrections substitute, they don't stack.** Either balanced sampling or focal
       modulation alone reaches ~0.30 mIoU; combining them yields no additional gain.
    3. **Focal CE is the right default for realistic (uniform) annotation.** It rescues rare-class
       IoU (e.g. water) at zero extra annotation cost.
    4. **Best-epoch tracking is essential** when the training budget is small. Final-epoch metrics
       hid the monotonic budget trend and the true configuration ordering.
    """)

# ===========================================================================
# TAB 3 — Gallery (full 300-sample dataset)
# ===========================================================================

FULL_IDS = [f"{i:04d}" for i in range(300)]
PER_PAGE = 6

with tab_gallery:
    st.header("Full DeepGlobe Dataset (300 Samples)")
    st.write(
        f"Pre-computed predictions on all 300 satellite images from the DeepGlobe Land Cover dataset, "
        f"using the best model (pfCE, 50 points/class, best-epoch mIoU = 0.300)."
    )

    # Navigation controls
    total_pages = (len(FULL_IDS) + PER_PAGE - 1) // PER_PAGE
    nav_cols = st.columns([1, 3, 1])
    with nav_cols[0]:
        if st.button("Previous", key="prev_btn"):
            st.session_state.page = max(0, st.session_state.get("page", 0) - 1)
    with nav_cols[1]:
        page = st.session_state.get("page", 0)
        page = st.number_input(
            f"Page (1-{total_pages})", min_value=1, max_value=total_pages,
            value=page + 1, key="page_input",
        )
        st.session_state.page = page - 1
    with nav_cols[2]:
        if st.button("Next", key="next_btn"):
            st.session_state.page = min(total_pages - 1, st.session_state.get("page", 0) + 1)

    page = st.session_state.get("page", 0)
    start = page * PER_PAGE
    end = min(start + PER_PAGE, len(FULL_IDS))
    st.caption(f"Showing samples {start + 1}–{end} of {len(FULL_IDS)}")

    # Display grid: PER_PAGE/2 rows x 2 columns, each with 3 sub-columns
    page_ids = FULL_IDS[start:end]
    for i in range(0, len(page_ids), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(page_ids):
                did = page_ids[idx]
                sample_num = int(did)
                c1, c2, c3 = col.columns(3)
                c1.image(f"full_dataset/images/{did}.jpg", caption=f"Sample {sample_num}", use_container_width=True)
                c2.image(f"full_dataset/masks/{did}.png", caption="Ground Truth", use_container_width=True)
                c3.image(f"full_dataset/predictions/{did}.png", caption="Prediction", use_container_width=True)

    st.subheader("Class Legend")
    show_legend()
