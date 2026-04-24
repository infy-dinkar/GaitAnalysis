"""
biomech_flow.py
Biomechanical Analysis flow for GaitVision (shoulder + neck range-of-motion).

Sub-flow tracked by st.session_state["biomech_step"]:
    patient   → body_part → movement → mode → capture → report

This module is reached only when st.session_state["app_mode"] == "biomech".
The render entry-point is `render_biomech_flow()`, called from app.py's
top-level dispatcher.

Commit 1 (this commit): stub only — patient-details onwards is implemented
in commits 3-6.
"""
from __future__ import annotations

import streamlit as st


def render_biomech_flow() -> None:
    """Top-level renderer for the biomech sub-flow.

    Stub implementation for Commit 1. Subsequent commits flesh out each
    biomech_step branch:
      Commit 2 — shoulder_engine + neck_engine modules (no UI here yet)
      Commit 3 — patient form, body-part chooser, movement chooser
      Commit 4 — video-upload capture + analysis
      Commit 6 — report page + PDF export
    """
    st.markdown(
        '<h2 style="text-align:center; margin-top:32px;">'
        'Biomechanical Analysis</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center; color:#94A3B8; margin-bottom:32px;">'
        'Range-of-motion assessment for shoulder and neck. UI under '
        'construction — landing in commits 2-6 of this feature.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.info(
        f"Current biomech step: **{st.session_state.get('biomech_step', 'patient')}**. "
        "This screen is a placeholder — the full wizard is being built up "
        "incrementally."
    )
