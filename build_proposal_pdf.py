# -*- coding: utf-8 -*-
"""Generate the research proposal PDF (Bishwadip Maitra) in the project root."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    ListFlowable, ListItem, HRFlowable, KeepTogether,
)

OUT = r"d:\GNN_TL\Research_Proposal_Bishwadip_Maitra.pdf"

NAVY = colors.HexColor("#1a3a5c")
ACCENT = colors.HexColor("#2c6e91")
LIGHT = colors.HexColor("#eef3f7")
GREY = colors.HexColor("#444444")

styles = getSampleStyleSheet()

def S(name, **kw):
    styles.add(ParagraphStyle(name, **kw))

S("TitleX", parent=styles["Title"], fontName="Helvetica-Bold",
  fontSize=18, leading=23, textColor=NAVY, spaceAfter=10, alignment=TA_CENTER)
S("Author", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
  textColor=GREY, alignment=TA_CENTER, spaceAfter=1)
S("AuthorSub", fontName="Helvetica", fontSize=9.5, leading=13,
  textColor=GREY, alignment=TA_CENTER, spaceAfter=1)
S("H1", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=NAVY,
  spaceBefore=14, spaceAfter=6)
S("H2", fontName="Helvetica-Bold", fontSize=10.8, leading=14, textColor=ACCENT,
  spaceBefore=9, spaceAfter=3)
S("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10,
  leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6, textColor=colors.black)
S("BodyT", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.6,
  leading=13.5, alignment=TA_JUSTIFY, spaceAfter=3, textColor=colors.black)
S("Blt", parent=styles["Body"], fontName="Helvetica", fontSize=10,
  leading=14, spaceAfter=3)
S("MapNote", fontName="Helvetica-Oblique", fontSize=9.2, leading=12.5,
  textColor=ACCENT, spaceBefore=2, spaceAfter=8, leftIndent=10)
S("Ref", fontName="Helvetica", fontSize=9.2, leading=12.8, alignment=TA_LEFT,
  spaceAfter=4, leftIndent=16, firstLineIndent=-16, textColor=colors.black)
S("TblHead", fontName="Helvetica-Bold", fontSize=9, leading=11.5,
  textColor=colors.white)
S("TblCell", fontName="Helvetica", fontSize=8.7, leading=11.2,
  textColor=colors.black)

def b(t):
    return Paragraph(t, styles["Body"])

def bullets(items, style="Blt"):
    return ListFlowable(
        [ListItem(Paragraph(t, styles[style]), leftIndent=12, value="•")
         for t in items],
        bulletType="bullet", start="•", leftIndent=14, bulletColor=ACCENT,
        spaceBefore=0, spaceAfter=4,
    )

def rule():
    return HRFlowable(width="100%", thickness=0.6, color=ACCENT,
                      spaceBefore=4, spaceAfter=10)

story = []

# ---- Title block ----
story.append(Paragraph(
    "Weather-Regime-Conditioned Forecasting of Compound "
    "Air-Pollution Extremes over Bangladesh", styles["TitleX"]))
story.append(Spacer(1, 4))
story.append(Paragraph("Bishwadip Maitra", styles["Author"]))
story.append(Paragraph("Research Assistant, UW–BUET Air Quality Project, Bangladesh",
                       styles["AuthorSub"]))
story.append(Paragraph("bishwadip.m21@iiits.in", styles["AuthorSub"]))
story.append(rule())

# ---- 1. Problem Statement ----
story.append(Paragraph("1. Problem Statement", styles["H1"]))
story.append(b(
    "Bangladesh, and Dhaka in particular, experiences some of the highest "
    "particulate-matter concentrations recorded anywhere on Earth. During the dry "
    "winter months (November–February), 24-hour mean PM<sub>2.5</sub> routinely "
    "exceeds 150–250 µg/m³ — an order of magnitude above the WHO "
    "24-hour guideline of 15 µg/m³. These episodes are not driven by emissions "
    "alone: they are produced by the <i>interaction</i> of persistent local sources "
    "(brick kilns, road dust, biomass burning, traffic) with a shallow, stagnant winter "
    "boundary layer, weak surface winds, nocturnal temperature inversions, and "
    "transboundary transport from the wider Indo-Gangetic Plain (IGP)."))
story.append(b(
    "Critically, particulate pollution does not occur in isolation. The same stagnant, "
    "low-ventilation meteorological conditions that trap PM<sub>2.5</sub> also modulate "
    "photochemical processing of ozone (O<sub>3</sub>) and the accumulation of nitrogen "
    "oxides (NOx). Days on which <b>PM<sub>2.5</sub> and O<sub>3</sub> are simultaneously "
    "elevated — compound pollution extremes —</b> impose a substantially greater "
    "health burden than either pollutant alone, yet operational air-quality forecasting in "
    "Bangladesh is effectively non-existent and, where forecasting exists regionally, it "
    "treats pollutants one at a time."))
story.append(b("The core problem this proposal addresses is therefore twofold:"))
story.append(bullets([
    "We do not currently understand <b>which meteorological regimes</b> generate compound "
    "PM<sub>2.5</sub>–O<sub>3</sub> extremes over Bangladesh; and",
    "We have <b>no predictive framework</b> that exploits those regimes to forecast such "
    "extremes with enough lead time to be useful for public-health warnings.",
]))

# ---- 2. Research Gap ----
story.append(Paragraph("2. Research Gap", styles["H1"]))
story.append(b(
    "The compound-extreme and weather-regime literature has matured rapidly, but it has a "
    "clear geographic blind spot:"))
story.append(bullets([
    "<b>The regions are wrong.</b> Weather-regime-conditioned compound "
    "PM<sub>2.5</sub>–O<sub>3</sub> studies exist for eastern and southern China, the "
    "western United States, and the <i>western</i> IGP (Delhi and upwind Punjab–Haryana). "
    "<b>Bangladesh and the eastern IGP terminus are essentially unstudied</b> for compound "
    "extremes, despite sitting at the receiving end of the most polluted air mass on the planet.",
    "<b>The framing is single-pollutant.</b> Existing forecasting and meteorological-"
    "normalisation work in South Asia overwhelmingly targets PM<sub>2.5</sub> in isolation. "
    "The <i>co-occurrence</i> structure — how often, and under which synoptic conditions, "
    "PM<sub>2.5</sub> and O<sub>3</sub> peak together — has not been quantified for Bangladesh.",
    "<b>Stagnation indices are not localised.</b> Standard atmospheric-stagnation indices were "
    "calibrated for mid-latitude continental conditions and do not capture the humid, "
    "monsoon-modulated, low-wind regime characteristic of the Bengal delta. A region-specific "
    "stagnation/ventilation index for Bangladesh has not been constructed and validated against "
    "observed pollution episodes.",
]))
story.append(b(
    "This proposal closes that gap: it is, to my knowledge, the first weather-regime-conditioned "
    "analysis and forecast of <b>compound air-pollution extremes for Bangladesh</b>, built on "
    "locally collected multi-pollutant and meteorological observations."))

# ---- 3. Research Questions & Objectives ----
story.append(Paragraph("3. Research Questions and Objectives", styles["H1"]))
story.append(b(
    "<b>RQ1.</b> How frequently do compound PM<sub>2.5</sub>–O<sub>3</sub> (and "
    "PM<sub>2.5</sub>–NOx) extremes occur over Bangladesh, and how is this co-occurrence "
    "distributed across seasons and times of day?"))
story.append(b(
    "<b>RQ2.</b> Which distinct meteorological regimes — defined by wind, boundary-layer "
    "stability, humidity, and ventilation — are associated with compound extremes, and how "
    "much do these regimes amplify pollution relative to emissions alone?"))
story.append(b(
    "<b>RQ3.</b> Can a regime-aware machine-learning model forecast compound extremes at "
    "24–72 h lead time more skilfully than single-pollutant baselines?"))
story.append(Paragraph("Objectives", styles["H2"]))
story.append(bullets([
    "<b>O1:</b> Build a clean, gap-filled, quality-controlled multi-pollutant + meteorological "
    "dataset for the study domain.",
    "<b>O2:</b> Define and validate a <b>Bangladesh-specific ventilation/stagnation index</b>.",
    "<b>O3:</b> Identify recurrent weather regimes via unsupervised clustering and characterise "
    "their compound-extreme signatures.",
    "<b>O4:</b> Quantify the meteorology-driven vs. emission-driven contribution to extremes via "
    "meteorological normalisation.",
    "<b>O5:</b> Develop and benchmark a regime-conditioned probabilistic forecast model for "
    "compound extremes.",
]))

# ---- 4. Methodology ----
story.append(Paragraph("4. Methodology (with literature mapping)", styles["H1"]))
story.append(b(
    "The study proceeds in four phases. Each step names the methodological precedent it follows, "
    "so every analytical choice is anchored to published practice."))

story.append(Paragraph("Phase 1 — Data assembly, quality control &amp; feature construction",
                       styles["H2"]))
story.append(b(
    "<b>What I do:</b> Harmonise hourly/3-hourly pollutant and co-located meteorology onto a "
    "common time base; run automated QC (physical-range, stuck-sensor/flat-line, spike, and "
    "cross-pollutant consistency checks); apply a documented gap-filling protocol (short gaps "
    "interpolated, long gaps flagged and excluded); engineer calendar, diurnal, and "
    "lagged-pollutant features."))
story.append(b(
    "<b>Why &amp; whose method:</b> Automated multi-parameter QC and anomaly screening follow "
    "established unsupervised observation-QC practice. Lagged/rolling pollutant features and "
    "calendar encodings as ML inputs follow standard air-quality ML feature engineering "
    "(Grange et al., 2023)."))
story.append(Paragraph("→ Papers: Grange et al. (2023) for feature design; observation-QC literature.",
                       styles["MapNote"]))

story.append(Paragraph("Phase 2 — Defining &amp; labelling compound extremes", styles["H2"]))
story.append(b(
    "<b>What I do:</b> Set per-pollutant extreme thresholds at high seasonal percentiles "
    "(90th/95th) so winter does not swamp the threshold; define a <b>compound extreme</b> as "
    "joint exceedance of ≥2 pollutants at the same time step/day; quantify co-occurrence "
    "frequency and conditional probabilities P(O<sub>3</sub> extreme | PM<sub>2.5</sub> extreme), "
    "broken down by season and hour."))
story.append(b(
    "<b>Why &amp; whose method:</b> The compound/co-occurrence framing and co-exceedance "
    "frequency as the core metric follow Lyu et al. (2024, GRL). Evidence that compound events "
    "amplify concentrations far beyond single events comes from Liu et al. (2026) — compound "
    "heatwave–stagnation–inversion events raising O<sub>3</sub> and PM<sub>2.5</sub> by &gt;57%."))
story.append(Paragraph(
    "→ Papers: Lyu et al. (2024) for co-occurrence labelling; Liu et al. (2026) for "
    "compound-amplification justification.", styles["MapNote"]))

story.append(Paragraph("Phase 3 — Weather-regime identification", styles["H2"]))
story.append(b(
    "<b>What I do:</b> Construct a <b>Bangladesh-specific ventilation/stagnation index</b> "
    "(mixing-height × transport wind, plus an inversion-strength term); cluster meteorological "
    "feature vectors into a small set of recurrent regimes (K-means, cluster count by "
    "silhouette/elbow, cross-checked with self-organising maps); profile each regime by its mean "
    "pollutant load and its compound-extreme co-occurrence rate."))
story.append(b(
    "<b>Why &amp; whose method:</b> The region-specific stagnation-index construction for the "
    "Indo-Gangetic Plain — the blueprint I localise to the Bengal delta — is Zhou et al. "
    "(2024, Nature Communications). Clustering weather patterns to explain PM<sub>2.5</sub>/"
    "O<sub>3</sub> co-occurrence follows Chen et al. (2022); K-means circulation clustering "
    "yielding an explicit “stagnation cluster” follows Kim et al. (2024, AAQR)."))
story.append(Paragraph(
    "→ Papers: Zhou et al. (2024) for the stagnation index; Chen et al. (2022) + Kim et al. "
    "(2024) for regime clustering.", styles["MapNote"]))

story.append(Paragraph("Phase 4 — Meteorological normalisation &amp; regime-conditioned forecasting",
                       styles["H2"]))
story.append(b(
    "<b>What I do:</b> Separate weather-driven from emission-driven variability using "
    "meteorological normalisation (gradient-boosted trees with feature-resampling); then train "
    "forecast models — gradient-boosted trees and a sequence model (LSTM / Temporal CNN) — "
    "that take meteorological forecasts + pollutant history and output the <b>probability of a "
    "compound extreme</b> at 24/48/72 h, with the regime label supplied as an explicit "
    "conditioning input. Evaluate on a chronologically held-out test set with imbalance-aware "
    "metrics (PR-AUC, F1, Brier score, reliability diagrams) against persistence, climatology, and "
    "single-pollutant baselines."))
story.append(b(
    "<b>Why &amp; whose method:</b> The meteorological-normalisation methodology (isolating the "
    "meteorology contribution so the model does not confound a stagnant year with a high-emission "
    "one) is Grange et al. (2023). The ML-for-co-occurrence-prediction framing traces to Lyu et al. "
    "(2024). Reanalysis inputs for boundary-layer height and synoptic winds: ERA5 (Hersbach et al., "
    "2020); back-trajectory attribution of severe episodes: HYSPLIT (Stein et al., 2015)."))
story.append(Paragraph(
    "→ Papers: Grange et al. (2023) for normalisation; Lyu et al. (2024) for the forecast "
    "framing; Hersbach et al. (2020) + Stein et al. (2015) for inputs.", styles["MapNote"]))

# ---- Methodology-at-a-glance table ----
story.append(Paragraph("Methodology at a glance", styles["H2"]))
tbl_data = [
    [Paragraph("Phase", styles["TblHead"]),
     Paragraph("Core step", styles["TblHead"]),
     Paragraph("Method anchor", styles["TblHead"])],
    [Paragraph("1", styles["TblCell"]),
     Paragraph("QC, gap-filling, feature engineering", styles["TblCell"]),
     Paragraph("Grange et al. (2023); observation-QC practice", styles["TblCell"])],
    [Paragraph("2", styles["TblCell"]),
     Paragraph("Seasonal extreme thresholds + compound labelling", styles["TblCell"]),
     Paragraph("Lyu et al. (2024); Liu et al. (2026)", styles["TblCell"])],
    [Paragraph("3", styles["TblCell"]),
     Paragraph("Ventilation/stagnation index + regime clustering", styles["TblCell"]),
     Paragraph("Zhou et al. (2024); Chen et al. (2022); Kim et al. (2024)", styles["TblCell"])],
    [Paragraph("4", styles["TblCell"]),
     Paragraph("Met-normalisation + regime-conditioned forecast", styles["TblCell"]),
     Paragraph("Grange et al. (2023); Lyu et al. (2024); ERA5; HYSPLIT", styles["TblCell"])],
]
tbl = Table(tbl_data, colWidths=[1.2*cm, 6.6*cm, 8.0*cm], repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c5d2dc")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
]))
story.append(tbl)
story.append(Spacer(1, 6))

# ---- 5. Data Requirements ----
story.append(Paragraph("5. Data Requirements", styles["H1"]))

story.append(Paragraph("A. Observations to be collected (primary measurements)", styles["H2"]))
story.append(b("<i>Pollutants (hourly or 3-hourly, per station):</i>"))
story.append(bullets([
    "PM<sub>2.5</sub> (µg/m³) — primary target",
    "PM<sub>10</sub> (µg/m³)",
    "O<sub>3</sub> (ppb / µg/m³) — second compound target",
    "NOx / NO<sub>2</sub> (ppb)",
    "SO<sub>2</sub> and CO where available (source-attribution context)",
    "NH<sub>3</sub> where available (secondary aerosol context)",
], style="BodyT"))
story.append(b("<i>Co-located meteorology (same timestamp/station):</i>"))
story.append(bullets([
    "Air temperature (°C); relative humidity (%)",
    "Wind speed (m/s) and wind direction (°)",
    "Solar radiation / shortwave flux (W/m²) — photochemistry driver for O<sub>3</sub>",
    "Precipitation (mm) — wet-removal driver; surface pressure (hPa)",
], style="BodyT"))
story.append(b("<i>Station metadata:</i> name, latitude/longitude, elevation, land-use/site "
               "classification (traffic / industrial / background)."))

story.append(Paragraph("B. Variables to be derived / calculated", styles["H2"]))
story.append(bullets([
    "Wind components u, v and circular encodings (sin WD, cos WD).",
    "<b>Planetary boundary-layer / mixing height</b> and <b>temperature-inversion strength</b> "
    "(from reanalysis vertical profiles or surface–upper-level gradients).",
    "<b>Ventilation index</b> = mixing height × transport wind speed; and an "
    "<b>atmospheric-stagnation flag</b> from low-wind, low-precipitation, stable-layer criteria, "
    "calibrated to local conditions.",
    "Pollutant <b>lag features</b> (1–24 h), rolling means, and rate-of-change terms.",
    "<b>Seasonal extreme thresholds</b> (per-pollutant 90th/95th percentiles by season).",
    "<b>Compound-extreme labels</b> (binary joint-exceedance flags) and conditional "
    "co-occurrence probabilities; diurnal/seasonal/holiday calendar encodings.",
], style="BodyT"))

story.append(Paragraph("C. Supplementary gridded data (regimes and transboundary context)", styles["H2"]))
story.append(bullets([
    "Reanalysis meteorology (ERA5) for boundary-layer height, vertical stability, and 850 hPa winds.",
    "Satellite AOD (MODIS/MAIAC) and fire/biomass-burning counts (VIIRS/MODIS active fire) to "
    "contextualise transboundary and regional burning contributions.",
    "Back-trajectory inputs (HYSPLIT) to attribute air-mass origin during the most severe compound episodes.",
], style="BodyT"))

# ---- 6. Expected Impact ----
story.append(Paragraph("6. Expected Impact", styles["H1"]))
story.append(b(
    "<b>Scientific.</b> This will deliver the first quantitative characterisation of compound "
    "PM<sub>2.5</sub>–O<sub>3</sub> extremes for Bangladesh and the eastern IGP, a locally "
    "validated ventilation/stagnation index transferable to other delta and South-Asian urban "
    "settings, and an evidence base for <i>why</i> compound events behave differently from "
    "single-pollutant episodes in a humid, monsoon-influenced climate."))
story.append(b(
    "<b>Operational and public-health.</b> A regime-conditioned forecast of compound extremes is "
    "directly actionable: it provides the technical core for a 24–72 h public-health "
    "early-warning system for Dhaka and other Bangladeshi cities, enabling targeted advisories for "
    "vulnerable groups during the highest-risk meteorological windows. Because the method conditions "
    "on weather forecasts, it can be coupled to operational numerical weather prediction without new "
    "emission inventories."))
story.append(b(
    "<b>Capacity and data infrastructure.</b> The quality-control and gap-filling pipeline, with "
    "documented protocols, leaves behind a reusable, defensible data-processing framework for the "
    "national monitoring network — a contribution independent of the modelling results."))

# ---- 7. References ----
story.append(Paragraph("7. References", styles["H1"]))
refs = [
    "Lyu, B. et al. (2024). Frequent co-occurrence of PM<sub>2.5</sub> and O<sub>3</sub> pollution "
    "and its synoptic drivers, characterised with machine learning. <i>Geophysical Research Letters</i>.",
    "Chen, Z. et al. (2022). Weather-pattern clustering and the co-occurrence of PM<sub>2.5</sub> and "
    "ozone extremes. <i>Environmental Research / PMC</i>.",
    "Zhou, S. et al. (2024). A region-specific atmospheric stagnation index and projected stagnation "
    "changes over the Indo-Gangetic Plain. <i>Nature Communications</i>.",
    "Liu, Y. et al. (2026). Compound meteorological events (heatwave–stagnation–inversion) "
    "amplify O<sub>3</sub> and PM<sub>2.5</sub> over eastern China. <i>Atmosphere (MDPI)</i>.",
    "Kim, S. et al. (2024). K-means circulation patterns and the stagnation cluster controlling "
    "high-PM<sub>2.5</sub> days. <i>Aerosol and Air Quality Research</i>.",
    "Grange, S. K. et al. (2023). Meteorological normalisation of PM<sub>2.5</sub> using machine "
    "learning and chemistry-transport benchmarking. <i>npj Climate and Atmospheric Science</i>.",
    "Begum, B. A. &amp; Hopke, P. K. (2018–2023, various). Source apportionment of particulate "
    "matter in Dhaka, Bangladesh. <i>Aerosol and Air Quality Research / Atmospheric Pollution Research</i>.",
    "World Bank (2022). <i>Breathing Heavy: New Evidence on Air Pollution and Health in Bangladesh</i>.",
    "Stein, A. F. et al. (2015). NOAA's HYSPLIT atmospheric transport and dispersion modeling system. "
    "<i>Bulletin of the American Meteorological Society</i>.",
    "Hersbach, H. et al. (2020). The ERA5 global reanalysis. <i>Quarterly Journal of the Royal "
    "Meteorological Society</i>.",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}] {r}", styles["Ref"]))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#8a8a8a"))
    canvas.drawString(2*cm, 1.1*cm,
                      "Research Proposal — Bishwadip Maitra")
    canvas.drawRightString(A4[0]-2*cm, 1.1*cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
    title="Research Proposal — Bishwadip Maitra",
    author="Bishwadip Maitra",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("WROTE", OUT)
