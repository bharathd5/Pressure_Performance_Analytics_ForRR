"""Builds reports/methodology_report.pdf from results/*.csv + figures.  PYTHONPATH=. python reports/build_report.py"""
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from src.config import RES, REP, HIGH_LI, EXTREME_LI, WEIGHTS, MIN_HI_BALLS, MIN_HI_MATCHES, STAKES_MULT

M = json.load(open(RES / "run_metadata.json")); chk = M["checks"]
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12.2, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, spaceBefore=8, spaceAfter=5, textColor=colors.HexColor("#1f3a5f"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#1f3a5f"))
SM = ParagraphStyle("sm", parent=B, fontSize=7.8, leading=10, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("c", parent=B, fontSize=7.6, leading=9.2, spaceAfter=0)
P = lambda t, s=B: Paragraph(t, s)
BUL = lambda items: [Paragraph("&bull; " + i, ParagraphStyle("bl", parent=B, leftIndent=10, firstLineIndent=-8, spaceAfter=2)) for i in items]


def tbl(rows, widths, hdr=True, fs=7.6):
    rows = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(rows, colWidths=widths, repeatRows=1 if hdr else 0)
    st = [("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if hdr: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f3")))
    t.setStyle(TableStyle(st)); return t


def img(name, w):
    from PIL import Image as PI
    iw, ih = PI.open(REP / "figures" / name).size
    return Image(str(REP / "figures" / name), width=w, height=w * ih / iw)


bat = pd.read_csv(RES / "top10_batters.csv"); bowl = pd.read_csv(RES / "top10_bowlers.csv")
tp = pd.read_csv(RES / "pressure_tier_profile.csv").set_index("pressure_tier")
sb, sw = pd.read_csv(RES / "sensitivity_bat.csv"), pd.read_csv(RES / "sensitivity_bowl.csv")
tb, tw = M["bat"]["temporal"], M["bowl"]["temporal"]
S = []
S += [P("Identifying the Most Reliable Players Under Pressure", ParagraphStyle("t", parent=ss["Title"], fontSize=19, textColor=colors.HexColor("#1f3a5f"))),
      P("Methodology report &mdash; SupeRR Selector Season 2, Round 1 &nbsp;|&nbsp; IPL + T20I ball-by-ball data, 2022&ndash;2026 &nbsp;|&nbsp; fully reproducible (see Section 9)", SM), Spacer(1, 6)]

S += [P("1. Summary", H1),
      P("We define <b>pressure as the leverage of a delivery</b>: the expected size of the swing in the batting side&rsquo;s win probability that the ball can produce, "
        "normalised so the average ball = 1.0. Leverage is a single, data-driven function of required run rate, wickets in hand, balls remaining and how close the contest is, "
        "and it is symmetric: a high-leverage ball is equally high-stakes for the batter and the bowler. Performance is measured in <b>runs above an average player in the same situation</b> "
        "(run-expectancy value), adjusted for batter-hand &times; pace/spin &times; phase and for the quality of the opponent. Both disciplines use the identical pipeline.", B),
      P("<b>Headline finding.</b> Pressure-specific &ldquo;clutch&rdquo; ability is statistically hard to detect in 4.5 seasons of data: players do <i>not</i> differ in how much they lift or drop under pressure "
        f"beyond sampling noise (heterogeneity test p = {M['bat']['hetero_p']['D']:.2f} batters, {M['bowl']['hetero_p']['D']:.2f} bowlers), "
        "whereas overall run-value skill differs hugely (p &lt; 0.001). A ranking that simply sorts raw pressure-ball statistics would therefore mostly rank luck. "
        "Our system uses <b>partial pooling</b>: a player&rsquo;s pressure rating is his overall skill (estimated from every delivery) plus a shrunk pressure-specific adjustment. "
        "The adjustment component is retained so that, on a bigger dataset where real clutch differences exist, the same code up-weights them automatically.", B)]
S += [P("Final rankings (as entered on Unstop)", H2)]
rows = [["Rank", "Batter", "Rating", "P(top-10)", "Rank", "Bowler", "Rating", "P(top-10)"]]
for i in range(10):
    a, b = bat.iloc[i], bowl.iloc[i]
    rows.append([i + 1, f"{a['name']} (id {int(a.pid)})", f"{a.rating:.1f}", f"{a.p_top10:.2f}", i + 1, f"{b['name']} (id {int(b.pid)})", f"{b.rating:.1f}", f"{b.p_top10:.2f}"])
S += [tbl(rows, [1.2*cm, 4.4*cm, 1.4*cm, 1.7*cm, 1.2*cm, 4.4*cm, 1.4*cm, 1.7*cm]),
      P("Rating = 50 + 10&times;(weighted z-score; section 5). P(top-10) = share of 300 match-resampled (Bayesian-bootstrap) replications in which the player stays in the top 10.", SM)]

S += [PageBreak(), P("2. Data, cleaning and what the feed does not contain", H1),
      P(f"Inputs: ball_by_ball ({chk['deliveries']:,} deliveries, {chk['matches_with_balls']} matches with balls) plus matches, competitions, teams, playing XI and mapping files. "
        "Exploration found five issues that would silently corrupt any naive analysis; each is handled in <font face='Courier'>src/data_processing/clean_data.py</font>:", B)]
S += BUL(["<b>`runs` is runs off the bat only.</b> Wide, bye, leg-bye and no-ball penalty runs are absent (the column <i>extras1_runs</i> is a category code, not a run count). "
          "We impute: wide = 1, no-ball = 1, bye/leg-bye = 1 (4 if boundary), penalty = 5, and keep extras separate so batters are never credited with byes and bowlers are not charged for them.",
          "<b>Illegal deliveries occupy slots in the over.</b> Wides, no-balls and the category &lsquo;other&rsquo; (which sits in the 7th slot of an over exactly like a no-ball) are excluded from the legal-ball count. "
          "After this fix every innings has &le;120 legal balls (before the fix 150 innings exceeded 120).",
          "<b>Super overs (innings 3/4) and 16 abandoned matches</b> are excluded; reduced-over matches use the supplied revised overs/targets.",
          f"<b>Validation.</b> Re-deriving each winner from reconstructed totals agrees with matches.csv in {chk['winner_agreement_pct']:.1f}% of {chk['n_checked']} matches. "
          "The residual disagreement is almost entirely chases won by 1&ndash;6 runs short of the reconstructed target (missing bye/leg-bye runs). "
          "For that reason <b>terminal states and win/loss labels use the recorded match result</b>, not our reconstructed totals.",
          "<b>Dismissals.</b> Wickets credited to the bowler: caught, bowled, LBW, stumped, hit-wicket. Run-outs and retirements cost the batter but not the bowler credit. "
          "Non-striker run-outs (~0.14% of balls) are not charged to anyone."])

S += [P("3. Defining pressure", H1),
      P("<b>Definition.</b> For every delivery we compute the pre-ball win probability <i>W</i> and, for each possible outcome o (dot, 1, 2, 3, 4, 6, wicket, wide) the post-ball probability W<sub>o</sub>. "
        "Leverage = &Sigma;<sub>o</sub> p(o)&middot;|W<sub>o</sub> &minus; W|, where p(o) is the league-wide outcome mix for the same phase and wickets bucket (the &lsquo;average player&rsquo;). "
        "It is divided by its mean so 1.0 = an average ball. The <b>Pressure Index</b> multiplies this by a small stakes factor for knock-outs "
        f"({', '.join(f'{k} &times;{v}' for k, v in STAKES_MULT.items() if v > 1)}).", B),
      P("<b>Why this is meaningful.</b> Every ingredient the brief lists &mdash; required rate, wickets in hand, balls remaining, innings stage, closeness &mdash; matters only through its effect on "
        "<i>how much the next ball can change who wins</i>. A win-probability model learns the right interaction of those inputs from data instead of us hand-weighting them. "
        "Leverage is also exactly what &lsquo;when the game is on the line&rsquo; means. It is symmetric, so the same number applies to the batter facing the ball and the bowler delivering it.", B),
      P("<b>Win-probability model.</b> Gradient-boosted classifier on innings, balls left, wickets lost, runs, runs vs par, runs needed, required rate and competition type. "
        "Predictions are <b>cross-fitted</b> (5-fold grouped by match: every match is scored by a model that never saw it), so no leakage. "
        "Out-of-fold Brier score 0.181; calibration in Fig. 3.", B),
      P(f"<b>Tiers.</b> Low (&lt;0.7), Medium (0.7&ndash;{HIGH_LI}), <b>High (&ge;{HIGH_LI})</b>, Extreme (&ge;{EXTREME_LI}, a subset of High used for description). "
        "&lsquo;High pressure&rsquo; in all ratings = High + Extreme; &lsquo;normal&rsquo; = everything else.", B),
      img("fig1_leverage_heatmap.png", 17*cm),
      P("Fig. 1 &mdash; Pressure is concentrated in the last overs of chases and peaks when the required rate is 8&ndash;15 (still winnable); it falls when the target is trivially close or hopeless. "
        "First-innings pressure is low except at the death, matching cricket intuition.", SM)]
rows = [["Tier", "% of balls", "Mean leverage", "Mean over", "Wkts in hand", "Closeness", "Run rate", "Wicket %", "|Win-prob swing| %"]]
for t, r in tp.iterrows():
    rows.append([t, f"{r.share_pct:.1f}", f"{r.mean_leverage:.2f}", f"{r.mean_overs:.1f}", f"{r.wkts_in_hand:.1f}", f"{r.closeness:.2f}", f"{r.run_rate:.2f}", f"{r.wicket_pct:.2f}", f"{r.abs_wpa:.2f}"])
S += [tbl(rows, [1.7*cm, 1.6*cm, 2*cm, 1.6*cm, 1.9*cm, 1.7*cm, 1.6*cm, 1.7*cm, 3*cm]),
      P("Table 1 &mdash; How pressure tiers differ from ordinary overs. Actual absolute win-probability movement per ball rises ~6&times; from Low to High and ~12&times; to Extreme; contests are far closer "
        "(0.44 &rarr; 0.67); Extreme balls come late with fewer wickets in hand and see a higher wicket rate (7.1% vs 5.5%).", SM),
      img("fig2_tiers.png", 16.5*cm)]

S += [PageBreak(), P("4. Measuring performance (identical for batters and bowlers)", H1),
      P("<b>Currency: run-expectancy value.</b> A model of expected runs remaining given balls left, wickets lost and competition type gives each state a value. "
        "Each delivery&rsquo;s value = runs scored + change in expected runs remaining; the wicket term charges the dismissed batter / credits the bowler with the run-value a wicket destroys. "
        "An average player scores 0 in every state, so values are <i>runs above average in the same situation</i>. Batter value = bat runs + ball-and-wicket effects; "
        "bowler value = &minus;(runs conceded + effects), so a positive number is good for both. Extras that are not the bowler&rsquo;s fault are excluded.", B),
      P("<b>Context baseline (the &lsquo;expected&rsquo; model).</b> From each value we subtract the league mean for the same <i>phase &times; batter hand &times; pace/spin</i> cell, so a left-hander facing spin in the powerplay "
        "is compared with other such deliveries, not with the whole league. This is the matchup layer of the brief.", B),
      P("<b>Opponent quality.</b> Every batter and bowler gets an empirical-Bayes skill estimate (shrunk toward his <i>team&rsquo;s</i> mean, so unknown players on weak T20I sides are not treated as average). "
        "A delivery&rsquo;s value is then adjusted by the skill of the opponent faced: beating a good bowler counts for more. IPL and T20I are pooled; the run-expectancy and win-probability models carry a competition flag, "
        "and the opponent adjustment is what stops T20I feasting on weaker sides from inflating ratings.", B)]

S += [P("5. From deliveries to a rating", H1),
      P("For each player we form three components, each estimated with empirical-Bayes shrinkage (DerSimonian&ndash;Laird between-player variance; <b>cluster-robust standard errors by match</b> because balls within a match are correlated):", B)]
S += BUL(["<b>g &mdash; overall skill</b>: shrunk mean adjusted value per delivery over <i>all</i> deliveries (the player&rsquo;s &lsquo;usual level&rsquo;, the most stable signal available).",
          "<b>D &mdash; pressure lift</b>: shrunk (High-pressure mean &minus; normal mean), i.e. performance under pressure compared with his own usual performance.",
          "<b>P &mdash; expected pressure performance</b> = g + (1&minus;s)&middot;D, with s = league share of High deliveries.",
          "<b>C &mdash; consistency</b>: shrunk share of the player&rsquo;s pressure matches (&ge;6 pressure deliveries) with a net positive value.",
          f"<b>Rating</b> = {WEIGHTS['P']}&middot;z(P) + {WEIGHTS['D']}&middot;z(D) + {WEIGHTS['C']}&middot;z(C), where each z is the posterior mean divided by the estimated SD of <i>true</i> between-player differences &mdash; "
          "so a component that is mostly noise has almost no spread and cannot dominate. Displayed as 50 + 10&times;rating."])
S += [P(f"<b>Eligibility (fair comparison).</b> &ge;{MIN_HI_BALLS['bat']} High-pressure deliveries and &ge;{MIN_HI_MATCHES['bat']} matches with at least one: {M['bat']['n_eligible']} batters and {M['bowl']['n_eligible']} bowlers qualify. "
        "Shrinkage handles the remaining sample-size differences; the bootstrap intervals (below) express what is left.", B)]

S += [PageBreak(), P("6. Evidence for the top 10", H1)]
rows = [["#", "Batter", "Matches", "Pressure balls", "SR pressure", "SR normal", "Outs/100 pressure", "Overall skill g", "Rating", "Rank (median, 90% int.)"]]
for i, r in bat.iterrows():
    rows.append([i + 1, r["name"], int(r.n_matches), int(r.hi_balls), f"{100*r.hi_runs/r.hi_balls:.0f}", f"{100*r.lo_runs/r.lo_balls:.0f}", f"{100*r.hi_out/r.hi_balls:.1f}", f"{r.g:.3f}", f"{r.rating:.1f}", f"{r.median_rank:.0f} ({r.rank_p05:.0f}-{r.rank_p95:.0f})"])
S += [tbl(rows, [.6*cm, 3.4*cm, 1.4*cm, 1.6*cm, 1.4*cm, 1.4*cm, 1.8*cm, 1.7*cm, 1.2*cm, 2.8*cm])]
rows = [["#", "Bowler", "Matches", "Pressure balls", "Econ pressure", "Econ normal", "Wkts/100 pressure", "Dot % pressure", "Overall skill g", "Rating", "Rank (median, 90% int.)"]]
for i, r in bowl.iterrows():
    rows.append([i + 1, r["name"], int(r.n_matches), int(r.hi_balls), f"{6*r.hi_runs/r.hi_balls:.2f}", f"{6*r.lo_runs/r.lo_balls:.2f}", f"{100*r.hi_wkt/r.hi_balls:.1f}", f"{100*r.hi_dots/r.hi_balls:.0f}", f"{r.g:.3f}", f"{r.rating:.1f}", f"{r.median_rank:.0f} ({r.rank_p05:.0f}-{r.rank_p95:.0f})"])
S += [Spacer(1, 5), tbl(rows, [.6*cm, 3.1*cm, 1.3*cm, 1.4*cm, 1.4*cm, 1.4*cm, 1.6*cm, 1.4*cm, 1.4*cm, 1.1*cm, 2.5*cm]),
      P("Tables 2&ndash;3 &mdash; Raw pressure statistics are shown for readability only; the ordering comes from the shrunk, context- and opponent-adjusted ratings. "
        "Strike rate/economy do not capture wicket cost or context, which is why e.g. Rajat Patidar and Glenn Phillips can rank above higher-SR players. "
        "<b>Transparency note:</b> Blessing Muzarabani (bowlers #4) concedes noticeably more in pressure balls than in normal overs (10.1 vs 7.1 economy over only 123 pressure balls); "
        "his rank rests on strong overall skill and the 90% interval (1&ndash;35) is the widest in the list. Such a case would be the first to move if more data arrived.", SM),
      img("fig5_rank_intervals.png", 17*cm),
      P("Fig. 5 &mdash; Rank stability under match-level resampling. Top ranks are firm; positions 7&ndash;10 are close to interchangeable with players just outside the list.", SM)]

S += [P("7. Reliability, validation and sensitivity", H1),
      img("fig4_reliability.png", 9*cm),
      P(f"<b>Split-half reliability (200 random match splits).</b> Overall skill repeats (rho {M['bat']['split_half']['overall']:.2f} batters, {M['bowl']['split_half']['overall']:.2f} bowlers). "
        f"Raw pressure-ball mean ({M['bat']['split_half']['pressure_mean']:.2f}/{M['bowl']['split_half']['pressure_mean']:.2f}), lift D ({M['bat']['split_half']['lift_D']:.2f}/{M['bowl']['split_half']['lift_D']:.2f}) "
        f"and hit rate ({M['bat']['split_half']['hit_rate']:.2f}/{M['bowl']['split_half']['hit_rate']:.2f}) essentially do not. This is the empirical reason for partial pooling.", B),
      P(f"<b>Heterogeneity (Cochran Q on cluster-robust SEs).</b> Overall skill p &lt; 0.001; pressure lift D p = {M['bat']['hetero_p']['D']:.2f}/{M['bowl']['hetero_p']['D']:.2f}; hit rate p = {M['bat']['hetero_p']['C']:.2f}/{M['bowl']['hetero_p']['C']:.2f} (batters/bowlers). "
        "Estimated true SD of pressure lift = 0; the system therefore (correctly) gives D and C near-zero influence on this dataset.", B),
      P(f"<b>Out-of-sample test.</b> Ratings built only from matches before {M['cutoff']} were compared with each player&rsquo;s High-pressure value <i>after</i> that date. "
        f"Bowlers: rho = {tw['rho_rating_vs_post']:.2f} (90% CI {tw['rho_rating_ci_lo']:.2f} to {tw['rho_rating_ci_hi']:.2f}, n = {int(tw['n_players'])}) versus {tw['rho_naive_raw_pressure_vs_post']:.2f} for the naive raw-pressure ranking; "
        f"top-10 bowlers later averaged {tw['post_mean_top10_rating']:+.3f} runs/ball vs {tw['post_mean_all_eligible']:+.3f} for all eligible bowlers. "
        f"<b>Batters: rho = {tb['rho_rating_vs_post']:.2f} (CI {tb['rho_rating_ci_lo']:.2f} to {tb['rho_rating_ci_hi']:.2f}, n = {int(tb['n_players'])}) &mdash; no demonstrable out-of-sample predictive power</b>; "
        f"the pre-cut-off top-10 batters later averaged {tb['post_mean_top10_rating']:+.3f} vs {tb['post_mean_all_eligible']:+.3f} for all. "
        "With ~60 players and ~100 post-period pressure balls each the test is low-powered, but we report it as is: the batter list is the less certain of the two.", B),
      img("fig6_temporal.png", 8*cm),
      P("<b>Sensitivity.</b> Number of players retained in the top 10 when the design changes:", B)]
rows = [["Variant", "Batters", "Bowlers"]] + [[sb.variant[i], f"{sb.overlap_with_base_top10[i]}/10", f"{sw.overlap_with_base_top10[i]}/10"] for i in range(len(sb))]
S += [tbl(rows, [6*cm, 2.5*cm, 2.5*cm]), P("Lists are insensitive to component weights (D and C carry ~no weight by construction) and moderately sensitive to the High threshold at the strict end (LI &ge; 2.0 leaves only ~34k High+Extreme balls).", SM)]

S += [PageBreak(), P("8. Limitations and how the brief&rsquo;s additional factors were treated", H1)]
S += BUL(["<b>Clutch is mostly unobservable here.</b> With ~150&ndash;450 pressure deliveries per player and ~2.7 runs of per-ball noise, pressure-specific skill cannot be separated from luck. Rankings are therefore driven by overall skill that has been shown to hold up where pressure arises. We say so explicitly rather than rank noise.",
          "<b>Imputed extras</b> (byes/leg-byes) are approximate; the effect is on innings totals by a few runs, not on batter/bowler attribution.",
          "<b>No line/length or ball-tracking coordinates.</b> The feed has delivery style and stroke only. Experience, form, head-to-head and in-match tendencies are implemented as features: <i>experience</i> &rarr; eligibility and shrinkage; "
          "<i>form</i> &rarr; exponentially-weighted recent value per player (reported, not rated); <i>matchups</i> &rarr; hand &times; pace/spin &times; phase baseline and opponent skill (in the rating) plus shrunk batter-vs-bowler head-to-head tables (descriptive); "
          "<i>in-match line/length/strong zones</i> &rarr; `in_match_tendencies()` returns what a bowler&rsquo;s styles and a batter&rsquo;s strokes achieved earlier in that match. "
          "We deliberately keep form and in-match tendencies <b>out of the retrospective rating</b>: they are predictive inputs for a live decision tool and would double-count performance already in the player&rsquo;s record.",
          "<b>Opponent adjustment is approximate</b> (two-pass, shrunk skill); T20I players meeting only weak sides remain harder to assess. IPL and T20I are pooled rather than modelled hierarchically.",
          "<b>Role effects</b>: finishers and openers face pressure in different phases; the phase-level baseline removes the average phase effect but not individual role selection.",
          "<b>Batter list less certain</b> than the bowler list (Section 7). We recommend treating positions 6&ndash;10 as a pool rather than a strict order."])

S += [P("9. Reproducing and reusing the system", H1),
      P("<font face='Courier'>pip install -r requirements.txt &nbsp;&rarr;&nbsp; place the six CSVs in data/raw/ &nbsp;&rarr;&nbsp; python run_pipeline.py &nbsp;&rarr;&nbsp; python reports/make_figures.py &nbsp;&rarr;&nbsp; python reports/build_report.py &nbsp;&rarr;&nbsp; streamlit run dashboard/app.py</font>", B),
      P("All thresholds, weights, eligibility limits and stakes factors live in <font face='Courier'>src/config.py</font>; the same code runs on any ball-by-ball feed with the same columns (another competition, women&rsquo;s T20, a new season). "
        "Modules are single-purpose: <i>pressure_engine</i> (leverage &amp; tiers), <i>models</i> (win probability, run expectancy, context/opponent adjustment, shrinkage, rating), "
        "<i>evaluation</i> (reliability, bootstrap, out-of-sample, sensitivity). Random seeds are fixed (SEED = 42). The rating code operates on player-match tables, so a new dataset only needs ball-level values and a pressure index.", B)]


def foot(c, d):
    c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(colors.grey); c.drawString(1.8*cm, 1*cm, "Pressure Performance Analytics - methodology report"); c.drawRightString(A4[0]-1.8*cm, 1*cm, f"Page {d.page}"); c.restoreState()


SimpleDocTemplate(str(REP / "methodology_report.pdf"), pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.6*cm, bottomMargin=1.6*cm,
                  title="Identifying the Most Reliable Players Under Pressure").build(S, onFirstPage=foot, onLaterPages=foot)
print("report built")
