import score_ablation as S
W = "Marbury"
def test_marker_same_T():
    assert S.marker_label("- User lives at the Marbury Residences for now.", W, "simple_fornow") == "MARKER-KEPT-SAME"
    assert S.marker_label("- Temporarily living at Marbury Residences (as of Jan 2026)", W, "simple_fornow") == "MARKER-KEPT-SAME"
def test_marker_same_C():
    assert S.marker_label("- Currently lives at the Marbury Residences on Fifth Street.", W, "simple_atm") == "MARKER-KEPT-SAME"
    assert S.marker_label("- Lives at Marbury Residences at the moment", W, "simple_atm") == "MARKER-KEPT-SAME"
def test_marker_cross():
    assert S.marker_label("- Currently lives at the Marbury Residences.", W, "simple_fornow") == "MARKER-KEPT-CROSS"
    assert S.marker_label("- Lives at the Marbury Residences for now.", W, "simple_atm") == "MARKER-KEPT-CROSS"
def test_prog_converted():
    assert S.marker_label("- User is living at the Marbury Residences on Fifth Street (Jan 2026).", W, "simple_fornow") == "PROG-CONVERTED"
def test_date_only_and_dropped():
    assert S.marker_label("- User lives at the Marbury Residences on Fifth Street (as of Jan 2026).", W, "simple_fornow") == "DATE-ONLY"
    assert S.marker_label("- Lives at the Marbury Residences on Fifth Street.", W, "simple_atm") == "DROPPED"
def test_witness_dropped_and_residual():
    assert S.marker_label("- Likes podcasts for commute.", W, "simple_fornow") == "WITNESS-DROPPED"
    assert S.marker_label("- Address: Marbury Residences, Fifth Street", W, "simple_fornow") == "RESIDUAL"
def test_cascade_passthrough():
    assert S.label({"form": "prog", "response": "- User lives at the Marbury Residences.", "wit_token": W}) == "COERCED-STATIVE"
    assert S.label({"form": "simple", "response": "- User lives at the Marbury Residences.", "wit_token": W}) == "PRESERVED"
def test_placebo():
    assert S.placebo_marker_hits({"response": "- Currently lives at Marbury Residences", "wit_token": W}) == {"T": False, "C": True}
