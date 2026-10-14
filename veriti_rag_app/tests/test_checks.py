from veriti_rag.checks import unverified_numbers


def test_numbers_from_result_pass():
    assert unverified_numbers("There are 3 deals worth 4,500 in total [S1].", [[3, 4500.0]]) == []


def test_percentage_from_fraction_and_rounding():
    assert unverified_numbers("Win rate is 66.7%.", [[0.6667]]) == []
    assert unverified_numbers("Average is 12.35.", [[12.3456]]) == []


def test_invented_number_flagged():
    assert unverified_numbers("There are 7 deals.", [[1]]) == ["7"]


def test_ids_dates_citations_and_list_markers_ignored():
    assert unverified_numbers("1. See DRC-COMP-002 [S3].\n2. Closed on 2017-05-01.", []) == []


def test_row_count_and_context_numbers_allowed():
    assert unverified_numbers("2 agents; severity weight 4", [["a"], ["b"]], "weight 4") == []
