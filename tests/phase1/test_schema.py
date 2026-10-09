ULB_COLUMNS = {"Time", "Amount", "Class", *(f"V{i}" for i in range(1, 29))}


def test_ulb_schema_contract_is_explicit():
    assert len(ULB_COLUMNS) == 31
    assert {"Time", "Amount", "Class"}.issubset(ULB_COLUMNS)
