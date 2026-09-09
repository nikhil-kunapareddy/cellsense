"""Pins cellsense.io.loaders: dtype inference, the ID-vs-date trap, CSV
encoding fallback, multi-sheet Excel, empty-sheet skip, duplicate-filename
disambiguation, and unsupported-extension handling.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cellsense.errors import DataError, UnsupportedFileTypeError
from cellsense.io.loaders import _looks_date_like, load_workspace
from tests.fixtures import builders as fb


class TestLooksDateLike:
    """Direct pins on the private gate that decides whether to even attempt
    ``pd.to_datetime`` on a text column.
    """

    def test_bare_digit_strings_with_non_date_name_are_not_date_like(self) -> None:
        """'1000'/'1001'/... under a non-date-hinting name must not look date-like,
        even though pd.to_datetime("1000") itself would happily parse to year 1000.
        """
        series = pd.Series(["1000", "1001", "1002"])
        assert _looks_date_like("product_code", series) is False

    def test_alnum_ids_are_never_date_like_regardless_of_name(self) -> None:
        series = pd.Series(["E2001", "E2002", "E2003"])
        assert _looks_date_like("employee_id", series) is False
        assert _looks_date_like("order_date", series) is False  # even with a date-hint name

    def test_separator_bearing_values_are_date_like(self) -> None:
        series = pd.Series(["2024-01-15", "2024-02-20", "2024-03-05"])
        assert _looks_date_like("anything", series) is True

    def test_date_hinting_name_with_parseable_values_is_date_like(self) -> None:
        series = pd.Series(["2024/01/15", "2024/02/20"])
        assert _looks_date_like("created_at", series) is True

    def test_date_shaped_but_invalid_values_are_rejected_by_the_roundtrip_check(self) -> None:
        """Looks like a date (two separators, three groups) but month=15 is invalid --
        the round-trip pd.to_datetime(..., errors="raise") must reject it.
        """
        series = pd.Series(["2024-15-99", "2024-16-98"])
        assert _looks_date_like("batch_code", series) is False

    def test_empty_series_is_never_date_like(self) -> None:
        assert _looks_date_like("order_date", pd.Series([], dtype=object)) is False


class TestLoadCsv:
    def test_alnum_id_column_stays_text_not_datetime(self, write_csv) -> None:
        path = write_csv("ids.csv", fb.alnum_id_frame())
        ws = load_workspace([path])
        df = ws.table("ids.csv")
        assert not pd.api.types.is_datetime64_any_dtype(df["employee_id"])
        assert list(df["employee_id"]) == ["E2001", "E2002", "E2003"]

    def test_bare_digit_id_column_becomes_numeric_not_datetime(self, write_csv) -> None:
        """The real trap: pd.to_datetime("1000") parses as year 1000. The loader
        must never reach that call for a plain numeric-looking ID column.
        """
        path = write_csv("codes.csv", fb.id_like_bare_digit_frame())
        ws = load_workspace([path])
        df = ws.table("codes.csv")
        assert not pd.api.types.is_datetime64_any_dtype(df["product_code"])
        assert pd.api.types.is_numeric_dtype(df["product_code"])
        assert df["product_code"].tolist() == [1000, 1001, 1002, 1003]

    def test_real_date_column_is_parsed_as_datetime(self, write_csv) -> None:
        path = write_csv("dates.csv", fb.real_date_frame())
        ws = load_workspace([path])
        df = ws.table("dates.csv")
        assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
        assert df["order_date"].iloc[0] == pd.Timestamp("2024-01-15")

    def test_partially_numeric_column_stays_text(self, write_csv) -> None:
        """A column that is *mostly* numeric but has a couple of sentinel
        strings ('N/A', 'unknown') must not be silently coerced to numeric --
        that would turn the sentinels into NaN and lose information.
        """
        path = write_csv("partial.csv", fb.partially_numeric_text_frame())
        ws = load_workspace([path])
        df = ws.table("partial.csv")
        assert not pd.api.types.is_numeric_dtype(df["code"])
        assert df["code"].tolist() == ["100", "200", "pending", "300", "unknown"]

    def test_nans_survive_load(self, write_csv) -> None:
        path = write_csv("nans.csv", fb.nan_frame())
        ws = load_workspace([path])
        df = ws.table("nans.csv")
        assert df["a"].isna().sum() == 1
        assert df["b"].isna().sum() == 1

    def test_unicode_values_round_trip(self, write_csv) -> None:
        path = write_csv("unicode.csv", fb.unicode_frame())
        ws = load_workspace([path])
        df = ws.table("unicode.csv")
        assert "Renée" in df["name"].tolist()
        assert "田中" in df["name"].tolist()
        assert "😀 Chris" in df["name"].tolist()

    def test_column_names_are_stripped_of_whitespace(self, write_csv) -> None:
        df = pd.DataFrame({" region ": ["a", "b"], "value": [1, 2]})
        path = write_csv("whitespace.csv", df)
        ws = load_workspace([path])
        assert list(ws.table("whitespace.csv").columns) == ["region", "value"]

    def test_string_values_are_stripped_of_whitespace(self, write_csv) -> None:
        df = pd.DataFrame({"name": [" Ana ", " Ben"], "n": [1, 2]})
        path = write_csv("strip.csv", df)
        ws = load_workspace([path])
        assert ws.table("strip.csv")["name"].tolist() == ["Ana", "Ben"]

    def test_empty_csv_raises_data_error(self, tmp_path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("a,b\n", encoding="utf-8")  # header only, zero data rows
        with pytest.raises(DataError):
            load_workspace([path])

    def test_latin1_encoded_csv_falls_back_and_loads(self, tmp_path) -> None:
        path = tmp_path / "latin1.csv"
        raw = "name,value\nCafé,1\nNaïve,2\n".encode("latin-1")
        path.write_bytes(raw)
        ws = load_workspace([path])
        df = ws.table("latin1.csv")
        assert df["name"].tolist() == ["Café", "Naïve"]

    def test_mixed_type_object_column_loads_without_raising(self, write_csv) -> None:
        # to_csv/read_csv round-trips everything as text, so this mainly pins
        # that a genuinely heterogeneous-looking column doesn't blow up load.
        path = write_csv("mixed.csv", fb.mixed_types_frame())
        ws = load_workspace([path])
        assert len(ws.table("mixed.csv")) == 5


class TestLoadExcel:
    def test_multi_sheet_workbook_loads_every_non_empty_sheet(self, write_xlsx) -> None:
        path = write_xlsx("book.xlsx", fb.multi_sheet_workbook())
        ws = load_workspace([path])
        fd = ws.files["book.xlsx"]
        assert set(fd.sheets) == {"Catalog", "Inventory"}

    def test_empty_sheet_is_skipped(self, write_xlsx) -> None:
        path = write_xlsx("book.xlsx", fb.multi_sheet_workbook())
        ws = load_workspace([path])
        assert "Empty" not in ws.files["book.xlsx"].sheets

    def test_all_sheets_empty_raises_data_error(self, write_xlsx) -> None:
        path = write_xlsx("allempty.xlsx", {"Sheet1": fb.empty_frame()})
        with pytest.raises(DataError):
            load_workspace([path])

    def test_file_type_is_excel(self, write_xlsx) -> None:
        path = write_xlsx("book.xlsx", fb.multi_sheet_workbook())
        ws = load_workspace([path])
        assert ws.files["book.xlsx"].file_type == "excel"


class TestUnsupportedExtension:
    def test_unsupported_extension_raises(self, tmp_path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(UnsupportedFileTypeError):
            load_workspace([path])

    def test_unsupported_extension_hint_lists_supported(self, tmp_path) -> None:
        path = tmp_path / "notes.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            load_workspace([path])
        assert ".csv" in (exc_info.value.hint or "")


class TestDuplicateFilenames:
    def test_same_name_in_different_directories_gets_disambiguated(self, write_csv) -> None:
        path_a = write_csv("report.csv", pd.DataFrame({"v": [1]}), subdir="north")
        path_b = write_csv("report.csv", pd.DataFrame({"v": [2]}), subdir="south")
        ws = load_workspace([path_a, path_b])
        assert set(ws.files) == {"north/report.csv", "south/report.csv"}
        assert ws.files["north/report.csv"].sheets["default"]["v"].iloc[0] == 1
        assert ws.files["south/report.csv"].sheets["default"]["v"].iloc[0] == 2

    def test_unique_names_are_not_disambiguated(self, write_csv) -> None:
        path_a = write_csv("a.csv", pd.DataFrame({"v": [1]}), subdir="north")
        path_b = write_csv("b.csv", pd.DataFrame({"v": [2]}), subdir="south")
        ws = load_workspace([path_a, path_b])
        assert set(ws.files) == {"a.csv", "b.csv"}
