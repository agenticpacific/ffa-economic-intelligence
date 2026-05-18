import base64
import hashlib
import json
import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

try:
    import pydeck as pdk
except ImportError:
    pdk = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None


ROOT = Path(__file__).resolve().parent
IS_BROWSER_RUNTIME = sys.platform == "emscripten"
JSON_DIR = ROOT / "data" / "json"
EEZ_PATH = ROOT / "data" / "eez_v12_pacific.gpkg"
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/ADM0/"
PACIFIC_VIEW = pdk.ViewState(latitude=-8, longitude=178, zoom=2.15, pitch=0) if pdk is not None else None

JSON_DATASET_FILES = [
    "compendium_a_catch_and_catch_value.json",
    "compendium_b_prices.json",
    "compendium_c_country_level_data.json",
    "compendium_contents.json",
    "tuna_10_value_catch_nat_wat.json",
    "tuna_11_value_catch_by_fleet.json",
    "tuna_12_val_fl_cat_nat_wat_ffa_me_.json",
    "tuna_13_value_fl_catch_own_waters.json",
    "tuna_1_contents.json",
    "tuna_2_intoduction_.json",
    "tuna_3_prices.json",
    "tuna_4_summary_of_catch.json",
    "tuna_5_summary_of_catch_value.json",
    "tuna_6_catch_by_national_waters.json",
    "tuna_7_catch_by_fleet.json",
    "tuna_8_fl_cat_nat_wat_ffa_members_.json",
    "tuna_9_fl_cat_own_nat_wat.json",
]

COUNTRY_ISO3 = {
    "Australia": "AUS",
    "Cook Islands": "COK",
    "Federated States of Micronesia": "FSM",
    "Fiji": "FJI",
    "Kiribati": "KIR",
    "Marshall Islands": "MHL",
    "Nauru": "NRU",
    "New Zealand": "NZL",
    "Niue": "NIU",
    "Palau": "PLW",
    "Papua New Guinea": "PNG",
    "Samoa": "WSM",
    "Solomon Islands": "SLB",
    "Tokelau": "TKL",
    "Tonga": "TON",
    "Tuvalu": "TUV",
    "Vanuatu": "VUT",
}

COUNTRY_ALIASES = {
    "Australia (includes Norfolk Island)": "Australia",
    "Cook Islands": "Cook Islands",
    "FSM": "Federated States of Micronesia",
    "Fiji": "Fiji",
    "Kiribati": "Kiribati",
    "Marshall Islands": "Marshall Islands",
    "Nauru": "Nauru",
    "New Zealand": "New Zealand",
    "Niue": "Niue",
    "Palau": "Palau",   
    "PNG": "Papua New Guinea",
    "Papua New Guinea": "Papua New Guinea",
    "Samoa": "Samoa",
    "Solomon Islands": "Solomon Islands",
    "Tokelau": "Tokelau",
    "Tonga": "Tonga",
    "Tuvalu": "Tuvalu",
    "Vanuatu": "Vanuatu",
}

SERIES_ORDER = ["Albacore", "Bigeye", "Skipjack", "Yellowfin", "TOTAL"]

st.set_page_config(
    page_title="FFA Economic Intelligence",
    page_icon=":material/query_stats:",
    layout="wide",
)
st.logo(str(ROOT / "ffa.png"), size="large")

logo_b64 = base64.b64encode((ROOT / "ffa.png").read_bytes()).decode("utf-8")


def clean_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\xa0", " ").strip()
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def normalize_country_name(name):
    text = clean_text(name)
    text = re.sub(r"\s*\(.*?\)", "", text).strip()
    return COUNTRY_ALIASES.get(text, text)


def coerce_number(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = clean_text(value).replace(",", "")
    if not text:
        return None
    number = pd.to_numeric(text, errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def coerce_year(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and 1900 <= float(value) <= 2100:
        return int(float(value))
    text = clean_text(value)
    if not text:
        return None
    if re.fullmatch(r"\d{4}(?:\.0+)?", text):
        year = int(float(text))
        if 1900 <= year <= 2100:
            return year
    return None


def strip_section_prefix(section_name):
    return re.sub(r"^[A-Z]\d+\s+", "", clean_text(section_name))


def pick_default_index(options, target):
    if not options:
        return 0
    try:
        return options.index(target)
    except ValueError:
        return 0


def enforce_total_only(species_values):
    if "TOTAL" in species_values:
        return ["TOTAL"]
    return species_values


def latest_value(dataframe, **filters):
    subset = dataframe.copy()
    for column, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            subset = subset[subset[column].isin(value)]
        else:
            subset = subset[subset[column] == value]
    if subset.empty:
        return None
    return subset.sort_values("year").iloc[-1]["value"]


def sparkline_values(dataframe, **filters):
    subset = dataframe.copy()
    for column, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            subset = subset[subset[column].isin(value)]
        else:
            subset = subset[subset[column] == value]
    if subset.empty:
        return None
    return subset.sort_values("year")["value"].tolist()


def format_value(value, unit=None):
    if value is None or pd.isna(value):
        return "N/A"
    unit_text = clean_text(unit).lower()
    if "us$" in unit_text and "mill" in unit_text:
        return f"US${value:,.1f}m"
    if "us$" in unit_text:
        return f"US${value:,.0f}"
    if "tonne" in unit_text:
        return f"{value:,.0f} t"
    if "number" in unit_text:
        return f"{value:,.0f}"
    if "percent" in unit_text or unit_text == "%":
        return f"{value:,.1f}%"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def format_delta_share(part, whole):
    if part is None or whole in (None, 0):
        return None
    return f"{part / whole:.1%} share"


def color_scale(values):
    valid_values = [value for value in values if value is not None and not pd.isna(value)]
    if not valid_values:
        return lambda _: [96, 108, 122, 90]
    minimum = min(valid_values)
    maximum = max(valid_values)

    def apply(value):
        if value is None or pd.isna(value):
            return [96, 108, 122, 90]
        if maximum == minimum:
            ratio = 1.0
        else:
            ratio = (value - minimum) / (maximum - minimum)
        start = (17, 56, 94)
        end = (238, 179, 61)
        rgb = [int(start[index] + ratio * (end[index] - start[index])) for index in range(3)]
        return rgb + [185]

    return apply


def render_metric(label, value, delta=None, chart_data=None, chart_type="line"):
    metric_kwargs = {"label": label, "value": value, "border": True}
    if delta:
        metric_kwargs["delta"] = delta
    if chart_data:
        metric_kwargs["chart_data"] = chart_data
        metric_kwargs["chart_type"] = chart_type
    st.metric(**metric_kwargs)


def render_line_chart(dataframe, series_field, value_title, value_format=".2f"):
    if dataframe.empty:
        st.caption("No data available for the current filters.")
        return
    chart = (
        alt.Chart(dataframe)
        .mark_line(point=alt.OverlayMarkDef(size=52, filled=True), strokeWidth=2.8)
        .encode(
            x=alt.X("year:Q", title="Year", axis=alt.Axis(format="d", tickMinStep=1)),
            y=alt.Y("value:Q", title=value_title),
            color=alt.Color(f"{series_field}:N", title="", legend=alt.Legend(orient="top")),
            tooltip=[
                alt.Tooltip(f"{series_field}:N", title="Series"),
                alt.Tooltip("year:Q", title="Year", format="d"),
                alt.Tooltip("value:Q", title=value_title, format=value_format),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def render_rank_chart(dataframe, category_field, value_title, value_format=".2f"):
    if dataframe.empty:
        st.caption("No data available for the current filters.")
        return
    chart = (
        alt.Chart(dataframe)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
        .encode(
            y=alt.Y(f"{category_field}:N", sort="-x", title=""),
            x=alt.X("value:Q", title=value_title),
            color=alt.Color("value:Q", legend=None, scale=alt.Scale(scheme="teals")),
            tooltip=[
                alt.Tooltip(f"{category_field}:N", title="Name"),
                alt.Tooltip("value:Q", title=value_title, format=value_format),
            ],
        )
        .properties(height=max(300, len(dataframe) * 28))
    )
    st.altair_chart(chart, width="stretch")


def dataset_file_signature(file_name):
    path = JSON_DIR / file_name
    try:
        stat = path.stat()
        return (file_name, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (file_name,)


def dataset_inventory_signature():
    signatures = []
    for file_name in JSON_DATASET_FILES:
        path = JSON_DIR / file_name
        try:
            stat = path.stat()
            signatures.append((file_name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            signatures.append((file_name,))
    return tuple(signatures)


@st.cache_data(show_spinner=False)
def load_json_dataset(file_name, file_signature):
    return pd.read_json(JSON_DIR / file_name, orient="split")


@st.cache_data(show_spinner=False)
def load_dataset_inventory(inventory_signature):
    rows = []
    for file_name in JSON_DATASET_FILES:
        path = JSON_DIR / file_name
        try:
            dataframe = pd.read_json(path, orient="split")
        except OSError:
            continue
        signature = hashlib.sha1(
            pd.util.hash_pandas_object(dataframe.astype("string"), index=True).values.tobytes()
        ).hexdigest()[:12]
        source = path.stem.split("_", 1)[0]
        rows.append(
            {
                "dataset": path.name,
                "source": source,
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "fingerprint": signature,
            }
        )
    inventory = pd.DataFrame.from_records(
        rows,
        columns=["dataset", "source", "rows", "columns", "fingerprint"],
    )
    if inventory.empty:
        inventory["duplicate_of"] = pd.Series(dtype="string")
        inventory["is_duplicate"] = pd.Series(dtype="bool")
        return inventory
    inventory["duplicate_of"] = inventory.groupby("fingerprint")["dataset"].transform("first")
    inventory["is_duplicate"] = inventory.duplicated("fingerprint", keep="first")
    return inventory


@st.cache_data(show_spinner=False)
def load_dataset_manifest():
    return pd.DataFrame.from_records(
        [
            {
                "dataset": file_name,
                "source": file_name.split("_", 1)[0],
            }
            for file_name in JSON_DATASET_FILES
        ],
        columns=["dataset", "source"],
    )


@st.cache_data(show_spinner=False)
def parse_trend_rows(file_name, file_signature):
    dataframe = load_json_dataset(file_name, file_signature)
    columns = list(dataframe.columns)
    current_section = ""
    years = {}
    records = []

    for _, row in dataframe.iterrows():
        label = clean_text(row[columns[0]])
        candidate_years = {column: coerce_year(row[column]) for column in columns[1:]}
        year_count = sum(year is not None for year in candidate_years.values())

        if year_count >= 5 and (not label or label.lower() == "year"):
            years = {column: year for column, year in candidate_years.items() if year is not None}
            continue

        if not years:
            continue

        numeric_values = {column: coerce_number(row[column]) for column in years}
        if label and all(value is None for value in numeric_values.values()):
            current_section = label
            continue
        if not label or all(value is None for value in numeric_values.values()):
            continue

        for column, year in years.items():
            value = numeric_values[column]
            if value is not None:
                records.append(
                    {
                        "section": current_section,
                        "series": label,
                        "year": year,
                        "value": value,
                    }
                )

    return pd.DataFrame.from_records(
        records,
        columns=["section", "series", "year", "value"],
    ).drop_duplicates()


@st.cache_data(show_spinner=False)
def parse_sectioned_table(file_name, file_signature):
    dataframe = load_json_dataset(file_name, file_signature)
    columns = list(dataframe.columns)
    current_section = ""
    headers = []
    records = []

    for _, row in dataframe.iterrows():
        values = [row[column] for column in columns]
        labels = [clean_text(value) for value in values]
        first = labels[0]

        if first and all(not label for label in labels[1:]):
            current_section = first
            headers = []
            continue

        if first.lower() == "year":
            headers = labels
            continue

        year = coerce_year(values[0])
        if not headers or year is None:
            continue

        record = {"section": current_section, "year": year}
        for index, header in enumerate(headers[1:], start=1):
            if not header or index >= len(values):
                continue
            numeric_value = coerce_number(values[index])
            if numeric_value is not None:
                record[header] = numeric_value
        if len(record) > 2:
            records.append(record)

    parsed = pd.DataFrame.from_records(records)
    if parsed.empty:
        return pd.DataFrame(columns=["section", "year"])
    return parsed.drop_duplicates()


@st.cache_data(show_spinner=False)
def parse_multi_block_table(file_name, file_signature):
    dataframe = load_json_dataset(file_name, file_signature)
    columns = list(dataframe.columns)
    block_labels = {}
    year_map = {}
    current_section = ""
    current_group = ""
    records = []

    for _, row in dataframe.iterrows():
        first = clean_text(row[columns[0]])

        rolling_label = ""
        candidate_labels = {}
        text_count = 0
        for column in columns[1:]:
            text = clean_text(row[column])
            if text and coerce_year(text) is None and coerce_number(text) is None:
                rolling_label = text
                text_count += 1
            if rolling_label:
                candidate_labels[column] = rolling_label

        if text_count >= 2:
            block_labels = candidate_labels
            year_map = {}
            continue

        candidate_years = {column: coerce_year(row[column]) for column in columns[1:]}
        year_count = sum(year is not None for year in candidate_years.values())
        if block_labels and year_count >= 10 and not first:
            year_map = {
                column: {"species": block_labels[column], "year": year}
                for column, year in candidate_years.items()
                if year is not None and column in block_labels
            }
            continue

        if not year_map:
            continue

        numeric_values = {column: coerce_number(row[column]) for column in year_map}
        if first and all(value is None for value in numeric_values.values()):
            if re.match(r"^\d+\.\d+\s+", first):
                current_section = first
                current_group = ""
            else:
                current_group = first
            continue
        if not first or all(value is None for value in numeric_values.values()):
            continue

        for column, metadata in year_map.items():
            value = numeric_values[column]
            if value is not None:
                records.append(
                    {
                        "section": current_section,
                        "group": current_group,
                        "entity": normalize_country_name(first),
                        "species": metadata["species"],
                        "year": metadata["year"],
                        "value": value,
                    }
                )

    return pd.DataFrame.from_records(
        records,
        columns=["section", "group", "entity", "species", "year", "value"],
    ).drop_duplicates()


@st.cache_data(show_spinner=False)
def parse_country_metrics(file_signature):
    dataframe = load_json_dataset("compendium_c_country_level_data.json", file_signature)
    columns = list(dataframe.columns)
    section_pattern = re.compile(r"^[A-Z]\d+\s+(.+?)\s*-\s*(.+)$")

    current_country = ""
    current_section = ""
    current_group = ""
    year_headers = {}
    records = []

    for _, row in dataframe.iterrows():
        label = clean_text(row[columns[0]])
        unit = clean_text(row[columns[1]])
        section_match = section_pattern.match(label)

        if section_match:
            current_country = normalize_country_name(section_match.group(1))
            current_section = clean_text(section_match.group(2))
            current_group = ""
            continue

        candidate_years = {column: coerce_year(row[column]) for column in columns[2:]}
        year_count = sum(year is not None for year in candidate_years.values())
        if current_country and unit.lower() == "units" and year_count >= 5:
            year_headers = {column: year for column, year in candidate_years.items() if year is not None}
            continue

        if not current_country or not year_headers:
            continue

        numeric_values = {column: coerce_number(row[column]) for column in year_headers}
        if label and not unit and all(value is None for value in numeric_values.values()):
            current_group = label
            continue
        if not label or all(value is None for value in numeric_values.values()):
            continue

        for column, year in year_headers.items():
            value = numeric_values[column]
            if value is not None:
                records.append(
                    {
                        "country": current_country,
                        "section": current_section,
                        "group": current_group,
                        "metric": label,
                        "unit": unit,
                        "year": year,
                        "value": value,
                    }
                )

    return pd.DataFrame.from_records(
        records,
        columns=["country", "section", "group", "metric", "unit", "year", "value"],
    ).drop_duplicates()


@st.cache_data(ttl=604800, show_spinner=False)
def load_country_boundaries(country_names):
    if gpd is None:
        raise RuntimeError("geopandas is required for the map layer")

    eez_gdf = load_eez_overlay()
    if eez_gdf is not None and not eez_gdf.empty:
        territory_iso_columns = [column for column in ["ISO_TER1", "ISO_TER2", "ISO_TER3"] if column in eez_gdf.columns]
        sovereign_iso_columns = [column for column in ["ISO_SOV1", "ISO_SOV2", "ISO_SOV3"] if column in eez_gdf.columns]
        territory_name_columns = [column for column in ["TERRITORY1", "TERRITORY2", "TERRITORY3"] if column in eez_gdf.columns]
        sovereign_name_columns = [column for column in ["SOVEREIGN1", "SOVEREIGN2", "SOVEREIGN3"] if column in eez_gdf.columns]

        derived_frames = []
        for country in country_names:
            iso3 = COUNTRY_ISO3[country]
            matches = pd.Series(False, index=eez_gdf.index)

            for column in territory_iso_columns:
                matches |= eez_gdf[column].fillna("").eq(iso3)

            if not matches.any():
                for column in sovereign_iso_columns:
                    matches |= eez_gdf[column].fillna("").eq(iso3)

            if not matches.any():
                normalized_country = normalize_country_name(country)
                for column in territory_name_columns:
                    matches |= eez_gdf[column].fillna("").map(normalize_country_name).eq(normalized_country)

            if not matches.any():
                normalized_country = normalize_country_name(country)
                for column in sovereign_name_columns:
                    matches |= eez_gdf[column].fillna("").map(normalize_country_name).eq(normalized_country)

            if not matches.any():
                continue

            dissolved = eez_gdf.loc[matches, ["geometry"]].dissolve().reset_index(drop=True)
            dissolved["country"] = country
            dissolved["iso3"] = iso3
            dissolved["boundary_name"] = country
            derived_frames.append(dissolved[["country", "iso3", "boundary_name", "geometry"]])

        if derived_frames:
            boundaries = gpd.GeoDataFrame(
                pd.concat(derived_frames, ignore_index=True),
                geometry="geometry",
                crs=eez_gdf.crs,
            ).to_crs("EPSG:4326")
            return boundaries[["country", "iso3", "boundary_name", "geometry"]], [
                "Local EEZ geopackage (territory geometries)"
            ]

    features = []
    sources = []
    for country in country_names:
        iso3 = COUNTRY_ISO3[country]
        metadata_response = requests.get(GEOBOUNDARIES_API.format(iso3=iso3), timeout=30)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        geojson_url = metadata.get("simplifiedGeometryGeoJSON") or metadata.get("gjDownloadURL")
        geometry_response = requests.get(geojson_url, timeout=30)
        geometry_response.raise_for_status()
        geometry = geometry_response.json()
        sources.append(f"{country}: geoBoundaries gbOpen ADM0 ({metadata.get('buildDate', 'current')})")
        for feature in geometry.get("features", []):
            properties = feature.setdefault("properties", {})
            properties["country"] = country
            properties["iso3"] = iso3
            properties["boundary_name"] = metadata.get("boundaryName", country)
            features.append(feature)

    boundaries = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    return boundaries[["country", "iso3", "boundary_name", "geometry"]], sorted(set(sources))


@st.cache_data(show_spinner=False)
def load_eez_overlay():
    if gpd is None or not EEZ_PATH.exists():
        return None
    layer_name = None
    try:
        layer_info = gpd.list_layers(EEZ_PATH)
        if len(layer_info):
            layer_name = layer_info.iloc[0]["name"]
    except Exception:
        layer_name = None

    if layer_name:
        eez = gpd.read_file(EEZ_PATH, layer=layer_name)
    else:
        eez = gpd.read_file(EEZ_PATH)
    return eez.to_crs("EPSG:4326")


tuna_catch_summary = parse_trend_rows(
    "tuna_4_summary_of_catch.json",
    dataset_file_signature("tuna_4_summary_of_catch.json"),
)
tuna_value_summary = parse_trend_rows(
    "tuna_5_summary_of_catch_value.json",
    dataset_file_signature("tuna_5_summary_of_catch_value.json"),
)
global_catch_table = parse_sectioned_table(
    "compendium_a_catch_and_catch_value.json",
    dataset_file_signature("compendium_a_catch_and_catch_value.json"),
)
price_table = parse_sectioned_table(
    "compendium_b_prices.json",
    dataset_file_signature("compendium_b_prices.json"),
)
national_waters_catch = parse_multi_block_table(
    "tuna_6_catch_by_national_waters.json",
    dataset_file_signature("tuna_6_catch_by_national_waters.json"),
)
fleet_catch = parse_multi_block_table(
    "tuna_7_catch_by_fleet.json",
    dataset_file_signature("tuna_7_catch_by_fleet.json"),
)
national_waters_value = parse_multi_block_table(
    "tuna_10_value_catch_nat_wat.json",
    dataset_file_signature("tuna_10_value_catch_nat_wat.json"),
)
fleet_value = parse_multi_block_table(
    "tuna_11_value_catch_by_fleet.json",
    dataset_file_signature("tuna_11_value_catch_by_fleet.json"),
)
country_metrics = parse_country_metrics(
    dataset_file_signature("compendium_c_country_level_data.json")
)

national_waters_catch_totals = national_waters_catch[
    national_waters_catch["section"].str.contains("All gears", na=False)
].copy()
national_waters_value_totals = national_waters_value[
    national_waters_value["section"].str.contains("All gears", na=False)
].copy()

available_years = sorted(
    set(tuna_catch_summary["year"]) | set(national_waters_catch["year"]) | set(country_metrics["year"])
)
year_min = int(min(available_years))
year_max = int(max(available_years))

gear_options = [
    series
    for series in tuna_catch_summary["series"].dropna().unique().tolist()
    if series != "GRAND TOTAL"
]
country_options = sorted(
    country
    for country in country_metrics["country"].dropna().unique().tolist()
    if country in COUNTRY_ISO3
)
species_options = [
    species
    for species in SERIES_ORDER
    if species in national_waters_catch["species"].dropna().unique().tolist()
]
price_species_lookup = {
    strip_section_prefix(section): section for section in price_table["section"].dropna().unique().tolist()
}
price_species_options = sorted(price_species_lookup)

with st.sidebar:
    st.markdown("### Filters")
    selected_years = st.slider(
        "Time horizon",
        min_value=year_min,
        max_value=year_max,
        value=(max(1997, year_min), year_max),
    )
    selected_gears = st.multiselect(
        "Gear focus",
        options=gear_options,
        default=gear_options,
    )
    selected_species = st.multiselect(
        "Species focus",
        options=species_options,
        default=["TOTAL"] if "TOTAL" in species_options else species_options[:2],
    )
    selected_country = st.selectbox(
        "Country focus",
        options=country_options,
        index=pick_default_index(country_options, "Fiji"),
    )
    map_metric = st.radio(
        "Map layer",
        options=["Catch volume", "Catch value"],
        horizontal=False,
    )
    show_eez = st.toggle("Show EEZ overlay", value=True)
    st.caption("If TOTAL is selected, the dashboard uses the aggregate series only to avoid double counting.")

start_year, end_year = selected_years
species_focus = enforce_total_only(selected_species or species_options)

tuna_catch_filtered = tuna_catch_summary[
    tuna_catch_summary["year"].between(start_year, end_year)
    & tuna_catch_summary["series"].isin(selected_gears)
]
tuna_value_filtered = tuna_value_summary[
    tuna_value_summary["year"].between(start_year, end_year)
    & tuna_value_summary["series"].isin(selected_gears)
]

regional_catch_total = latest_value(tuna_catch_summary, series="GRAND TOTAL")
regional_value_total = latest_value(tuna_value_summary, series="GRAND TOTAL")
regional_catch_trend = sparkline_values(
    tuna_catch_summary[tuna_catch_summary["year"].between(start_year, end_year)],
    series="GRAND TOTAL",
)
regional_value_trend = sparkline_values(
    tuna_value_summary[tuna_value_summary["year"].between(start_year, end_year)],
    series="GRAND TOTAL",
)

country_catch_series = national_waters_catch_totals[
    national_waters_catch_totals["entity"].eq(selected_country)
    & national_waters_catch_totals["species"].isin(species_focus)
    & national_waters_catch_totals["year"].between(start_year, end_year)
]
country_value_series = national_waters_value_totals[
    national_waters_value_totals["entity"].eq(selected_country)
    & national_waters_value_totals["species"].isin(species_focus)
    & national_waters_value_totals["year"].between(start_year, end_year)
]

country_catch_by_year = country_catch_series.groupby("year", as_index=False)["value"].sum()
country_value_by_year = country_value_series.groupby("year", as_index=False)["value"].sum()
selected_country_catch = latest_value(country_catch_by_year) if not country_catch_by_year.empty else None
selected_country_value = latest_value(country_value_by_year) if not country_value_by_year.empty else None

catch_year_data = national_waters_catch_totals[
    national_waters_catch_totals["year"].eq(end_year) & national_waters_catch_totals["species"].isin(species_focus)
]
catch_year_data = catch_year_data[catch_year_data["entity"].isin(country_options)]
catch_year_data = catch_year_data.groupby("entity", as_index=False)["value"].sum()
catch_year_total = catch_year_data["value"].sum() if not catch_year_data.empty else None

value_year_data = national_waters_value_totals[
    national_waters_value_totals["year"].eq(end_year) & national_waters_value_totals["species"].isin(species_focus)
]
value_year_data = value_year_data[value_year_data["entity"].isin(country_options)]
value_year_data = value_year_data.groupby("entity", as_index=False)["value"].sum()
value_year_total = value_year_data["value"].sum() if not value_year_data.empty else None

map_year_data = catch_year_data if map_metric == "Catch volume" else value_year_data
map_unit = "tonnes" if map_metric == "Catch volume" else "US$"
map_value_format = ",.0f" if map_metric == "Catch volume" else ",.1f"

global_catch = global_catch_table[
    global_catch_table["section"].str.contains("Global catch by Ocean", na=False)
    & global_catch_table["year"].between(start_year, end_year)
].copy()
global_catch_long = global_catch.melt(
    id_vars=["section", "year"],
    value_vars=[
        column
        for column in global_catch.columns
        if column not in {"section", "year", "Total"}
    ],
    var_name="ocean",
    value_name="value",
).dropna(subset=["value"])

top_waters = map_year_data.sort_values("value", ascending=False).head(10).rename(columns={"entity": "country"})
fleet_focus = fleet_catch[
    fleet_catch["year"].eq(end_year) & fleet_catch["species"].isin(species_focus)
]
fleet_focus = fleet_focus.groupby("entity", as_index=False)["value"].sum().sort_values("value", ascending=False).head(10)

country_section_options = sorted(
    country_metrics[country_metrics["country"].eq(selected_country)]["section"].dropna().unique().tolist()
)
selected_country_section = st.session_state.get("selected_country_section", "Catch and catch values")
if selected_country_section not in country_section_options and country_section_options:
    selected_country_section = country_section_options[pick_default_index(country_section_options, "Catch and catch values")]

country_section_data = country_metrics[
    country_metrics["country"].eq(selected_country)
    & country_metrics["section"].eq(selected_country_section)
    & country_metrics["year"].between(start_year, end_year)
].copy()
country_metric_options = (
    country_section_data[["group", "metric", "unit"]]
    .drop_duplicates()
    .assign(
        option=lambda dataframe: dataframe.apply(
            lambda row: " / ".join(part for part in [clean_text(row["group"]), clean_text(row["metric"])] if part)
            + (f" ({row['unit']})" if clean_text(row["unit"]) else ""),
            axis=1,
        )
    )
)
country_metric_option_labels = country_metric_options["option"].tolist()
selected_country_metric_label = st.session_state.get("selected_country_metric_label")
if selected_country_metric_label not in country_metric_option_labels and country_metric_option_labels:
    selected_country_metric_label = country_metric_option_labels[0]

if selected_country_metric_label:
    selected_country_metric_row = country_metric_options[
        country_metric_options["option"].eq(selected_country_metric_label)
    ].iloc[0]
    country_metric_series = country_section_data[
        country_section_data["group"].eq(selected_country_metric_row["group"])
        & country_section_data["metric"].eq(selected_country_metric_row["metric"])
        & country_section_data["unit"].eq(selected_country_metric_row["unit"])
    ].copy()
else:
    country_metric_series = pd.DataFrame(columns=["year", "value"])

country_latest_year = None
if not country_section_data.empty:
    country_latest_year = int(country_section_data["year"].max())
country_latest_table = country_section_data[country_section_data["year"].eq(country_latest_year)].copy()
country_latest_table["indicator"] = country_latest_table.apply(
    lambda row: " / ".join(part for part in [clean_text(row["group"]), clean_text(row["metric"])] if part),
    axis=1,
)
country_latest_table["latest_value"] = country_latest_table.apply(
    lambda row: format_value(row["value"], row["unit"]),
    axis=1,
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:opsz,wght@8..60,600&display=swap');

.stApp {{
    background:
        radial-gradient(circle at top left, rgba(25, 108, 163, 0.14), transparent 30%),
        radial-gradient(circle at top right, rgba(240, 180, 45, 0.16), transparent 24%),
        linear-gradient(180deg, #f4f7fb 0%, #eef4f8 42%, #f8fafc 100%);
}}

.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}}

.hero-shell {{
    display: grid;
    grid-template-columns: 220px minmax(0, 1.45fr) minmax(260px, 0.9fr);
    gap: 1.4rem;
    align-items: stretch;
    margin-bottom: 1.3rem;
}}

.hero-card, .hero-meta {{
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(18, 58, 92, 0.10);
    border-radius: 24px;
    box-shadow: 0 22px 48px rgba(18, 58, 92, 0.08);
    backdrop-filter: blur(14px);
}}

.hero-card {{
    padding: 1.5rem;
}}

.hero-logo {{
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 220px;
    padding: 1.9rem;
    overflow: hidden;
    background: transparent;
    border: 0;
    box-shadow: none;
    backdrop-filter: none;
}}

.hero-logo::before {{
    content: none;
}}

.hero-logo img {{
    position: relative;
    width: 100%;
    max-width: 168px;
    height: auto;
    object-fit: contain;
    filter: drop-shadow(0 16px 26px rgba(6, 24, 43, 0.22));
}}

.eyebrow {{
    margin: 0;
    color: #1f6aa5;
    font: 700 0.78rem/1.2 'Space Grotesk', sans-serif;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}

.hero-card h1 {{
    margin: 0.45rem 0 0;
    color: #0f2740;
    font: 600 2.5rem/1.02 'Source Serif 4', serif;
}}

.hero-card p {{
    margin: 0.85rem 0 0;
    color: #35536d;
    font: 500 1rem/1.65 'Space Grotesk', sans-serif;
}}

.hero-meta {{
    padding: 1.25rem 1.35rem;
    display: grid;
    gap: 1rem;
}}

.hero-meta div {{
    padding-bottom: 0.9rem;
    border-bottom: 1px solid rgba(18, 58, 92, 0.08);
}}

.hero-meta div:last-child {{
    border-bottom: 0;
    padding-bottom: 0;
}}

.hero-meta span {{
    display: block;
    color: #5f7488;
    font: 500 0.83rem/1.2 'Space Grotesk', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.hero-meta strong {{
    display: block;
    margin-top: 0.35rem;
    color: #0f2740;
    font: 600 1rem/1.45 'Space Grotesk', sans-serif;
}}

@media (max-width: 980px) {{
    .hero-shell {{
        grid-template-columns: 1fr;
    }}

    .hero-logo {{
        min-height: 180px;
    }}

    .hero-logo img {{
        max-width: 124px;
    }}
}}
</style>

<br/>

<div class="hero-shell">
    <div class="hero-card hero-logo">
        <img  src="data:image/png;base64,{logo_b64}" alt="FFA logo" />
    </div>
    <div class="hero-card">
        <p class="eyebrow">Agentic Pacific Prototype</p>
        <h1>FFA Economic Intelligence</h1>
        <p>ECONOMIC AND DEVELOPMENT INDICATORS AND STATISTICS for TUNA FISHERIES OF THE WESTERN AND CENTRAL PACIFIC OCEAN</p>
    </div>
    <div class="hero-meta">
        <div>
            <span>Datasets in use</span>
            <strong>{len(JSON_DATASET_FILES)} JSON datasets</strong>
        </div>
        <div>
            <span>Time coverage</span>
            <strong>{year_min} to {year_max}</strong>
        </div>
        <div>
            <span>Spatial layers</span>
            <strong>Local EEZ geopackage</strong>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    "Agentic Pacific - www.agenticpacific.com.fj"
)

with st.container(horizontal=True):
    render_metric(
        f"Regional catch, {year_max}",
        format_value(regional_catch_total, "tonnes"),
        chart_data=regional_catch_trend,
    )
    render_metric(
        f"Regional catch value, {year_max}",
        format_value(regional_value_total, "US$ mill"),
        chart_data=regional_value_trend,
    )
    render_metric(
        f"{selected_country} catch, {end_year}",
        format_value(selected_country_catch, "tonnes"),
        delta=format_delta_share(selected_country_catch, catch_year_total),
        chart_data=country_catch_by_year["value"].tolist() if not country_catch_by_year.empty else None,
    )
    render_metric(
        f"{selected_country} catch value, {end_year}",
        format_value(selected_country_value, "US$"),
        delta=format_delta_share(selected_country_value, value_year_total),
        chart_data=country_value_by_year["value"].tolist() if not country_value_by_year.empty else None,
    )

overview_tab, country_tab, spatial_tab, data_tab = st.tabs(
    ["Regional Overview", "Country Explorer", "Spatial Context", "Dataset Inventory"]
)

with overview_tab:
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Catch by gear")
            render_line_chart(tuna_catch_filtered, "series", "Metric tonnes", ",.0f")
    with col2:
        with st.container(border=True):
            st.subheader("Catch value by gear")
            render_line_chart(tuna_value_filtered, "series", "US$ millions", ",.1f")

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.subheader("Global catch by ocean")
            render_line_chart(global_catch_long, "ocean", "'000 metric tonnes", ",.1f")
    with col4:
        with st.container(border=True):
            st.subheader("Indicative tuna prices")
            selected_price_species_default = st.session_state.get("selected_price_species", "Albacore")
            if selected_price_species_default not in price_species_options:
                selected_price_species_default = price_species_options[0]
            selected_price_species = st.selectbox(
                "Price series",
                options=price_species_options,
                index=pick_default_index(price_species_options, selected_price_species_default),
                key="selected_price_species",
            )
            active_price_section = price_species_lookup[selected_price_species]
            active_price_long = price_table[
                price_table["section"].eq(active_price_section)
                & price_table["year"].between(start_year, end_year)
            ].melt(
                id_vars=["section", "year"],
                value_vars=[column for column in price_table.columns if column not in {"section", "year"}],
                var_name="market",
                value_name="value",
            ).dropna(subset=["value"])
            render_line_chart(active_price_long, "market", "US$/mt", ",.0f")

    col5, col6 = st.columns(2)
    with col5:
        with st.container(border=True):
            st.subheader(f"Top FFA national waters, {end_year}")
            render_rank_chart(top_waters, "country", map_metric, map_value_format)
    with col6:
        with st.container(border=True):
            st.subheader(f"Top fleets, {end_year}")
            render_rank_chart(fleet_focus, "entity", "Metric tonnes", ",.0f")

with country_tab:
    filters_col, latest_col = st.columns([1.2, 1.8])
    with filters_col:
        with st.container(border=True):
            st.subheader(selected_country)
            selected_country_section = st.selectbox(
                "Compendium section",
                options=country_section_options,
                index=pick_default_index(country_section_options, selected_country_section),
                key="selected_country_section",
            )
            country_section_data = country_metrics[
                country_metrics["country"].eq(selected_country)
                & country_metrics["section"].eq(selected_country_section)
                & country_metrics["year"].between(start_year, end_year)
            ].copy()
            country_metric_options = (
                country_section_data[["group", "metric", "unit"]]
                .drop_duplicates()
                .assign(
                    option=lambda dataframe: dataframe.apply(
                        lambda row: " / ".join(part for part in [clean_text(row["group"]), clean_text(row["metric"])] if part)
                        + (f" ({row['unit']})" if clean_text(row["unit"]) else ""),
                        axis=1,
                    )
                )
            )
            country_metric_option_labels = country_metric_options["option"].tolist()
            selected_country_metric_label = st.selectbox(
                "Indicator",
                options=country_metric_option_labels,
                index=pick_default_index(country_metric_option_labels, selected_country_metric_label),
                key="selected_country_metric_label",
            )
            if selected_country_metric_label:
                selected_country_metric_row = country_metric_options[
                    country_metric_options["option"].eq(selected_country_metric_label)
                ].iloc[0]
                country_metric_series = country_section_data[
                    country_section_data["group"].eq(selected_country_metric_row["group"])
                    & country_section_data["metric"].eq(selected_country_metric_row["metric"])
                    & country_section_data["unit"].eq(selected_country_metric_row["unit"])
                ].copy()
                render_line_chart(
                    country_metric_series,
                    "metric",
                    clean_text(selected_country_metric_row["unit"] or "Value"),
                    ",.1f",
                )
            else:
                st.caption("No country indicators available for the selected section.")
    with latest_col:
        with st.container(border=True):
            st.subheader("Latest country snapshot")
            if country_section_data.empty:
                st.caption("No country indicators available for the selected filters.")
            else:
                country_latest_year = int(country_section_data["year"].max())
                country_latest_table = country_section_data[country_section_data["year"].eq(country_latest_year)].copy()
                country_latest_table["indicator"] = country_latest_table.apply(
                    lambda row: " / ".join(
                        part for part in [clean_text(row["group"]), clean_text(row["metric"])] if part
                    ),
                    axis=1,
                )
                country_latest_table["latest value"] = country_latest_table.apply(
                    lambda row: format_value(row["value"], row["unit"]),
                    axis=1,
                )
                st.caption(f"Latest observed year: {country_latest_year}")
                st.dataframe(
                    country_latest_table[["indicator", "latest value", "unit"]],
                    hide_index=True,
                    width="stretch",
                )

with spatial_tab:
    with st.container(border=True):
        st.subheader(f"Pacific national waters context, {end_year}")
        if IS_BROWSER_RUNTIME:
            st.info(
                "The browser build uses a lightweight spatial fallback so the packaged app can finish loading. "
                "Open the local Streamlit app for the full interactive map.",
                icon=":material/travel_explore:",
            )
            st.dataframe(
                top_waters.assign(value=lambda dataframe: dataframe["value"].map(lambda value: format_value(value, map_unit))),
                hide_index=True,
                width="stretch",
            )
        elif gpd is None or pdk is None:
            st.warning("Install geopandas and pydeck to enable the spatial overlay view.", icon=":material/map:")
        else:
            try:
                boundary_gdf, boundary_sources = load_country_boundaries(tuple(country_options))
                boundary_gdf = boundary_gdf.merge(
                    map_year_data.rename(columns={"entity": "country"}),
                    on="country",
                    how="left",
                )
                fill = color_scale(boundary_gdf["value"].tolist())
                boundary_gdf["fill_color"] = boundary_gdf["value"].apply(fill)
                boundary_gdf["line_color"] = boundary_gdf["country"].apply(
                    lambda name: [244, 180, 62, 255] if name == selected_country else [23, 61, 97, 180]
                )
                boundary_gdf["line_width"] = boundary_gdf["country"].apply(lambda name: 3 if name == selected_country else 1.2)
                boundary_gdf["metric_label"] = map_metric
                boundary_gdf["display_value"] = boundary_gdf["value"].apply(lambda value: format_value(value, map_unit))

                layers = [
                    pdk.Layer(
                        "GeoJsonLayer",
                        json.loads(boundary_gdf.to_json()),
                        pickable=True,
                        filled=True,
                        stroked=True,
                        get_fill_color="properties.fill_color",
                        get_line_color="properties.line_color",
                        get_line_width="properties.line_width",
                        line_width_min_pixels=1,
                    )
                ]

                if show_eez:
                    eez_gdf = load_eez_overlay()
                    if eez_gdf is not None and not eez_gdf.empty:
                        boundary_union = (
                            boundary_gdf.geometry.union_all()
                            if hasattr(boundary_gdf.geometry, "union_all")
                            else boundary_gdf.geometry.unary_union
                        )
                        relevant_eez = eez_gdf[eez_gdf.geometry.intersects(boundary_union)].copy()
                        relevant_eez["geometry"] = relevant_eez.geometry.simplify(0.04, preserve_topology=True)
                        layers.append(
                            pdk.Layer(
                                "GeoJsonLayer",
                                json.loads(relevant_eez.to_json()),
                                pickable=False,
                                filled=False,
                                stroked=True,
                                get_line_color=[27, 103, 153, 140],
                                line_width_min_pixels=1,
                            )
                        )

                st.pydeck_chart(
                    pdk.Deck(
                        layers=layers,
                        initial_view_state=PACIFIC_VIEW,
                        map_style=None,
                        tooltip={
                            "html": "<b>{country}</b><br/>{metric_label}: {display_value}",
                            "style": {
                                "backgroundColor": "rgba(15, 39, 64, 0.92)",
                                "color": "white",
                                "fontFamily": "Space Grotesk, sans-serif",
                            },
                        },
                    ),
                    height=620,
                )
                source_summary = " | ".join(boundary_sources[:4]) + (" ..." if len(boundary_sources) > 4 else "")
                caption_parts = []
                if source_summary:
                    caption_parts.append(f"Spatial source: {source_summary}.")
                if show_eez:
                    caption_parts.append("EEZ source: local geopackage in data/eez_v12_pacific.gpkg.")
                st.caption(" ".join(caption_parts))
                st.dataframe(
                    top_waters.assign(value=lambda dataframe: dataframe["value"].map(lambda value: format_value(value, map_unit))),
                    hide_index=True,
                    width="stretch",
                )
            except Exception as exc:
                st.warning(f"Map layers could not be loaded: {exc}", icon=":material/travel_explore:")

with data_tab:
    browser_inventory = load_dataset_manifest() if IS_BROWSER_RUNTIME else None
    inventory = None
    unique_inventory = None
    if not IS_BROWSER_RUNTIME:
        inventory = load_dataset_inventory(dataset_inventory_signature())
        unique_inventory = inventory[~inventory["is_duplicate"]].copy()

    col7, col8 = st.columns([1.1, 1.3])
    with col7:
        with st.container(border=True):
            st.subheader("Dataset registry")
            if IS_BROWSER_RUNTIME:
                st.caption(
                    f"{len(browser_inventory)} packaged datasets are listed. Browser mode uses a lightweight registry to keep stlite responsive."
                )
                st.dataframe(
                    browser_inventory[["dataset", "source"]],
                    hide_index=True,
                    width="stretch",
                )
            else:
                duplicate_count = int(inventory["is_duplicate"].sum())
                st.caption(
                    f"{len(unique_inventory)} unique datasets are shown. {duplicate_count} duplicates are suppressed by content fingerprint."
                )
                st.dataframe(
                    unique_inventory[["dataset", "source", "rows", "columns", "fingerprint"]],
                    hide_index=True,
                    width="stretch",
                )
    with col8:
        with st.container(border=True):
            st.subheader("Dataset preview")
            preview_options = browser_inventory["dataset"].tolist() if IS_BROWSER_RUNTIME else unique_inventory["dataset"].tolist()
            if IS_BROWSER_RUNTIME:
                st.caption("Preview loads one packaged JSON file at a time in the browser build.")
            preview_dataset = st.selectbox(
                "Preview a dataset",
                options=preview_options,
                key="preview_dataset",
            )
            preview_frame = load_json_dataset(
                preview_dataset,
                dataset_file_signature(preview_dataset),
            ).head(25)
            st.dataframe(preview_frame, hide_index=True, width="stretch")
