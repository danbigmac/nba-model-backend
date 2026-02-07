#!/usr/bin/env python3
"""
Fetch daily NBA odds from The Odds API and append to a CSV file.

Output columns: date,home,away,total,spread,whos_favored
"""

import argparse
import csv
import json
import os
import statistics
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None

TEAM_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NBA odds and append to CSV.")
    parser.add_argument(
        "--api-key",
        default=os.getenv("ODDS_API_KEY"),
        help="The Odds API key (or set ODDS_API_KEY env var).",
    )
    parser.add_argument(
        "--output",
        default="data/nba_2025-26_live.csv",
        help="CSV file to append to.",
    )
    parser.add_argument(
        "--regions",
        default="us",
        help="Regions to fetch odds for.",
    )
    parser.add_argument(
        "--bookmakers",
        default="",
        help="Optional comma-separated bookmaker keys to restrict.",
    )
    return parser.parse_args()


def _fetch_odds(api_key: str, regions: str, bookmakers: str) -> List[Dict]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    query = urllib.parse.urlencode(params)
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?{query}"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data


def _to_local_date(iso_time: str) -> str:
    dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
    if ZoneInfo is not None:
        dt = dt.astimezone(ZoneInfo("America/New_York"))
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.date().isoformat()

def _today_local_date() -> str:
    if ZoneInfo is not None:
        dt = datetime.now(ZoneInfo("America/New_York"))
    else:
        dt = datetime.now(timezone.utc)
    return dt.date().isoformat()


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _extract_totals(bookmakers: List[Dict]) -> Optional[float]:
    totals = []
    for book in bookmakers:
        for market in book.get("markets", []):
            if market.get("key") != "totals":
                continue
            outcomes = market.get("outcomes", [])
            if not outcomes:
                continue
            point = outcomes[0].get("point")
            if point is None:
                continue
            totals.append(float(point))
    return _median(totals)


def _extract_spread_and_favorite(bookmakers: List[Dict]) -> Tuple[Optional[float], Optional[str]]:
    spreads = []
    favorites = []
    for book in bookmakers:
        for market in book.get("markets", []):
            if market.get("key") != "spreads":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) < 2:
                continue
            try:
                favorite = min(outcomes, key=lambda o: float(o.get("point")))
            except (TypeError, ValueError):
                continue
            point = favorite.get("point")
            if point is None:
                continue
            spreads.append(abs(float(point)))
            favorites.append(favorite.get("name"))
    spread = _median(spreads)
    favorite_name = None
    if favorites:
        favorite_name = max(set(favorites), key=favorites.count)
    return spread, favorite_name


def _normalize_team(name: str) -> Optional[str]:
    if name in TEAM_ABBR:
        return TEAM_ABBR[name]
    return None


def _load_existing_keys(path: str) -> Set[Tuple[str, str, str]]:
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            date = row.get("date")
            home = row.get("home")
            away = row.get("away")
            if date and home and away:
                keys.add((date, home.upper(), away.upper()))
    return keys


def _append_rows(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "home", "away", "total", "spread", "whos_favored"],
        )
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = _parse_args()
    if not args.api_key:
        print("ERROR: Missing API key. Set ODDS_API_KEY or use --api-key.", file=sys.stderr)
        return 1

    events = _fetch_odds(args.api_key, args.regions, args.bookmakers)
    if not isinstance(events, list):
        print("ERROR: Unexpected response from API.", file=sys.stderr)
        return 1

    existing_keys = _load_existing_keys(args.output)
    new_rows = []
    today_local = _today_local_date()

    for event in events:
        home_name = event.get("home_team")
        away_name = event.get("away_team")
        if not home_name or not away_name:
            continue
        home = _normalize_team(home_name)
        away = _normalize_team(away_name)
        if not home or not away:
            continue

        commence_time = event.get("commence_time", "")
        if not commence_time:
            continue
        date = _to_local_date(commence_time)
        if date != today_local:
            continue
        key = (date, home, away)
        if key in existing_keys:
            continue

        bookmakers = event.get("bookmakers", [])
        total = _extract_totals(bookmakers)
        spread, favorite_name = _extract_spread_and_favorite(bookmakers)
        if total is None or spread is None:
            continue

        whos_favored = "unknown"
        if favorite_name == home_name:
            whos_favored = "home"
        elif favorite_name == away_name:
            whos_favored = "away"

        new_rows.append(
            {
                "date": date,
                "home": home,
                "away": away,
                "total": total,
                "spread": spread,
                "whos_favored": whos_favored,
            }
        )

    _append_rows(args.output, new_rows)
    print(f"Appended {len(new_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
