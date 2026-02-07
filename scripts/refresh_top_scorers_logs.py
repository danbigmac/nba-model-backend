#!/usr/bin/env python3
"""
Refresh current season logs for top scorers and report stale cache entries.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from nbacache import CacheManager  # noqa: E402
from nbastats import (  # noqa: E402
    CURRENT_SEASON_TTL_SECONDS,
    PAST_SEASON_TTL_SECONDS,
    get_current_season_label,
    get_player_log_metadata,
    get_player_season_log,
    get_top_scorers,
    get_all_players,
    is_current_season,
)


def _local_today() -> datetime.date:
    if ZoneInfo is not None:
        now = datetime.now(ZoneInfo("America/New_York"))
    else:
        now = datetime.now(timezone.utc)
    return now.date()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh top scorer logs for a season.")
    parser.add_argument("--season", default=None, help="Season label like 2025-26 (default: current season).")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top scorers to refresh.")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between players.")
    parser.add_argument("--per-mode", default="PerGame", help="League leaders per-mode (PerGame, Totals, etc).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    season = args.season or get_current_season_label()
    yesterday = _local_today() - timedelta(days=1)
    yesterday_str = yesterday.isoformat()

    with CacheManager() as cache:
        top_df = get_top_scorers(cache, season, top_n=args.top_n, per_mode=args.per_mode)
        all_players = get_all_players(cache)
        top_ids = set(top_df["player_id"].tolist())
        players = [p for p in all_players if p["id"] in top_ids]

        print(f"Refreshing logs for {len(players)} players (season={season}).")

        stale_players = []
        errors = 0
        cache_hits = 0
        refreshed = 0

        for idx, player in enumerate(players, start=1):
            pid = player["id"]
            df_cached, last_updated = cache.load_player_logs(pid, season)
            if last_updated is None:
                last_updated = 0
            ttl_seconds = CURRENT_SEASON_TTL_SECONDS if is_current_season(season) else PAST_SEASON_TTL_SECONDS
            needs_refresh = df_cached is None or (time.time() - last_updated) > ttl_seconds

            try:
                df = get_player_season_log(cache, player, season)
                last_game_date, row_count = get_player_log_metadata(df)
                if last_game_date and last_game_date < yesterday_str:
                    stale_players.append((player["full_name"], last_game_date, row_count))
                if needs_refresh:
                    refreshed += 1
                else:
                    cache_hits += 1
            except Exception as exc:
                errors += 1
                print(f"[ERROR] {player['full_name']} ({season}): {exc}")

            if args.sleep:
                time.sleep(args.sleep)

            if idx % 25 == 0:
                print(f"...processed {idx}/{len(players)} players")

        print("Summary:")
        print(f"- refreshed: {refreshed}")
        print(f"- cache_hits: {cache_hits}")
        print(f"- errors: {errors}")
        if stale_players:
            print(f"- stale (last_game_date < {yesterday_str}): {len(stale_players)}")
            for name, last_game_date, row_count in stale_players[:15]:
                print(f"  {name}: last_game_date={last_game_date}, rows={row_count}")
        else:
            print(f"- stale (last_game_date < {yesterday_str}): 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
