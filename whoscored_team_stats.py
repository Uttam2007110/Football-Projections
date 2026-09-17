"""
Created on Wed Sep 02 20:34:50 2026
Extract every data table from a WhoScored team-statistics stage page
@author: Subramanya.Ganti
"""

#%% imports
from __future__ import annotations

import numpy as np
import pandas as pd 
pd.options.mode.chained_assignment = None

import argparse
import ast
import json
import re
import ssl
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import urllib3

try:
    # Corporate TLS-inspecting proxies (Zscaler/Netskope/...) re-sign traffic with
    # a private root CA that Python's bundled certifi store does not know about,
    # causing CERTIFICATE_VERIFY_FAILED. truststore makes Python validate against
    # the OS trust store (which *does* trust that CA), keeping verification ON.
    import truststore

    truststore.inject_into_ssl()
    _TRUSTSTORE = True
except Exception:  # noqa: BLE001 - optional dependency, never fatal
    _TRUSTSTORE = False

try:
    import pandas as pd
except ImportError:  # DataFrame output needs pandas; raw dicts still work without it
    pd = None

try:
    # curl_cffi mimics a real Chrome TLS fingerprint; useful if plain requests
    # starts getting served Cloudflare challenge pages instead of data.
    from curl_cffi import requests as requests_impl

    _IMPERSONATE = {"impersonate": "chrome"}
except ImportError:
    import requests as requests_impl

    _IMPERSONATE = {}

#%% constraints
def build_url(
    region_id: int | str,
    tournament_id: int | str,
    season_id: int | str,
    stage_id: int | str,
    slug: str,
) -> str:
    """Build a WhoScored stage team-statistics URL from its component ids."""
    return (
        f"https://www.whoscored.com/regions/{region_id}"
        f"/tournaments/{tournament_id}/seasons/{season_id}"
        f"/stages/{stage_id}/teamstatistics/{slug}"
    )


def build_stage_url(
    region_id: int | str,
    tournament_id: int | str,
    season_id: int | str,
    stage_id: int | str,
    slug: str,
) -> str:
    """Build the stage "show" url, which carries that stage's standings.

    e.g. .../seasons/9159/stages/21087/show/italy-serie-a-2022-2023

    Preferred over the season landing url: a season can have several stages
    (2022/23 Serie A has "Serie A" and "Serie A Relegation Playoff") and the
    season url silently serves whichever stage WhoScored defaults to -- the
    playoff, which carries no standings at all.
    """
    return (
        f"https://www.whoscored.com/regions/{region_id}"
        f"/tournaments/{tournament_id}/seasons/{season_id}"
        f"/stages/{stage_id}/show/{slug}"
    )


def build_season_url(
    region_id: int | str,
    tournament_id: int | str,
    season_id: int | str,
    slug: str,
) -> str:
    """Build the season landing url, which carries the league standings.

    e.g. https://www.whoscored.com/regions/108/tournaments/5/seasons/10732/italy-serie-a
    (WhoScored routes on the ids, so either the short or the dated slug works).
    """
    return (
        f"https://www.whoscored.com/regions/{region_id}"
        f"/tournaments/{tournament_id}/seasons/{season_id}/{slug}"
    )


def parse_url(url: str) -> dict:
    """Inverse of build_url: pull the component ids back out of a page url."""
    m = re.search(
        r"/regions/(?P<region_id>\d+)"
        r"/tournaments/(?P<tournament_id>\d+)"
        r"/seasons/(?P<season_id>\d+)"
        r"/stages/(?P<stage_id>\d+)"
        r"/teamstatistics/(?P<slug>[^/?#]+)",
        url,
    )
    if not m:
        raise ValueError(f"Not a WhoScored stage teamstatistics url: {url!r}")
    parts = m.groupdict()
    return {k: (v if k == "slug" else int(v)) for k, v in parts.items()}


def get_config(
    region_id: int | str,
    tournament_id: int | str,
    season_id: int | str,
    stage_id: int | str,
    slug: str,
) -> SimpleNamespace:
    """Return every constant the scraper needs, as a namespace.

    Call this wherever a constant is required instead of reading module-level
    globals. The competition/season ids are supplied by the caller (main), so
    DEFAULT_URL is interpolated per run rather than hardcoded:

        cfg = get_config(108, 5, 10732, 24500, "italy-serie-a-2025-2026")
        cfg.DEFAULT_URL, cfg.UA, cfg.DETAILED_CATEGORIES, ...
    """
    return SimpleNamespace(
        DEFAULT_URL=build_url(region_id, tournament_id, season_id, stage_id, slug),
        SEASON_URL=build_season_url(region_id, tournament_id, season_id, slug),
        STAGE_URL=build_stage_url(region_id, tournament_id, season_id, stage_id, slug),
        REGION_ID=region_id,
        TOURNAMENT_ID=tournament_id,
        SEASON_ID=season_id,
        STAGE_ID=stage_id,
        SLUG=slug,
        UA=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        # --- "Summary" tab table (top grid, rating/possession/pass overview) -
        SUMMARY_TABLE=("summaryteam", "all", "rating"),
        # --- "Defensive" tab table (Overall subfilter: tackles/interceptions/
        # fouls/offsides/shots-conceded per game) --------------------------
        DEFENSIVE_TABLE=("summaryteam", "defensive", "tacklePerGame"),
        # --- "Detailed" tab tables: (category, [subcategories to fetch]) -----
        # WhoScored's Detailed tab has one category dropdown and, for most
        # categories, a subcategory dropdown that defaults to its first entry.
        # "shots" is the only category the site (and this scraper) pulls every
        # subcategory for; every other category is fetched with just its
        # default (first-listed) subcategory. The 3rd dropdown -- Per90 /
        # PerGame / Total -- is fixed to Total via statsAccumulationType=2.
        DETAILED_CATEGORIES=[
            ("shots", ["zones", "situations", "accuracy", "bodyparts"]),
            ("goals", ["zones"]),
            ("conversion", ["zones"]),
            ("passes", ["length"]),
            ("key-passes", ["length"]),
            ("assists", ["type"]),
            ("blocks", ["type"]),
            ("offsides", ["type"]),
            ("fouls", ["type"]),
            ("cards", ["type"]),
            ("possession-loss", ["type"]),
            ("dribbles", ["success"]),
            ("tackles", ["success"]),
            ("interception", ["success"]),
            ("clearances", ["success"]),
            ("aerial", ["success"]),
            ("saves", ["shotzone"]),
        ],
        DETAILED_ACCUMULATION_TOTAL=2,
        # --- standings rows on the season page -------------------------------
        # Each row is [stageId, teamId, teamName, then an Overall block of 9,
        # then Home (9) and Away (9), then recent form]. Only the Overall
        # sub-selection is extracted, i.e. indices 3..11.
        STANDINGS_COLUMNS=["rank", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"],
        STANDINGS_OVERALL_SLICE=(3, 12),
        SET_PIECE_SITUATIONS={
            "corner",
            "crossedfreekick",
            "directfreekick",
            "throwin",
            "setpiece",
        },
        CARD_FOUL=1,
        CARD_DIVE=2,
        CARD_UNPROFESSIONAL={7, 8},
        SSL_HELP=(
            "TLS certificate verification failed for whoscored.com.\n\n"
            "This almost always means a corporate HTTPS-inspecting proxy is re-signing\n"
            "traffic with a private root CA that Python does not trust. Fixes, best first:\n\n"
            "  1. pip install truststore     -> use the Windows cert store (keeps verification ON).\n"
            "     This module enables it automatically once installed; restart the kernel after.\n"
            "  2. fetch_tables(verify='C:/path/corp-ca.pem')  -> point at your company's root CA.\n"
            "  3. fetch_tables(verify=False) -> skip verification entirely (last resort).\n\n"
            "In Spyder, also restart the kernel (Consoles > Restart kernel) so the updated\n"
            "module is reloaded rather than served from cache."
        ),
    )


#%% helper functions for data extraction
def _is_cert_error(exc: BaseException) -> bool:
    """True if `exc` (or anything it wraps) is a TLS/certificate failure."""
    seen = set()
    markers = ("CERTIFICATE_VERIFY_FAILED", "SSLCertVerificationError", "X509", "self-signed")
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ssl.SSLError):  # covers SSLCertVerificationError
            return True
        text = str(exc)
        if type(exc).__name__ == "SSLError" or any(m in text for m in markers):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _as_number(value) -> float:
    """Coerce a standings cell to a number; JS elisions arrive as None."""
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _js_array_to_python(text: str) -> str:
    """Make a JavaScript array literal safe for ``ast.literal_eval``.

    Older WhoScored pages (e.g. Serie A 2009/10 and 2010/11) emit *elisions* --
    empty slots such as ``[1,,,2]`` or a trailing ``,'Italy',108,,,]``. Those
    are legal JavaScript (the holes read back as ``undefined``) but a
    ``SyntaxError`` in Python, so every empty slot is filled with ``None``.

    Commas inside quoted team names are left alone.
    """
    out: list[str] = []
    in_string = False
    quote = ""
    prev = ""  # last significant character seen outside a string
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                in_string = False
                prev = "x"  # a completed value
            i += 1
            continue
        if ch in "\"'":
            in_string, quote, prev = True, ch, "x"
            out.append(ch)
            i += 1
            continue
        if ch.isspace():
            out.append(ch)
            i += 1
            continue
        if ch in ",]" and prev in ("[", ","):
            out.append("None")  # the slot before this separator was empty
        out.append(ch)
        prev = ch
        i += 1
    return "".join(out)


def _extract_bracketed(text: str, start: int) -> str:
    """Return the balanced ``[...]`` literal beginning at `start`.

    A plain regex cannot do this: the standings array nests several levels deep
    and contains bracket characters inside quoted team names.
    """
    depth = 0
    in_string = False
    quote = ""
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                in_string = False
        elif ch in "\"'":
            in_string = True
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    raise ValueError("Unbalanced brackets while parsing standings")


class WhoScored:
    def __init__(
        self,
        url: str | None = None,
        delay: float = 3.0,
        verify: bool | str = True,
        config: SimpleNamespace | None = None,
    ):
        """Either `config` (from get_config) or `url` must be supplied.

        No competition ids are hardcoded here: when only a url is given the ids
        are parsed back out of it, so the config always matches the page fetched.
        """
        if config is None and url is None:
            raise ValueError(
                "Pass config=get_config(region_id, tournament_id, season_id, "
                "stage_id, slug) or an explicit url."
            )
        self.cfg = config if config is not None else get_config(**parse_url(url))
        self.url = url or self.cfg.DEFAULT_URL
        self.delay = delay
        # `verify` may be True, False (skip TLS checks), or a path to a CA bundle
        # containing your corporate proxy's root certificate.
        self.verify = verify
        if verify is False:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.session = requests_impl.Session()
        self.session.headers.update(
            {
                "User-Agent": self.cfg.UA,
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        ids = parse_url(self.url)
        self.stage_id = ids["stage_id"]
        self.tournament_id = ids["tournament_id"]
        self._html = ""
        self._auth_header: dict[str, str] = {}
        self._bootstrap()

    # -- session / anti-bot token ------------------------------------------
    def _bootstrap(self) -> None:
        try:
            r = self.session.get(self.url, timeout=30, verify=self.verify, **_IMPERSONATE)
        except Exception as exc:  # noqa: BLE001 - re-raised below with guidance
            if _is_cert_error(exc):
                raise RuntimeError(self.cfg.SSL_HELP) from exc
            raise
        r.raise_for_status()
        self._html = r.text
        name = re.search(r"gSiteHeaderName:\s*'([^']+)'", self._html)
        value = re.search(r"gSiteHeaderValue:\s*'([^']+)'", self._html)
        if not (name and value):
            raise RuntimeError(
                "Could not find gSiteHeaderName/gSiteHeaderValue in the page. "
                "WhoScored may have changed its anti-bot scheme."
            )
        self._auth_header = {name.group(1): value.group(1)}

    def _feed_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": self.url,
            **self._auth_header,
        }

    def _get(self, url: str, params: dict | None = None) -> str:
        time.sleep(self.delay)  # WhoScored rate-limits bursts hard
        r = self.session.get(
            url,
            params=params,
            headers=self._feed_headers(),
            timeout=30,
            verify=self.verify,
            **_IMPERSONATE,
        )
        r.raise_for_status()
        text = r.text.lstrip("\ufeff").strip()
        if text[:15].lower().startswith("<!doctype") or text[:5].lower() == "<html":
            raise RuntimeError(f"Blocked / non-data response from {r.url}")
        return text

    def _get_page(self, url: str) -> str:
        """Fetch a normal HTML page (not a JSON feed) with the primed session."""
        time.sleep(self.delay)
        r = self.session.get(
            url,
            headers={"Referer": self.url, "Accept": "text/html,application/xhtml+xml"},
            timeout=30,
            verify=self.verify,
            **_IMPERSONATE,
        )
        r.raise_for_status()
        return r.text

    # -- standings (stage / season page) ------------------------------------
    def standings(self, url: str | None = None) -> list[dict]:
        """League standings for the Overall sub-selection only.

        The page primes the table into its HTML as
        ``tables.push({... "standings": [[...]] ...})``; each row holds an
        Overall block followed by Home and Away blocks, and only Overall is
        returned here.

        The stage "show" url is tried first because it pins the standings to
        the stage_id that was actually requested. The season landing url is
        only a fallback: for multi-stage seasons it serves whichever stage
        WhoScored defaults to (e.g. 2022/23 Serie A lands on the relegation
        playoff, whose page carries no standings block at all).
        """
        if url is not None:
            return self._parse_standings(self._get_page(url))

        errors: list[str] = []
        for candidate in (self.cfg.STAGE_URL, self.cfg.SEASON_URL):
            try:
                return self._parse_standings(self._get_page(candidate))
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                errors.append(f"{candidate}: {exc}")
        raise RuntimeError("Could not read standings.\n  " + "\n  ".join(errors))

    def _parse_standings(self, html: str) -> list[dict]:
        cfg = self.cfg
        start, stop = cfg.STANDINGS_OVERALL_SLICE
        records: list[list] = []
        for m in re.finditer(r'"standings"\s*:\s*\[', html):
            block = _extract_bracketed(html, html.index("[", m.start()))
            records.extend(ast.literal_eval(_js_array_to_python(block)))

        # record[0] is the stage the row belongs to; when a page primes more
        # than one stage keep only the stage that was asked for.
        stage_ids = {record[0] for record in records}
        if len(stage_ids) > 1 and self.stage_id in stage_ids:
            records = [record for record in records if record[0] == self.stage_id]

        rows: list[dict] = []
        seen: set[int] = set()
        for record in records:
            team_id = record[1]
            if team_id in seen:  # a page can prime the same group twice
                continue
            seen.add(team_id)
            rows.append(
                {
                    "teamId": team_id,
                    "team": record[2],
                    **dict(zip(cfg.STANDINGS_COLUMNS, record[start:stop])),
                }
            )
        if not rows:
            raise RuntimeError(
                "No standings block found on the page. For a multi-stage season "
                "check the stage_id, or WhoScored may have changed its markup."
            )

        # Older pages append placeholder rows for clubs that never played in the
        # stage (2009/10 carries a second, zeroed "Parma Calcio 1913" under a
        # newer teamId). Drop rows with no games once any team has played --
        # before a season kicks off every row is legitimately zero, so the
        # guard leaves that case untouched.
        played = [r for r in rows if _as_number(r.get("P")) > 0]
        if played:
            rows = played

        rows.sort(key=lambda r: (r["rank"] is None, r["rank"]))
        return rows

    # -- top grid ----------------------------------------------------------
    def team_statistics(
        self,
        category: str,
        subcategory: str,
        sort_by: str,
        stats_accumulation_type: int = 0,
    ) -> list[dict]:
        """Rows of the big team-statistics grid for one tab.

        `stats_accumulation_type`: 0=PerGame, 1=Per90, 2=Total (matches the
        3rd dropdown on the Detailed tab).
        """
        params = {
            "category": category,
            "subcategory": subcategory,
            "statsAccumulationType": stats_accumulation_type,
            "field": "Overall",
            "tournamentOptions": "",
            "timeOfTheGameStart": "",
            "timeOfTheGameEnd": "",
            "teamIds": "",
            "stageId": self.stage_id,
            "sortBy": sort_by,
            "sortAscending": "false",
            "page": "",
            "numberOfTeamsToPick": "",
            "isCurrent": "true",
            "formation": "",
            "incPens": "",
            "against": "",
        }
        raw = self._get(
            "https://www.whoscored.com/statisticsfeed/1/getteamstatistics", params
        )
        return json.loads(raw)["teamTableStats"]

    def player_statistics(
        self,
        category: str,
        subcategory: str,
        sort_by: str,
        stats_accumulation_type: int = 0,
    ) -> list[dict]:
        """Rows of the big player-statistics grid for one Detailed category.

        Same category/subcategory catalog as `team_statistics`, just against
        the per-player feed (.../getplayerstatistics) instead of the
        per-team one.
        """
        params = {
            "category": category,
            "subcategory": subcategory,
            "statsAccumulationType": stats_accumulation_type,
            "isCurrent": "true",
            "playerId": "",
            "teamIds": "",
            "matchId": "",
            "stageId": self.stage_id,
            "tournamentOptions": "",
            "sortBy": sort_by,
            "sortAscending": "false",
            "age": "",
            "ageComparisonType": "",
            "appearances": "",
            "appearancesComparisonType": "",
            "field": "Overall",
            "nationality": "",
            "positionOptions": "",
            "timeOfTheGameEnd": "",
            "timeOfTheGameStart": "",
            "isMinApp": "",
            "page": "",
            "includeZeroValues": "true",
            "numberOfPlayersToPick": "",
        }
        raw = self._get(
            "https://www.whoscored.com/statisticsfeed/1/getplayerstatistics", params
        )
        return json.loads(raw)["playerTableStats"]

    # -- lower grids -------------------------------------------------------
    def stage_team_feed(self, feed_type: int, field: int = 2, against: int = 0) -> list:
        """Raw records from /stagestatfeed. `field`: 2=Overall, 1=Home, 0=Away."""
        params = {"against": against, "field": field, "teamId": -1, "type": feed_type}
        raw = self._get(
            f"https://www.whoscored.com/stagestatfeed/{self.stage_id}/stageteams/", params
        )
        # The feed is JS-literal (single quotes), not strict JSON.
        return ast.literal_eval(raw)[0]

    def _embedded(self, key: str) -> list | None:
        """Some feeds are pre-seeded into the page HTML; use them for free."""
        m = re.search(rf"{key}\s*:\s*(\[\[.*?\]\]),\s*\n", self._html, re.S)
        if not m:
            return None
        try:
            return ast.literal_eval(m.group(1))[0]
        except (ValueError, SyntaxError):
            return None

    # -- parsers -----------------------------------------------------------
    @staticmethod
    def _parse_triples(records: list, columns: list[str]) -> list[dict]:
        rows = []
        for team_id, team_name, _region, payload in records:
            values = payload[0]
            rows.append(
                {"teamId": team_id, "team": team_name, **dict(zip(columns, values))}
            )
        return rows

    def _parse_goals(self, records: list) -> list[dict]:
        cfg = self.cfg
        rows = []
        for team_id, team_name, _region, payload in records:
            buckets = defaultdict(int)
            for outcome, situation, _body_part, counts in payload[0][0]:
                if outcome != "goal":
                    continue  # the feed also carries every missed attempt
                n = counts[0]
                if situation == "owngoal":
                    buckets["Own Goal"] += n
                elif situation == "penalty":
                    buckets["Penalty"] += n
                elif situation == "fastbreak":
                    buckets["Counter Attack"] += n
                elif situation in cfg.SET_PIECE_SITUATIONS:
                    buckets["Set Piece"] += n
                else:
                    buckets["Open Play"] += n
            rows.append(
                {
                    "teamId": team_id,
                    "team": team_name,
                    "Open Play": buckets["Open Play"],
                    "Counter Attack": buckets["Counter Attack"],
                    "Set Piece": buckets["Set Piece"],
                    "Penalty": buckets["Penalty"],
                    "Own Goal": buckets["Own Goal"],
                }
            )
        return rows

    @staticmethod
    def _parse_passes(records: list, apps: dict[int, int]) -> list[dict]:
        labels = {
            "cross": "Cross pg",
            "throughball": "Through Ball pg",
            "longball": "Long Balls pg",
            "short": "Short Passes pg",
        }
        rows = []
        for team_id, team_name, _region, payload in records:
            totals = defaultdict(int)
            for _outcome, pass_type, counts in payload[0][1]:
                totals[pass_type] += counts[0]
            played = apps.get(team_id) or 1
            row = {"teamId": team_id, "team": team_name, "apps": apps.get(team_id)}
            for key, label in labels.items():
                row[label] = round(totals[key] / played, 2)
            rows.append(row)
        return rows

    def _parse_cards(self, records: list) -> list[dict]:
        cfg = self.cfg
        rows = []
        for team_id, team_name, _region, payload in records:
            entries = payload[0][0]
            fouls = dive = unprofessional = other = 0
            for entry in entries:
                if len(entry) != 2 or not isinstance(entry[1], int):
                    continue  # trailing aggregate block (fk_foul_lost, cards, ...)
                reason, count = entry
                if reason == cfg.CARD_FOUL:
                    fouls += count
                elif reason == cfg.CARD_DIVE:
                    dive += count
                elif reason in cfg.CARD_UNPROFESSIONAL:
                    unprofessional += count
                else:
                    other += count
            rows.append(
                {
                    "teamId": team_id,
                    "team": team_name,
                    "Fouls": fouls,
                    "Unprofessional": unprofessional,
                    "Dive": dive,
                    "Other": other,
                }
            )
        return rows

    # -- orchestration -----------------------------------------------------
    def all_tables(self, as_frame: bool = True, quiet: bool = False):
        """Fetch every table on the page.

        Returns a dict of ``{table_name: pandas.DataFrame}`` by default, or of
        ``{table_name: list[dict]}`` when ``as_frame=False``.
        """
        tables: dict[str, list[dict]] = {}
        apps: dict[int, int] = {}
        cfg = self.cfg

        def log(msg: str, err: bool = False) -> None:
            if not quiet:
                print(msg, file=sys.stderr if err else sys.stdout)

        try:
            tables["standings"] = self.standings()
            log(f"  + standings: {len(tables['standings'])} rows")
        except Exception as exc:  # noqa: BLE001 - keep going on partial failure
            log(f"  ! standings: {exc}", err=True)

        category, subcategory, sort_by = cfg.SUMMARY_TABLE
        try:
            rows = self.team_statistics(category, subcategory, sort_by)
            tables["team_stats_summary"] = rows
            for row in rows:
                if row.get("apps"):
                    apps.setdefault(row["teamId"], int(row["apps"]))
            log(f"  + team_stats_summary: {len(rows)} rows")
        except Exception as exc:  # noqa: BLE001 - keep going on partial failure
            log(f"  ! team_stats_summary: {exc}", err=True)

        category, subcategory, sort_by = cfg.DEFENSIVE_TABLE
        try:
            rows = self.team_statistics(category, subcategory, sort_by)
            tables["team_stats_defensive"] = rows
            for row in rows:
                if row.get("apps"):
                    apps.setdefault(row["teamId"], int(row["apps"]))
            log(f"  + team_stats_defensive: {len(rows)} rows")
        except Exception as exc:  # noqa: BLE001 - keep going on partial failure
            log(f"  ! team_stats_defensive: {exc}", err=True)

        for category, subcategories in cfg.DETAILED_CATEGORIES:
            for subcategory in subcategories:
                name = f"detailed_{category}_{subcategory}".replace("-", "_")
                try:
                    rows = self.team_statistics(
                        category, subcategory, sort_by="",
                        stats_accumulation_type=cfg.DETAILED_ACCUMULATION_TOTAL,
                    )
                except Exception as exc:  # noqa: BLE001 - keep going on partial failure
                    log(f"  ! {name}: {exc}", err=True)
                    continue
                tables[name] = rows
                for row in rows:
                    if row.get("apps"):
                        apps.setdefault(row["teamId"], int(row["apps"]))
                log(f"  + {name}: {len(rows)} rows")

        if not as_frame:
            return tables
        return {name: _to_frame(rows) for name, rows in tables.items()}

    def all_player_tables(self, as_frame: bool = True, quiet: bool = False):
        """Fetch every Detailed-tab table from the player-statistics page.

        Mirrors `all_tables`, minus `standings` and the Summary tab (neither
        applies at player granularity) -- same category/subcategory catalog,
        Total accumulation, one DataFrame per (category, subcategory) keyed
        as ``detailed_{category}_{subcategory}``, indexed by player name.
        """
        tables: dict[str, list[dict]] = {}
        cfg = self.cfg

        def log(msg: str, err: bool = False) -> None:
            if not quiet:
                print(msg, file=sys.stderr if err else sys.stdout)

        for category, subcategories in cfg.DETAILED_CATEGORIES:
            for subcategory in subcategories:
                name = f"detailed_{category}_{subcategory}".replace("-", "_")
                try:
                    rows = self.player_statistics(
                        category, subcategory, sort_by="",
                        stats_accumulation_type=cfg.DETAILED_ACCUMULATION_TOTAL,
                    )
                except Exception as exc:  # noqa: BLE001 - keep going on partial failure
                    log(f"  ! {name}: {exc}", err=True)
                    continue
                tables[name] = rows
                log(f"  + {name}: {len(rows)} rows")

        if not as_frame:
            return tables
        return {
            name: _to_frame(rows, index_col="name", lead=("name", "playerId", "teamName", "ranking", "apps"))
            for name, rows in tables.items()
        }


def _to_frame(
    rows: list[dict],
    index_col: str = "team",
    lead: tuple[str, ...] = ("team", "teamId", "ranking", "apps"),
) -> "pd.DataFrame":
    """Build a tidy DataFrame: `index_col` first, numerics coerced, indexed by it."""
    if pd is None:
        raise ImportError(
            "pandas is required for DataFrame output. "
            "Install it (`pip install pandas`) or call all_tables(as_frame=False)."
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    lead_cols = [c for c in lead if c in df.columns]
    df = df[lead_cols + [c for c in df.columns if c not in lead_cols]]

    for col in df.columns:
        if col in ("team", "teamId", "name", "playerId"):
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            # only adopt the numeric version if nothing was lost to NaN
            if converted.notna().sum() == df[col].notna().sum():
                df[col] = converted

    if index_col in df.columns:
        df = df.set_index(index_col)
    return df


def fetch_tables(
    url: str | None = None,
    delay: float = 3.0,
    verify: bool | str = True,
    quiet: bool = True,
    as_frame: bool = True,
    config: SimpleNamespace | None = None,
):
    """One-call helper for notebooks/REPL. Give it a url or a config.

    >>> from whoscored_team_stats import fetch_tables, get_config
    >>> cfg = get_config(108, 5, 10732, 24500, "italy-serie-a-2025-2026")
    >>> tables = fetch_tables(config=cfg)
    >>> tables["standings"].head()
    >>> tables["detailed_shots_zones"].sort_values("shotsTotal", ascending=False)

    Or just paste the page url and let the ids be parsed from it:

    >>> tables = fetch_tables("https://www.whoscored.com/regions/108/...")
    """
    return WhoScored(url, delay=delay, verify=verify, config=config).all_tables(
        as_frame=as_frame, quiet=quiet
    )

def fetch_player_tables(
    url: str | None = None,
    delay: float = 3.0,
    verify: bool | str = True,
    quiet: bool = True,
    as_frame: bool = True,
    config: SimpleNamespace | None = None,
):
    """One-call helper for the player-statistics Detailed tables.

    Bootstraps against the (team) stage teamstatistics page -- that page's
    anti-bot token is valid for every feed on the same stage, including
    .../getplayerstatistics -- then fetches the same Detailed category
    catalog as `fetch_tables`, but per player instead of per team.

    >>> from whoscored_team_stats import fetch_player_tables, get_config
    >>> cfg = get_config(108, 5, 10732, 24500, "italy-serie-a-2025-2026")
    >>> player_tables = fetch_player_tables(config=cfg)
    >>> player_tables["detailed_shots_zones"].sort_values("shotsTotal", ascending=False)
    """
    return WhoScored(url, delay=delay, verify=verify, config=config).all_player_tables(
        as_frame=as_frame, quiet=quiet
    )

def teams_data(region_id,tournament_id,season_id,stage_id,slug, argv: list[str] | None = None):
    """Entry point. The competition/season coordinates live here and are passed
    down to get_config; nothing is hardcoded at module level.

    Edit the defaults above (or override them on the command line / when calling
    main directly) to point the scraper at a different league or season.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="overrides the ids below")
    parser.add_argument("--region-id", type=int, default=region_id)
    parser.add_argument("--tournament-id", type=int, default=tournament_id)
    parser.add_argument("--season-id", type=int, default=season_id)
    parser.add_argument("--stage-id", type=int, default=stage_id)
    parser.add_argument("--slug", default=slug)
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="skip TLS verification (use behind a corporate HTTPS-inspecting proxy)",
    )
    parser.add_argument(
        "--cacert",
        help="path to a CA bundle including your corporate root CA (preferred over --insecure)",
    )
    parser.add_argument(
        "--save",
        metavar="OUTDIR",
        help="also write each DataFrame to OUTDIR as CSV + JSON (optional)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="drop into a Python REPL with `tables` (and each table as a variable) loaded",
    )
    # argv defaults to [] so calling main() from a Spyder/IPython console is not
    # tripped up by kernel arguments in sys.argv; the __main__ guard passes the
    # real command line.
    args = parser.parse_args(argv or [])

    verify: bool | str = args.cacert or (not args.insecure)

    # ids flow from here into get_config -> build_url
    cfg = get_config(
        region_id=args.region_id,
        tournament_id=args.tournament_id,
        season_id=args.season_id,
        stage_id=args.stage_id,
        slug=args.slug,
    )
    url = args.url or cfg.DEFAULT_URL
    if args.url:
        cfg = get_config(**parse_url(args.url))  # keep config consistent with url

    print(f"Fetching {url}")
    tables = WhoScored(url, delay=args.delay, verify=verify, config=cfg).all_tables()

    if pd is not None:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 50)

    print(f"\nLoaded {len(tables)} DataFrames:\n")
    for name, df in tables.items():
        print(f"  tables[{name!r}]  {df.shape[0]} rows x {df.shape[1]} cols")

    if args.save:
        outdir = Path(args.save)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(outdir / f"{name}.csv")
            df.reset_index().to_json(outdir / f"{name}.json", orient="records", indent=2)
        print(f"\nSaved {len(tables)} tables to {outdir.resolve()}")

    if args.interactive:
        import code

        banner = (
            "\nDataFrames are in `tables`; each is also bound to its own name, e.g.\n"
            "  standings.head()\n"
            "  detailed_shots_zones.sort_values('shotsTotal', ascending=False)\n"
        )
        code.interact(banner=banner, local={"tables": tables, "pd": pd, **tables})
    else:
        preview = tables.get("standings")
        if preview is not None:
            print("\ntables['standings'] preview:\n")
            print(preview.head())
        print(
            "\nTip: `from whoscored_team_stats import fetch_tables` to use these "
            "in a notebook, or re-run with --interactive for a REPL."
        )
    return tables

def players_data(region_id,tournament_id,season_id,stage_id,slug, argv: list[str] | None = None):
    """Entry point for the player-statistics Detailed tables (same tables as
    teams_data's Detailed tab, minus standings/summary, but per player).

    Bootstraps against the team teamstatistics page's url (the anti-bot token
    it carries is valid for the sibling .../getplayerstatistics feed too), so
    the ids flow through get_config -> build_url exactly like teams_data.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="overrides the ids below")
    parser.add_argument("--region-id", type=int, default=region_id)
    parser.add_argument("--tournament-id", type=int, default=tournament_id)
    parser.add_argument("--season-id", type=int, default=season_id)
    parser.add_argument("--stage-id", type=int, default=stage_id)
    parser.add_argument("--slug", default=slug)
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="skip TLS verification (use behind a corporate HTTPS-inspecting proxy)",
    )
    parser.add_argument(
        "--cacert",
        help="path to a CA bundle including your corporate root CA (preferred over --insecure)",
    )
    parser.add_argument(
        "--save",
        metavar="OUTDIR",
        help="also write each DataFrame to OUTDIR as CSV + JSON (optional)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="drop into a Python REPL with `tables` (and each table as a variable) loaded",
    )
    args = parser.parse_args(argv or [])

    verify: bool | str = args.cacert or (not args.insecure)

    cfg = get_config(
        region_id=args.region_id,
        tournament_id=args.tournament_id,
        season_id=args.season_id,
        stage_id=args.stage_id,
        slug=args.slug,
    )
    url = args.url or cfg.DEFAULT_URL
    if args.url:
        cfg = get_config(**parse_url(args.url))  # keep config consistent with url

    print(f"Fetching {url}")
    tables = WhoScored(url, delay=args.delay, verify=verify, config=cfg).all_player_tables()

    if pd is not None:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 50)

    print(f"\nLoaded {len(tables)} DataFrames:\n")
    for name, df in tables.items():
        print(f"  tables[{name!r}]  {df.shape[0]} rows x {df.shape[1]} cols")

    if args.save:
        outdir = Path(args.save)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(outdir / f"{name}.csv")
            df.reset_index().to_json(outdir / f"{name}.json", orient="records", indent=2)
        print(f"\nSaved {len(tables)} tables to {outdir.resolve()}")

    if args.interactive:
        import code

        banner = (
            "\nDataFrames are in `tables`; each is also bound to its own name, e.g.\n"
            "  detailed_shots_zones.sort_values('shotsTotal', ascending=False)\n"
        )
        code.interact(banner=banner, local={"tables": tables, "pd": pd, **tables})
    else:
        print(
            "\nTip: `from whoscored_team_stats import fetch_player_tables` to use these "
            "in a notebook, or re-run with --interactive for a REPL."
        )
    return tables

def league_codes(country,season):
    league_code = {
        ('Argentina',2023): (11,68,9326,21480,"argentina-liga-profesional-2023"),
        ('Argentina',2022): (11,68,9081,20951,"argentina-liga-profesional-2022"),
        ('Argentina',2021): (11,68,8679,19893,"argentina-liga-profesional-2021"),
        ('Argentina',2020): (11,68,7905,17737,"argentina-liga-profesional-2019-2020"),
        ('Argentina',2019): (11,68,7460,16537,"argentina-liga-profesional-2018-2019"),
        ('Argentina',2018): (11,68,6995,15436,"argentina-liga-profesional-2017-2018"),
        ('Argentina',2017): (11,68,6484,14066,"argentina-liga-profesional-2016-2017"),
        ('Argentina',2016): (11,68,6144,13288,"argentina-liga-profesional-2016"),
        ('Belgium',2027): (22,18,11160,25564,"belgium-jupiler-pro-league-2026-2027"),
        ('Belgium',2026): (22,18,10759,24549,"belgium-jupiler-pro-league-2025-2026"),
        ('Belgium',2025): (22,18,10304,23382,"belgium-jupiler-pro-league-2024-2025"),
        ('Belgium',2024): (22,18,9660,22144,"belgium-jupiler-pro-league-2023-2024"),
        ('Belgium',2023): (22,18,9156,21083,"belgium-jupiler-pro-league-2022-2023"),
        ('Belgium',2022): (22,18,8627,19808,"belgium-jupiler-pro-league-2021-2022"),
        ('Belgium',2021): (22,18,8197,18620,"belgium-jupiler-pro-league-2020-2021"),
        ('Brazil',2026): (31,95,10980,25039,"brazil-brasileirão-2026"),
        ('Brazil',2025): (31,95,10621,24121,"brazil-brasileirão-2025"),
        ('Brazil',2024): (31,95,10003,22961,"brazil-brasileirão-2024"),
        ('Brazil',2023): (31,95,9428,21743,"brazil-brasileirão-2023"),
        ('Brazil',2022): (31,95,8984,20566,"brazil-brasileirão-2022"),
        ('Brazil',2021): (31,95,8555,19551,"brazil-brasileirão-2021"),
        ('Brazil',2020): (31,95,8158,18472,"brazil-brasileirão-2020"),
        ('Brazil',2019): (31,95,7683,17175,"brazil-brasileirão-2019"),
        ('Brazil',2018): (31,95,7243,15996,"brazil-brasileirão-2018"),
        ('Brazil',2017): (31,95,6700,14746,"brazil-brasileirão-2017"),
        ('Brazil',2016): (31,95,6242,13492,"brazil-brasileirão-2016"),
        ('Brazil',2015): (31,95,5713,12121,"brazil-brasileirão-2015"),
        ('Brazil',2014): (31,95,4185,8677,"brazil-brasileirão-2014"),
        ('Brazil',2013): (31,95,3753,7479,"brazil-brasileirão-2013"),
        ('China',2020): (45,162,8122,18518,"china-super-league-2020"),
        ('China',2019): (45,162,7667,17135,"china-super-league-2019"),
        ('China',2018): (45,162,7242,15995,"china-super-league-2018"),
        ('China',2017): (45,162,6685,14726,"china-super-league-2017"),
        ('China',2016): (45,162,6213,13418,"china-super-league-2016"),
        ('England',2027): (252,2,11141,25544,"england-premier-league-2026-2027"),
        ('England',2026): (252,2,10743,24533,"england-premier-league-2025-2026"),
        ('England',2025): (252,2,10316,23400,"england-premier-league-2024-2025"),
        ('England',2024): (252,2,9618,22076,"england-premier-league-2023-2024"),
        ('England',2023): (252,2,9075,20934,"england-premier-league-2022-2023"),
        ('England',2022): (252,2,8618,19793,"england-premier-league-2021-2022"),
        ('England',2021): (252,2,8228,18685,"england-premier-league-2020-2021"),
        ('England',2020): (252,2,7811,17590,"england-premier-league-2019-2020"),
        ('England',2019): (252,2,7361,16368,"england-premier-league-2018-2019"),
        ('England',2018): (252,2,6829,15151,"england-premier-league-2017-2018"),
        ('England',2017): (252,2,6335,13796,"england-premier-league-2016-2017"),
        ('England',2016): (252,2,5826,12496,"england-premier-league-2015-2016"),
        ('England',2015): (252,2,4311,9155,"england-premier-league-2014-2015"),
        ('England',2014): (252,2,3853,7794,"england-premier-league-2013-2014"),
        ('England',2013): (252,2,3389,6531,"england-premier-league-2012-2013"),
        ('England',2012): (252,2,2935,5476,"england-premier-league-2011-2012"),
        ('England',2011): (252,2,2458,4345,"england-premier-league-2010-2011"),
        ('England2',2027): (252,7,11184,25588,"england-championship-2026-2027"),
        ('England2',2026): (252,7,10784,24580,"england-championship-2025-2026"),
        ('England2',2025): (252,7,10343,23427,"england-championship-2024-2025"),
        ('England2',2024): (252,7,9622,22080,"england-championship-2023-2024"),
        ('England2',2023): (252,7,9141,21050,"england-championship-2022-2023"),
        ('England2',2022): (252,7,8619,19794,"england-championship-2021-2022"),
        ('England2',2021): (252,7,8304,18825,"england-championship-2020-2021"),
        ('England2',2020): (252,7,7840,17629,"england-championship-2019-2020"),
        ('England2',2019): (252,7,7379,16389,"england-championship-2018-2019"),
        ('England2',2018): (252,7,6848,15177,"england-championship-2017-2018"),
        ('England2',2017): (252,7,6365,13832,"england-championship-2016-2017"),
        ('England2',2016): (252,7,5827,12497,"england-championship-2015-2016"),
        ('England2',2015): (252,7,4312,9156,"england-championship-2014-2015"),
        ('England2',2014): (252,7,3859,7800,"england-championship-2013-2014"),
        ('England3',2027): (252,8,11185,25589,"england-league-one-2026-2027"),
        ('England3',2026): (252,8,10785,24581,"england-league-one-2025-2026"),
        ('England3',2025): (252,8,10342,23428,"england-league-one-2024-2025"),
        ('England3',2024): (252,8,9623,22081,"england-league-one-2023-2024"),
        ('England3',2023): (252,8,9142,21051,"england-league-one-2022-2023"),
        ('England3',2022): (252,8,8620,19795,"england-league-one-2021-2022"),
        ('England3',2021): (252,8,8305,18826,"england-league-one-2020-2021"),
        ('England3',2020): (252,8,7841,17630,"england-league-one-2019-2020"),
        ('England4',2027): (252,9,11186,25590,"england-league-two-2026-2027"),
        ('England4',2026): (252,9,10786,24582,"england-league-two-2025-2026"),
        ('England4',2025): (252,9,10344,23429,"england-league-two-2024-2025"),
        ('England4',2024): (252,9,9621,22079,"england-league-two-2023-2024"),
        ('England4',2023): (252,9,9143,21052,"england-league-two-2022-2023"),
        ('England4',2022): (252,9,8621,19796,"england-league-two-2021-2022"),
        ('England4',2021): (252,9,8306,18827,"england-league-two-2020-2021"),
        ('England4',2020): (252,9,7842,17631,"england-league-two-2019-2020"),
        ('Europe',2027): (250,12,11295,25800,"europe-champions-league-2026-2027"),
        ('Europe',2026): (250,12,10903,24796,"europe-champions-league-2025-2026"),
        ('Europe',2025): (250,12,10456,23663,"europe-champions-league-2024-2025"),
        ('Europe',2024): (250,12,9664,22490,"europe-champions-league-2023-2024"),
        ('Europe',2023): (250,12,9086,20961,"europe-champions-league-2022-2023"),
        ('Europe',2022): (250,12,8623,20088,"europe-champions-league-2021-2022"),
        ('Europe',2021): (250,12,8177,18972,"europe-champions-league-2020-2021"),
        ('Europe',2020): (250,12,7804,17937,"europe-champions-league-2019-2020"),
        ('Europe',2019): (250,12,7352,16652,"europe-champions-league-2018-2019"),
        ('Europe',2018): (250,12,6842,15482,"europe-champions-league-2017-2018"),
        ('Europe',2017): (250,12,6349,14167,"europe-champions-league-2016-2017"),
        ('Europe',2016): (250,12,5848,12857,"europe-champions-league-2015-2016"),
        ('Europe',2015): (250,12,4333,11505,"europe-champions-league-2014-2015"),
        ('Europe',2014): (250,12,3872,8150,"europe-champions-league-2013-2014"),
        ('Europe',2013): (250,12,3416,6849,"europe-champions-league-2012-2013"),
        ('Europe',2012): (250,12,2944,5748,"europe-champions-league-2011-2012"),
        ('Europe',2011): (250,12,2474,4753,"europe-champions-league-2010-2011"),
        ('Europe2',2027): (250,30,11294,25801,"europe-europa-league-2026-2027"),
        ('Europe2',2026): (250,30,10904,24798,"europe-europa-league-2025-2026"),
        ('Europe2',2025): (250,30,10458,23665,"europe-europa-league-2024-2025"),
        ('Europe2',2024): (250,30,9778,22510,"europe-europa-league-2023-2024"),
        ('Europe2',2023): (250,30,9087,20971,"europe-europa-league-2022-2023"),
        ('Europe2',2022): (250,30,8741,20106,"europe-europa-league-2021-2022"),
        ('Europe2',2021): (250,30,8178,18981,"europe-europa-league-2020-2021"),
        ('Europe2',2020): (250,30,7805,17956,"europe-europa-league-2019-2020"),
        ('Europe2',2019): (250,30,7353,16677,"europe-europa-league-2018-2019"),
        ('Europe2',2018): (250,30,6843,15490,"europe-europa-league-2017-2018"),
        ('Europe2',2017): (250,30,6275,14186,"europe-europa-league-2016-2017"),
        ('Europe2',2016): (250,30,5849,12869,"europe-europa-league-2015-2016"),
        ('Europe2',2015): (250,30,4332,11513,"europe-europa-league-2014-2015"),
        ('Europe2',2014): (250,30,3871,8158,"europe-europa-league-2013-2014"),
        ('Europe2',2013): (250,30,3418,6857,"europe-europa-league-2012-2013"),
        ('France',2027): (74,22,11150,25554,"france-ligue-1-2026-2027"),
        ('France',2026): (74,22,10792,24609,"france-ligue-1-2025-2026"),
        ('France',2025): (74,22,10329,23414,"france-ligue-1-2024-2025"),
        ('France',2024): (74,22,9635,22105,"france-ligue-1-2023-2024"),
        ('France',2023): (74,22,9129,21037,"france-ligue-1-2022-2023"),
        ('France',2022): (74,22,8671,19866,"france-ligue-1-2021-2022"),
        ('France',2021): (74,22,8185,18594,"france-ligue-1-2020-2021"),
        ('France',2020): (74,22,7814,17593,"france-ligue-1-2019-2020"),
        ('France',2019): (74,22,7344,16348,"france-ligue-1-2018-2019"),
        ('France',2018): (74,22,6833,15155,"france-ligue-1-2017-2018"),
        ('France',2017): (74,22,6318,13768,"france-ligue-1-2016-2017"),
        ('France',2016): (74,22,5830,12501,"france-ligue-1-2015-2016"),
        ('France',2015): (74,22,4279,9105,"france-ligue-1-2014-2015"),
        ('France',2014): (74,22,3836,7771,"france-ligue-1-2013-2014"),
        ('France',2013): (74,22,3356,6476,"france-ligue-1-2012-2013"),
        ('France',2012): (74,22,2920,5451,"france-ligue-1-2011-2012"),
        ('France',2011): (74,22,2417,4273,"france-ligue-1-2010-2011"),
        ('Germany',2027): (81,3,11217,25666,"germany-bundesliga-2026-2027"),
        ('Germany',2026): (81,3,10720,24478,"germany-bundesliga-2025-2026"),
        ('Germany',2025): (81,3,10365,23471,"germany-bundesliga-2024-2025"),
        ('Germany',2024): (81,3,9649,22128,"germany-bundesliga-2023-2024"),
        ('Germany',2023): (81,3,9120,21026,"germany-bundesliga-2022-2023"),
        ('Germany',2022): (81,3,8667,19862,"germany-bundesliga-2021-2022"),
        ('Germany',2021): (81,3,8279,18762,"germany-bundesliga-2020-2021"),
        ('Germany',2020): (81,3,7872,17682,"germany-bundesliga-2019-2020"),
        ('Germany',2019): (81,3,7405,16427,"germany-bundesliga-2018-2019"),
        ('Germany',2018): (81,3,6902,15243,"germany-bundesliga-2017-2018"),
        ('Germany',2017): (81,3,6392,13872,"germany-bundesliga-2016-2017"),
        ('Germany',2016): (81,3,5870,12559,"germany-bundesliga-2015-2016"),
        ('Germany',2015): (81,3,4336,9192,"germany-bundesliga-2014-2015"),
        ('Germany',2014): (81,3,3863,7806,"germany-bundesliga-2013-2014"),
        ('Germany',2013): (81,3,3424,6576,"germany-bundesliga-2012-2013"),
        ('Germany',2012): (81,3,2949,5492,"germany-bundesliga-2011-2012"),
        ('Germany',2011): (81,3,2520,4448,"germany-bundesliga-2010-2011"),
        ('Germany2',2027): (81,6,11218,25667,"germany-2-bundesliga-2026-2027"),
        ('Germany2',2026): (81,6,10721,24479,"germany-2-bundesliga-2025-2026"),
        ('Germany2',2025): (81,6,10366,23472,"germany-2-bundesliga-2024-2025"),
        ('Germany2',2024): (81,6,9650,22129,"germany-2-bundesliga-2023-2024"),
        ('Germany2',2023): (81,6,9121,21027,"germany-2-bundesliga-2022-2023"),
        ('Germany2',2022): (81,6,8668,19863,"germany-2-bundesliga-2021-2022"),
        ('Germany2',2021): (81,6,8280,18763,"germany-2-bundesliga-2020-2021"),
        ('Germany2',2020): (81,6,7873,17683,"germany-2-bundesliga-2019-2020"),
        ('Germany2',2019): (81,6,7406,16428,"germany-2-bundesliga-2018-2019"),
        ('Germany2',2018): (81,6,6903,15244,"germany-2-bundesliga-2017-2018"),
        ('Germany2',2017): (81,6,6393,13873,"germany-2-bundesliga-2016-2017"),
        ('Germany2',2016): (81,6,5871,12560,"germany-2-bundesliga-2015-2016"),
        ('Italy',2027): (108,5,11126,25518,"italy-serie-a-2026-2027"),
        ('Italy',2026): (108,5,10732,24500,"italy-serie-a-2025-2026"),
        ('Italy',2025): (108,5,10375,23490,"italy-serie-a-2024-2025"),
        ('Italy',2024): (108,5,9659,22143,"italy-serie-a-2023-2024"),
        ('Italy',2023): (108,5,9159,21087,"italy-serie-a-2022-2023"),
        ('Italy',2022): (108,5,8735,19982,"italy-serie-a-2021-2022"),
        ('Italy',2021): (108,5,8330,18873,"italy-serie-a-2020-2021"),
        ('Italy',2020): (108,5,7928,17835,"italy-serie-a-2019-2020"),
        ('Italy',2019): (108,5,7468,16548,"italy-serie-a-2018-2019"),
        ('Italy',2018): (108,5,6974,15404,"italy-serie-a-2017-2018"),
        ('Italy',2017): (108,5,6461,14014,"italy-serie-a-2016-2017"),
        ('Italy',2016): (108,5,5970,12770,"italy-serie-a-2015-2016"),
        ('Italy',2015): (108,5,5441,11369,"italy-serie-a-2014-2015"),
        ('Italy',2014): (108,5,3978,8019,"italy-serie-a-2013-2014"),
        ('Italy',2013): (108,5,3512,6739,"italy-serie-a-2012-2013"),
        ('Italy',2012): (108,5,3054,5667,"italy-serie-a-2011-2012"),
        ('Italy',2011): (108,5,2626,4659,"italy-serie-a-2010-2011"),
        ('Netherlands',2027): (155,13,11135,25537,"netherlands-eredivisie-2026-2027"),
        ('Netherlands',2026): (155,13,10752,24542,"netherlands-eredivisie-2025-2026"),
        ('Netherlands',2025): (155,13,10321,23405,"netherlands-eredivisie-2024-2025"),
        ('Netherlands',2024): (155,13,9705,22225,"netherlands-eredivisie-2023-2024"),
        ('Netherlands',2023): (155,13,9112,21021,"netherlands-eredivisie-2022-2023"),
        ('Netherlands',2022): (155,13,8625,19802,"netherlands-eredivisie-2021-2022"),
        ('Netherlands',2021): (155,13,8187,18596,"netherlands-eredivisie-2020-2021"),
        ('Netherlands',2020): (155,13,7815,17594,"netherlands-eredivisie-2019-2020"),
        ('Netherlands',2019): (155,13,7354,16360,"netherlands-eredivisie-2018-2019"),
        ('Netherlands',2018): (155,13,6826,15148,"netherlands-eredivisie-2017-2018"),
        ('Netherlands',2017): (155,13,6331,13792,"netherlands-eredivisie-2016-2017"),
        ('Netherlands',2016): (155,13,5810,12467,"netherlands-eredivisie-2015-2016"),
        ('Netherlands',2015): (155,13,4289,9121,"netherlands-eredivisie-2014-2015"),
        ('Netherlands',2014): (155,13,3851,7790,"netherlands-eredivisie-2013-2014"),
        ('Norway',2017): (165,41,6609,14501,"norway-eliteserien-2017"),
        ('Norway',2016): (165,41,6115,13179,"norway-eliteserien-2016"),
        ('Portugal',2027): (177,21,11222,25672,"portugal-liga-2026-2027"),
        ('Portugal',2026): (177,21,10774,24568,"portugal-liga-2025-2026"),
        ('Portugal',2025): (177,21,10378,23494,"portugal-liga-2024-2025"),
        ('Portugal',2024): (177,21,9730,22254,"portugal-liga-2023-2024"),
        ('Portugal',2023): (177,21,9191,21149,"portugal-liga-2022-2023"),
        ('Portugal',2022): (177,21,8714,19947,"portugal-liga-2021-2022"),
        ('Portugal',2021): (177,21,8315,18842,"portugal-liga-2020-2021"),
        ('Portugal',2020): (177,21,7893,17710,"portugal-liga-2019-2020"),
        ('Portugal',2019): (177,21,7429,16461,"portugal-liga-2018-2019"),
        ('Portugal',2018): (177,21,6933,15323,"portugal-liga-2017-2018"),
        ('Portugal',2017): (177,21,6438,13957,"portugal-liga-2016-2017"),
        ('Russia',2027): (182,77,11195,25631,"russia-premier-league-2026-2027"),
        ('Russia',2026): (182,77,10764,24555,"russia-premier-league-2025-2026"),
        ('Russia',2025): (182,77,10332,23417,"russia-premier-league-2024-2025"),
        ('Russia',2024): (182,77,9693,22201,"russia-premier-league-2023-2024"),
        ('Russia',2023): (182,77,9153,21079,"russia-premier-league-2022-2023"),
        ('Russia',2022): (182,77,8639,19826,"russia-premier-league-2021-2022"),
        ('Russia',2021): (182,77,8242,18707,"russia-premier-league-2020-2021"),
        ('Russia',2020): (182,77,7802,17565,"russia-premier-league-2019-2020"),
        ('Russia',2019): (182,77,7389,16400,"russia-premier-league-2018-2019"),
        ('Russia',2018): (182,77,6819,15139,"russia-premier-league-2017-2018"),
        ('Russia',2017): (182,77,6357,13823,"russia-premier-league-2016-2017"),
        ('Russia',2016): (182,77,5859,12537,"russia-premier-league-2015-2016"),
        ('Russia',2015): (182,77,4303,9145,"russia-premier-league-2014-2015"),
        ('Russia',2014): (182,77,3861,7803,"russia-premier-league-2013-2014"),
        ('Scotland',2027): (253,20,11142,25545,"scotland-premiership-2026-2027"),
        ('Scotland',2026): (253,20,10760,24550,"scotland-premiership-2025-2026"),
        ('Scotland',2025): (253,20,10351,23451,"scotland-premiership-2024-2025"),
        ('Scotland',2024): (253,20,9713,22235,"scotland-premiership-2023-2024"),
        ('Scotland',2023): (253,20,9125,21031,"scotland-premiership-2022-2023"),
        ('Scotland',2022): (253,20,8636,19821,"scotland-premiership-2021-2022"),
        ('Scotland',2021): (253,20,8191,18606,"scotland-premiership-2020-2021"),
        ('South America',2024): (265,105,9923,23248,"south-america-copa-libertadores-2024"),
        ('Spain',2027): (206,4,11213,25662,"spain-laliga-2026-2027"),
        ('Spain',2026): (206,4,10803,24622,"spain-laliga-2025-2026"),
        ('Spain',2025): (206,4,10317,23401,"spain-laliga-2024-2025"),
        ('Spain',2024): (206,4,9682,22176,"spain-laliga-2023-2024"),
        ('Spain',2023): (206,4,9149,21073,"spain-laliga-2022-2023"),
        ('Spain',2022): (206,4,8681,19895,"spain-laliga-2021-2022"),
        ('Spain',2021): (206,4,8321,18851,"spain-laliga-2020-2021"),
        ('Spain',2020): (206,4,7889,17702,"spain-laliga-2019-2020"),
        ('Spain',2019): (206,4,7466,16546,"spain-laliga-2018-2019"),
        ('Spain',2018): (206,4,6960,15375,"spain-laliga-2017-2018"),
        ('Spain',2017): (206,4,6436,13955,"spain-laliga-2016-2017"),
        ('Spain',2016): (206,4,5933,12647,"spain-laliga-2015-2016"),
        ('Spain',2015): (206,4,5435,11363,"spain-laliga-2014-2015"),
        ('Spain',2014): (206,4,3922,7920,"spain-laliga-2013-2014"),
        ('Spain',2013): (206,4,3470,6652,"spain-laliga-2012-2013"),
        ('Spain',2012): (206,4,3004,5577,"spain-laliga-2011-2012"),
        ('Spain',2011): (206,4,2596,4624,"spain-laliga-2010-2011"),
        ('Sweden',2017): (212,40,6594,14410,"sweden-allsvenskan-2017"),
        ('Sweden',2016): (212,40,6127,13224,"sweden-allsvenskan-2016"),
        ('Turkey',2027): (225,17,11231,25681,"turkey-super-lig-2026-2027"),
        ('Turkey',2026): (225,17,10807,24627,"turkey-super-lig-2025-2026"),
        ('Turkey',2025): (225,17,10397,23530,"turkey-super-lig-2024-2025"),
        ('Turkey',2024): (225,17,9764,22331,"turkey-super-lig-2023-2024"),
        ('Turkey',2023): (225,17,9183,21137,"turkey-super-lig-2022-2023"),
        ('Turkey',2022): (225,17,8732,19975,"turkey-super-lig-2021-2022"),
        ('Turkey',2021): (225,17,8313,18838,"turkey-super-lig-2020-2021"),
        ('Turkey',2020): (225,17,7912,17795,"turkey-super-lig-2019-2020"),
        ('Turkey',2019): (225,17,7437,16477,"turkey-super-lig-2018-2019"),
        ('Turkey',2018): (225,17,6936,15334,"turkey-super-lig-2017-2018"),
        ('Turkey',2017): (225,17,6447,13974,"turkey-super-lig-2016-2017"),
        ('Turkey',2016): (225,17,5915,12625,"turkey-super-lig-2015-2016"),
        ('Turkey',2015): (225,17,5411,11306,"turkey-super-lig-2014-2015"),
        ('USA',2026): (233,85,10957,24960,"usa-major-league-soccer-2026"),
        ('USA',2025): (233,85,10568,24001,"usa-major-league-soccer-2025"),
        ('USA',2024): (233,85,9927,22796,"usa-major-league-soccer-2024"),
        ('USA',2023): (233,85,9365,21622,"usa-major-league-soccer-2023"),
        ('USA',2022): (233,85,8883,20434,"usa-major-league-soccer-2022"),
        ('USA',2021): (233,85,8545,19513,"usa-major-league-soccer-2021"),
        ('USA',2020): (233,85,8055,18171,"usa-major-league-soccer-2020"),
        ('USA',2019): (233,85,7609,17011,"usa-major-league-soccer-2019"),
        ('USA',2018): (233,85,7172,15823,"usa-major-league-soccer-2018"),
        ('USA',2017): (233,85,6620,14550,"usa-major-league-soccer-2017"),
        ('USA',2016): (233,85,6137,13276,"usa-major-league-soccer-2016"),
        ('USA',2015): (233,85,5607,11890,"usa-major-league-soccer-2015"),
        ('USA',2014): (233,85,4091,8358,"usa-major-league-soccer-2014"),
        ('USA',2013): (233,85,3672,7250,"usa-major-league-soccer-2013"),
        }
    region_id,tournament_id,season_id,stage_id,slug = league_code[country,season]
    return region_id,tournament_id,season_id,stage_id,slug

#%% helper functions for data analysis
def multi_season_team_extract(country,start,end):
    i=start; full_team_stats = []; full_regression_stats = []
    while(i<=end):
        print("season",i)
        r,t,s,ss,slug = league_codes(country,i)
        all_tables = teams_data(r,t,s,ss,slug,argv=sys.argv[1:])
        full_stats,regression_stats = team_stats_summarize(all_tables,i)
        if(i==start):
            full_team_stats = full_stats
            full_regression_stats = regression_stats
        else:
            full_team_stats = pd.concat([full_stats,full_team_stats])
            full_regression_stats = pd.concat([regression_stats,full_regression_stats])
        i+=1
        print()    
    return full_team_stats,full_regression_stats

def team_stats_summarize(all_tables,season):
    standings = all_tables['standings']
    standings = standings.reset_index()
    standings = standings[['team', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']]

    aerials = all_tables['detailed_aerial_success']
    aerials['aerial_win%'] = aerials['duelAerialWon']/aerials['duelAerialTotal']
    aerials = aerials[['teamName','minsPlayed','duelAerialTotal','aerial_win%']]
    aerials = aerials.rename(columns={'teamName': 'team'})
    
    blocks = all_tables['detailed_blocks_type']
    blocks = blocks[['teamName','outfielderBlock', 'passCrossBlockedDefensive', 'outfielderBlockedPass']]
    blocks = blocks.rename(columns={'teamName': 'team','outfielderBlock':'Shots_blocked','passCrossBlockedDefensive':'Cross_blocked'})
    
    cards = all_tables['detailed_cards_type']
    cards = cards[['teamName','yellowCard', 'redCard']]
    cards = cards.rename(columns={'teamName': 'team'})
    
    clear = all_tables['detailed_clearances_success']
    clear = clear[['teamName','clearanceTotal']]
    clear = clear.rename(columns={'teamName': 'team'})
    
    dribbles = all_tables['detailed_dribbles_success']
    dribbles = dribbles[['teamName','dribbleLost', 'dribbleWon', 'dribbleTotal']]
    dribbles['dribble_win%'] = dribbles['dribbleWon']/dribbles['dribbleTotal']
    dribbles = dribbles.rename(columns={'teamName': 'team'})
    
    fouls = all_tables['detailed_fouls_type']
    fouls = fouls[['teamName','foulGiven', 'foulCommitted']]
    fouls = fouls.rename(columns={'teamName': 'team'})
    
    interceptions = all_tables['detailed_interception_success']
    interceptions = interceptions[['teamName','interceptionAll']]
    interceptions = interceptions.rename(columns={'teamName': 'team'})
    
    passes = all_tables['detailed_passes_length']
    passes['long_success%'] = passes['passLongBallAccurate']/(passes['passLongBallAccurate']+passes['passLongBallInaccurate'])
    passes['short_success%'] = passes['shortPassAccurate']/(passes['shortPassAccurate']+passes['shortPassInaccurate'])
    passes['long_bias'] = (passes['passLongBallAccurate']+passes['passLongBallInaccurate'])/(passes['shortPassAccurate']+passes['shortPassInaccurate'])
    passes = passes[['teamName','passTotal', 'long_success%', 'short_success%', 'long_bias']]
    passes = passes.rename(columns={'teamName': 'team'})
    
    kp = all_tables['detailed_key_passes_length']
    kp = kp[['teamName','keyPassesTotal']]
    kp = kp.rename(columns={'teamName': 'team'})
    
    offside = all_tables['detailed_offsides_type']
    offside = offside[['teamName','offsideGiven']]
    offside = offside.rename(columns={'teamName': 'team'})
    
    possloss = all_tables['detailed_possession_loss_type']
    possloss = possloss[['teamName','turnover', 'dispossessed']]
    possloss = possloss.rename(columns={'teamName': 'team'})
    
    saves = all_tables['detailed_saves_shotzone']
    saves = saves[['teamName','saveTotal']]
    saves = saves.rename(columns={'teamName': 'team'})
    
    shots = all_tables['detailed_shots_accuracy']
    shots['shots_target%'] = shots['shotOnTarget']/shots['shotsTotal']
    shots['shots_blocked%'] = shots['shotBlocked']/shots['shotsTotal']
    shots = shots[['teamName', 'shotsTotal', 'shots_target%','shots_blocked%']]
    shots = shots.rename(columns={'teamName': 'team'})
    
    tackles = all_tables['detailed_tackles_success']
    tackles['tackle_success%'] = tackles['tackleWonTotal']/tackles['tackleTotalAttempted']
    tackles = tackles[['teamName','tackleTotalAttempted','tackle_success%']]
    tackles = tackles.rename(columns={'teamName': 'team'})
    
    possession = all_tables['team_stats_summary']
    possession = possession[['teamName','tournamentName','possession']]
    possession = possession.rename(columns={'teamName': 'team'})

    def_stats = all_tables['team_stats_defensive']
    def_stats = def_stats[['teamName', 'shotsConcededPerGame']]
    def_stats = def_stats.rename(columns={'teamName': 'team'})

    standings = standings.merge(aerials, on='team', how='left')
    standings = standings.merge(blocks, on='team', how='left')
    standings = standings.merge(cards, on='team', how='left')
    standings = standings.merge(clear, on='team', how='left')
    standings = standings.merge(dribbles, on='team', how='left')
    standings = standings.merge(fouls, on='team', how='left')
    standings = standings.merge(interceptions, on='team', how='left')
    standings = standings.merge(passes, on='team', how='left')
    standings = standings.merge(kp, on='team', how='left')
    standings = standings.merge(possloss, on='team', how='left')
    standings = standings.merge(offside, on='team', how='left')
    standings = standings.merge(saves, on='team', how='left')
    standings = standings.merge(shots, on='team', how='left')
    standings = standings.merge(tackles, on='team', how='left')
    standings = standings.merge(possession, on='team', how='left')
    standings = standings.merge(def_stats, on='team', how='left')

    standings['season'] = season
    standings['Save%'] = standings['saveTotal']/(standings['saveTotal']+standings['GA'])
    standings['MPG'] = standings['minsPlayed']/standings['P']

    standings.iloc[:,2:8+1] = standings.iloc[:,2:8+1].div(standings['P'], axis=0)

    #L2 regularization needed for teams with missing possesion values
    #from sklearn.linear_model import Ridge
    #ridge_model = Ridge(alpha=1.0)
    #ridge_model.fit(standings[['long_success%','short_success%','long_bias','passTotal']], standings['possession'])
    #standings['possession_adj'] = ridge_model.predict(standings[['long_success%','short_success%','long_bias','passTotal','dispossessed']])
    standings['possession_adj'] =  -0.051*standings['long_success%'] +\
                                    0.014*standings['short_success%'] -\
                                    0.0048*standings['long_bias'] +\
                                    0.0000181*standings['passTotal'] -\
                                    0.000045*standings['dispossessed'] + 0.22
    standings['delta'] = standings['possession'] - standings['possession_adj']
    standings['possession'] = np.where(standings['delta']<-0.1,standings['possession_adj'],standings['possession'])
    standings['possession'] *= (0.5/standings['possession'].mean())

    analysis = standings.copy()
    analysis['Passes against'] = analysis['passTotal'] * (1-analysis['possession']) / analysis['possession']
    analysis['Pace'] = analysis['passTotal'] + analysis['Passes against']
    analysis['Pace'] /= analysis['P']
    analysis['Sh'] = analysis['shotsTotal']/analysis['passTotal']
    analysis['ShA'] = analysis['shotsConcededPerGame']/analysis['Passes against']
    analysis['Drb'] = analysis['dribbleTotal']/analysis['passTotal']
    analysis['FlsW'] = analysis['foulGiven']/analysis['passTotal']
    analysis['FlsC'] = analysis['foulCommitted']/analysis['Passes against']
    analysis['Tkl'] = analysis['tackleTotalAttempted']/analysis['Passes against']
    analysis['Int'] = analysis['interceptionAll']/analysis['Passes against']
    analysis['Blk'] = (analysis['Shots_blocked']+analysis['Cross_blocked']+analysis['outfielderBlockedPass'])/analysis['Passes against']
    analysis['Clr'] = analysis['clearanceTotal']/analysis['passTotal']
    analysis['TO%'] = (analysis['turnover']+analysis['dispossessed'])/analysis['passTotal']
    analysis['Off'] = analysis['offsideGiven']/analysis['Passes against']
    analysis['YC'] = analysis['yellowCard']/analysis['Passes against']
    analysis['RC'] = analysis['redCard']/analysis['Passes against']
    analysis['KP'] = analysis['keyPassesTotal']/analysis['passTotal']
    analysis['Head'] = analysis['duelAerialTotal']/analysis['Passes against']
    
    analysis = analysis[['tournamentName','team', 'season','P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts','MPG',
                         'Pace','possession','long_success%','short_success%','long_bias','Sh','shots_target%','shots_blocked%',
                         'KP','TO%','Drb','dribble_win%', 'FlsW', 'FlsC', 'Tkl','tackle_success%', 'Int','Blk','Clr', 'Off',
                         'Head', 'aerial_win%', 'ShA', 'YC', 'RC','Save%']]
    return standings,analysis

def team_stats_regresion(analysis,target,split_season):
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    variables = ['possession','Pace','long_success%','short_success%','long_bias','Sh','shots_target%','shots_blocked%',
                 'KP','TO%','Drb','dribble_win%', 'FlsW', 'FlsC', 'Tkl','tackle_success%', 'Int','Blk','Clr', 'Off',
                 'Head', 'aerial_win%', 'YC', 'RC', 'ShA','Save%'] #

    #X = analysis[variables]
    #y = analysis[target]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = analysis[analysis['season']<split_season][variables]
    y_train = analysis[analysis['season']<split_season][target]
    X_test = analysis[analysis['season']>=split_season][variables]
    y_test = analysis[analysis['season']>=split_season][target]

    # alpha chosen by season-grouped CV, not random folds
    train_seasons = analysis[analysis['season'] < split_season]['season']
    inner = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=train_seasons))

    if(True):
        # Gaussian
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        regressor = make_pipeline(StandardScaler(),GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42))
    else:
        # Ridge
        regressor = make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3, 3, 61), cv=inner))

    reg_model = TransformedTargetRegressor(regressor=regressor,func=np.sqrt, inverse_func=np.square) # GF/GA/Pts only — NOT GD
    """
    reg_model = XGBRegressor(
        n_estimators=400,
        max_depth=2,
        learning_rate=0.03,
        reg_lambda=5, 
        min_child_weight=5, 
        colsample_bytree=0.8, # Use 80% of data to grow trees (prevents overfitting)
        random_state=42
    )
    """
    reg_model.fit(X_train, y_train)

    predictions = reg_model.predict(X_test)
    analysis[f'Pred_{target}'] = reg_model.predict(analysis[variables])

    mse = mean_squared_error(y_test, predictions)
    train_r2 = r2_score(y_train, reg_model.predict(X_train))
    test_r2 = r2_score(y_test, predictions)
    
    print(target,"rmse is",mse**0.5)
    print(target,"train R^2 is",train_r2)
    print(target,"test R^2 is",test_r2)
    print()
    return analysis

#%% extract data
#all_tables = teams_data(108,5,10732,24500,"italy-serie-a-2025-2026",argv=sys.argv[1:])
#all_player_tables = players_data(108,5,2626,4659,"italy-serie-a-2010-2011",argv=sys.argv[1:])

#r,t,s,ss,slug = league_codes('Spain',2022)
#all_tables = teams_data(r,t,s,ss,slug,argv=sys.argv[1:])
#full_stats,regression_stats = team_stats_summarize(all_tables,2022)

full_stats,regression_stats = multi_season_team_extract('Spain',2011,2020)

#%% regerssion model
regression_stats = team_stats_regresion(regression_stats,'Pts',2021)
regression_stats = team_stats_regresion(regression_stats,'GF',2021)
regression_stats = team_stats_regresion(regression_stats,'GA',2021)
regression_stats['Pred_GD'] = regression_stats['Pred_GF'] - regression_stats['Pred_GA']