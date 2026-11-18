"""
NYC 311 Service Performance dashboard - data refresh.

Pulls the most recent full month of 311 service requests, the city's published
service-level targets and the monthly channel history from NYC Open Data, then
writes the small JSON files the dashboard page reads (nyc311/data/*.json).

Run by GitHub Actions every week (.github/workflows/nyc311-refresh.yml).
Local test without internet:  python build_data.py --requests r.csv --sla s.csv --trend t.csv
Force a month:                 PERIOD=2026-10 python build_data.py
"""
import argparse, io, json, os, re, sys, time, pathlib, datetime as dt
import numpy as np
import pandas as pd

BASE = 'https://data.cityofnewyork.us/resource'
REQUESTS_ID, SLA_ID = 'erm2-nwe9', 'cs9t-e3x8'
OUT = pathlib.Path(__file__).parent / 'data'

AGENCY_CODES = {
    'Department of Consumer and Worker Protection': 'DCWP', 'Department of Health and Mental Hygiene': 'DOHMH',
    'Department of Parks and Recreation': 'DPR', 'Department of Transportation': 'DOT', 'Taxi and Limousine Commission': 'TLC',
    'Department of Housing Preservation and Development': 'HPD', 'Department of Sanitation': 'DSNY', 'Department of Buildings': 'DOB',
    'Department of Environmental Protection': 'DEP', 'New York City Police Department': 'NYPD', 'Office of Technology and Innovation': 'OTI',
    'Department of Education': 'DOE', 'Department of Homeless Services': 'DHS'}


def soql_csv(dataset, params, page=50000):
    """Download a SoQL query as CSV, page by page."""
    import requests
    frames, offset = [], 0
    while True:
        p = dict(params, **{'$limit': page, '$offset': offset})
        for attempt in range(4):
            r = requests.get(f'{BASE}/{dataset}.csv', params=p, timeout=180)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        frames.append(df)
        print(f'{dataset}: {offset + len(df):,} rows', flush=True)
        if len(df) < page:
            return pd.concat(frames, ignore_index=True)
        offset += page


def period_bounds():
    """The last full calendar month (UTC), or PERIOD=YYYY-MM if set."""
    if os.environ.get('PERIOD'):
        start = dt.datetime.strptime(os.environ['PERIOD'], '%Y-%m')
    else:
        first_this = dt.datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start = (first_this - dt.timedelta(days=1)).replace(day=1)
    end = (start + dt.timedelta(days=32)).replace(day=1)
    return start, end


def load(args, start, end):
    if args.requests:
        return pd.read_csv(args.requests), pd.read_csv(args.sla), pd.read_csv(args.trend)
    fmt = '%Y-%m-%dT%H:%M:%S'
    req = soql_csv(REQUESTS_ID, {
        '$select': 'unique_key,created_date,closed_date,agency,complaint_type,borough,status,open_data_channel_type',
        '$where': f"created_date >= '{start.strftime(fmt)}' AND created_date < '{end.strftime(fmt)}'",
        '$order': 'unique_key'})
    sla = soql_csv(SLA_ID, {})
    trend = soql_csv(REQUESTS_ID, {
        '$select': 'date_trunc_ym(created_date) AS month, agency, open_data_channel_type, count(*) AS requests',
        '$where': "created_date >= '2022-01-01T00:00:00'",
        '$group': 'month, agency, open_data_channel_type', '$order': 'month'})
    return req, sla, trend


def to_days(x):
    m = re.match(r'(\d+)\s*(day|hour)', str(x))
    return np.nan if not m else int(m[1]) * (1 if m[2] == 'day' else 1 / 24)


def sla_label(d):
    h = round(d * 24)
    return f'{h} hour' + ('s' if h != 1 else '') if d < 1 else f'{int(d)} day' + ('s' if d != 1 else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--requests'); ap.add_argument('--sla'); ap.add_argument('--trend'); ap.add_argument('--asof')
    args = ap.parse_args()
    start, end = period_bounds()
    r, s, trend = load(args, start, end)

    r['created_date'] = pd.to_datetime(r.created_date)
    r['closed_date'] = pd.to_datetime(r.closed_date)
    r = r[(r.created_date >= start) & (r.created_date < end)].copy()
    if r.empty:
        sys.exit('No requests found for the period - nothing written.')

    # service-level targets: one per agency + complaint type (the most common one listed)
    s['agency_code'] = s.agency.map(AGENCY_CODES)
    s['sla'] = s.sla_days.map(to_days)
    s['key'] = s.problem.str.lower().str.strip()
    sla = (s.dropna(subset=['agency_code', 'sla']).groupby(['agency_code', 'key']).sla
           .agg(lambda v: v.mode().iloc[0]).rename('sla_days').reset_index())
    r['key'] = r.complaint_type.str.lower().str.strip()
    r = r.merge(sla, left_on=['agency', 'key'], right_on=['agency_code', 'key'], how='left')

    r['hrs'] = (r.closed_date - r.created_date).dt.total_seconds() / 3600
    r['known_ch'] = r.open_data_channel_type.isin(['ONLINE', 'MOBILE', 'PHONE'])
    r['digital'] = np.where(r.known_ch, r.open_data_channel_type.isin(['ONLINE', 'MOBILE']), np.nan)
    closed = r.closed_date.notna()
    bad = closed & (r.hrs < 1 / 60)          # closed before opening or in under a minute
    valid = closed & ~bad
    r['in_sla'] = np.where(valid & r.sla_days.notna(), r.hrs <= r.sla_days * 24, np.nan)
    asof = pd.Timestamp(args.asof) if args.asof else (r.closed_date.max() if args.requests else pd.Timestamp.utcnow().tz_localize(None))
    r['age_days'] = (asof - r.created_date).dt.total_seconds() / 86400

    def perf(g):
        v = g[valid.loc[g.index]]
        t = g[valid.loc[g.index] & g.sla_days.notna()]
        late = t[t.in_sla == 0]
        mode = g.sla_days.dropna().mode()
        return pd.Series({
            'requests': len(g),
            'digital_pct': round(100 * g.digital.mean(), 1) if g.known_ch.any() else None,
            'open': int(g.closed_date.isna().sum()),
            'median_hrs': round(v.hrs.median(), 1) if len(v) else None,
            'p90_hrs': round(v.hrs.quantile(.9), 1) if len(v) else None,
            'median_days': round(v.hrs.median() / 24, 1) if len(v) else None,
            'p90_days': round(v.hrs.quantile(.9) / 24, 1) if len(v) else None,
            'sla_days': float(mode.iloc[0]) if len(mode) else None,
            'sla_label': sla_label(mode.iloc[0]) if len(mode) else None,
            'sla_measured': int(len(t)),
            'within_sla_pct': round(100 * t.in_sla.mean(), 1) if len(t) else None,
            'late': int(len(late)),
            'days_over_target': round(((late.hrs - late.sla_days * 24) / 24).median(), 1) if len(late) else None})

    ag = r.groupby('agency').apply(perf).reset_index()
    ag = ag[ag.requests >= 500].sort_values('requests', ascending=False)
    ct = r.groupby(['agency', 'complaint_type']).apply(perf).reset_index().sort_values('requests', ascending=False).head(25)

    op = r[r.closed_date.isna()].copy()
    op['age_band'] = pd.cut(op.age_days, [-1, 14, 30, 10000], labels=['0-14 days', '15-30 days', '31+ days'])
    bl = op.groupby(['agency', 'age_band'], observed=False).size().rename('open').reset_index()
    bl = bl[bl.agency.isin(op.agency.value_counts().head(8).index)]
    bl['age_band'] = bl.age_band.astype(str)

    t = r[valid & r.sla_days.notna()]
    total_late = int((t.in_sla == 0).sum())
    top = ag.sort_values('late', ascending=False).iloc[0]
    hard = ct[(ct.sla_measured >= 1000) & ct.within_sla_pct.notna()].sort_values('within_sla_pct').iloc[0]
    lowdig = ag[ag.digital_pct.notna() & (ag.requests >= 1000)].sort_values('digital_pct').iloc[0]
    kpi = [{
        'period_month': start.strftime('%Y-%m'), 'period_label': start.strftime('%B %Y'),
        'prev_year_month': start.replace(year=start.year - 1).strftime('%Y-%m'),
        'prev_year_label': start.replace(year=start.year - 1).strftime('%b %Y'),
        'as_of': asof.strftime('%Y-%m-%d'), 'as_of_label': f'{asof.day} {asof.strftime("%b %Y")}',
        'requests': len(r), 'agencies': int(len(ag)),
        'digital_pct': round(100 * r.digital.mean(), 1),
        'phone_pct': round(100 * (r.open_data_channel_type[r.known_ch] == 'PHONE').mean(), 1),
        'unknown_channel_pct': round(100 * (~r.known_ch).mean(), 1),
        'median_hrs': round(r[valid].hrs.median(), 1), 'open': int((~closed).sum()), 'open_pct': round(100 * (~closed).mean(), 1),
        'within_sla_pct': round(100 * t.in_sla.mean(), 1), 'late': total_late,
        'sla_coverage_pct': round(100 * r.sla_days.notna().mean(), 1), 'excluded_bad_times': int(bad.sum()),
        'top_late_agency': top.agency, 'top_late_share': round(100 * top.late / total_late, 1) if total_late else None,
        'hard_type': hard.complaint_type, 'hard_agency': hard.agency, 'hard_target': hard.sla_label, 'hard_within_pct': hard.within_sla_pct,
        'low_digital_agency': lowdig.agency, 'low_digital_pct': lowdig.digital_pct}]

    trend = trend.rename(columns={'open_data_channel_type': 'channel'})
    trend['month'] = trend.month.astype(str).str.slice(0, 7)
    trend['requests'] = trend.requests.astype(int)
    trend = trend.groupby(['month', 'channel'], as_index=False).requests.sum()

    meta = [{'label': 'Input', 'value': f'NYC Open Data {REQUESTS_ID} (requests created {start:%d %b %Y} to {end - dt.timedelta(days=1):%d %b %Y}) and {SLA_ID}'}]
    OUT.mkdir(exist_ok=True)
    for name, df in [('kpi', pd.DataFrame(kpi)), ('agency_perf', ag), ('complaint_perf', ct), ('backlog', bl), ('trend', trend)]:
        rows = json.loads(df.to_json(orient='records'))
        (OUT / f'{name}.json').write_text(json.dumps({'rows': rows, 'meta': meta}, separators=(',', ':')))
        print(f'wrote {name}.json ({len(rows)} rows)')


if __name__ == '__main__':
    main()
