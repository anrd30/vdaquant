"""
Check every decimal number in a LaTeX paper against the fact table
(results/2026-11-02_streaming/analysis/facts.json, built by scripts/build_facts.py).

    python -X utf8 scripts/check_claims.py                       # C:/Users/aniru/Downloads/cvpr2027/paper/main.tex
    python -X utf8 scripts/check_claims.py paper.tex --facts facts.json --allow allow.txt --report out.md

What counts as "matched": a printed number with d decimals matches a fact if the
fact's full-precision value (the *_raw twin), rounded half-up to d decimals, equals
the printed value. So "0.9878" must equal round(raw, 4); "1.7" matches 1.7107 but
"1.8" does not. Signs: an explicit leading "-"/"+" (also "$-$") must agree with the
fact's sign; an unsigned number matches by magnitude. A number followed by "%" or
"\\%" also matches fact*100 (fractions printed as percent).

Match strength (a chance match against ~1600 facts is likely for short numbers):
  tagged     covered by a `% fact:` tag on that line
  specific   untagged, >= 3 decimals and <= 3 facts share that value
  ambiguous  anything else untagged (proves little; tag it, use --scope, or --require-tags)

Extra conventions
  % fact: key1, key2        trailing comment on a line (or a comment-only line, which
                            then applies to the next code line): the listed fact keys
                            must exist AND each key's value must appear among the
                            numbers of that line (otherwise: "bad tag"). Numbers on a
                            tagged line that match NO fact at all are still unmatched.
  % claims: ignore          suppress all numbers on this line.
  --require-tags            strict: every non-allowlisted number must be covered by a
                            `% fact:` tag that matches it.

Allowlist file (default <analysis>/claims_allowlist.txt), one entry per line, '#' comments:
  num 0.05                  exact printed token (as in the source text)
  re  ^1\\.0+$              regex matched against the token
  ctx \\\\tau               regex matched against the (cleaned) source line: ignore all
                            numbers on lines that match (e.g. hyper-parameter lines)
  line 120-135              ignore numbers on source lines 120..135 (inclusive)

Not scanned: preamble (before \\begin{document}), everything after \\end{document}, verbatim /
lstlisting / thebibliography, comments, arguments of \\cite \\ref \\label \\url \\includegraphics
etc., and decimals carrying a LaTeX length unit (0.9\\linewidth, 1.5pt, 2mm). By default only
decimals (with a '.') are checked; --ints also checks integers >= 10 (years/citations then need
allowlisting).

Exit status: 0 = all numbers matched/allowlisted and all tags valid; 1 = unmatched numbers or
bad tags; 2 = usage/input error.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
ANALYSIS = REPO / 'results' / '2026-11-02_streaming' / 'analysis'
DEFAULT_TEX = 'C:/Users/aniru/Downloads/cvpr2027/paper/main.tex'

COMMENT_RE = re.compile(r'(?<!\\)%')
TAG_RE = re.compile(r'fact:\s*(.+)$')
IGNORE_RE = re.compile(r'claims:\s*ignore')
SECTION_RE = re.compile(r'\\(sub)*section\*?\{([^}]*)\}')
NUM_RE = re.compile(r'(?<![\w.])(\d+\.\d+)(?!\.\d)(?![\w])')
INT_RE = re.compile(r'(?<![\w.])(\d{2,})(?![\w.])')
UNIT_AFTER = re.compile(r'^\s*(\\(linewidth|textwidth|columnwidth|textheight|hsize|baselineskip|paperwidth)|'
                        r'(pt|em|ex|cm|mm|in|bp|sp|pc|mu)\b)')
PERCENT_AFTER = re.compile(r'^\s*(\\,|\\!|~)?\s*(\\%|%|\s?percent)')
# whole lines that are layout / setup, never claims
DROP_LINE_RE = re.compile(r'^\s*\\(renewcommand|setlength|definecolor|usepackage|documentclass|def|newcommand|'
                          r'addtolength|setcounter|geometry|pagestyle|hypersetup|colorlet|newcolumntype|'
                          r'bibliographystyle|bibliography|pagenumbering|DeclareMathOperator)\b')
# commands whose (first) braced/bracketed arguments are never claims
ARG_STRIP_RE = re.compile(
    r'\\(cite[a-zA-Z]*|ref|eqref|autoref|cref|Cref|label|url|href|input|include|includegraphics|'
    r'rowcolor|cellcolor|textcolor|color|colorbox|scalebox|resizebox|vspace|hspace|addlinespace|'
    r'rule|cmidrule|hline|footnotemark|bibitem|nocite)\*?(\[[^\]]*\])*(\([^)]*\))?(\{[^}]*\})?')
SKIP_ENV = ('verbatim', 'lstlisting', 'thebibliography', 'comment')
SIGN_BEFORE = re.compile(r'(?:^|[\s(\[{,~$=;:&|])(?:\\!|\\,|\{|\$)*(?P<s>[-+\u2212\u2013])\s*(?:\$|\{|\\!|\\,)*$')


def rnd_abs(x, d):
    return Decimal(repr(abs(float(x)))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)


def ndec(token):
    return len(token.split('.')[1]) if '.' in token else 0


class FactIndex:
    def __init__(self, facts):
        self.raw = {k[:-4]: float(v) for k, v in facts.items() if k.endswith('_raw')}
        # keys with no _raw twin (should not happen for build_facts output) fall back to the value
        for k, v in facts.items():
            if not k.endswith('_raw') and k not in self.raw and isinstance(v, (int, float)):
                self.raw[k] = float(v)
        self._cache = {}

    def lookup(self, d, scale=1.0):
        """{rounded |value| at d decimals -> [(key, signed raw*scale)]}"""
        ck = (d, scale)
        if ck not in self._cache:
            m = defaultdict(list)
            for k, v in self.raw.items():
                m[rnd_abs(v * scale, d)].append((k, v * scale))
            self._cache[ck] = m
        return self._cache[ck]

    def matches(self, token, sign, percent):
        d = ndec(token)
        want = Decimal(token).copy_abs()
        out = []
        for scale in ((1.0, 100.0) if percent else (1.0,)):
            for k, v in self.lookup(d, scale).get(want, []):
                if sign == '-' and not (v < 0 or rnd_abs(v, d) == 0):
                    continue
                if sign == '+' and not (v > 0 or rnd_abs(v, d) == 0):
                    continue
                out.append(k)
        return out

    def near(self, token, k=2, units=3):
        d = ndec(token)
        v = abs(float(token))
        tol = units * 10 ** (-d)
        c = [(abs(abs(x) - v), key, x) for key, x in self.raw.items() if abs(abs(x) - v) <= tol]
        c.sort()
        return [f'{key}={x:.6g}' for _, key, x in c[:k]]


def load_allow(path):
    A = dict(num=set(), re=[], ctx=[], line=[])
    if path and Path(path).exists():
        for raw in Path(path).read_text(encoding='utf-8').splitlines():
            s = raw.split('#', 1)[0].strip()
            if not s:
                continue
            kind, _, arg = s.partition(' ')
            arg = arg.strip()
            try:
                if kind in ('re', 'ctx'):
                    re.compile(arg)
            except re.error as e:
                sys.exit(f'allowlist {path}: bad regex {arg!r}: {e}')
            if kind == 'num':
                A['num'].add(arg)
            elif kind == 're':
                A['re'].append(re.compile(arg))
            elif kind == 'ctx':
                A['ctx'].append(re.compile(arg))
            elif kind == 'line':
                lo, _, hi = arg.partition('-')
                A['line'].append((int(lo), int(hi or lo)))
            else:
                print(f'allowlist: unknown entry kind {kind!r} in {path}', file=sys.stderr)
    return A


def allowed(A, token, line_no, line_text):
    if token in A['num'] or any(r.search(token) for r in A['re']):
        return True
    if any(c.search(line_text) for c in A['ctx']):
        return True
    return any(lo <= line_no <= hi for lo, hi in A['line'])


def scan(tex_path, ints=False):
    """Return (numbers, tag_lines). numbers: list of dicts(line, section, token, sign, percent,
    ctx, tags, ignore, text); tag_lines: {line: (tags, text)}."""
    lines = Path(tex_path).read_text(encoding='utf-8', errors='replace').splitlines()
    in_doc = False
    skip_env = None
    section = '(front matter)'
    pending_tags = []
    out, tag_lines = [], {}
    for i, raw in enumerate(lines, 1):
        if not in_doc:
            if r'\begin{document}' in raw:
                in_doc = True
            continue
        if r'\end{document}' in raw:
            break
        m = re.search(r'\\begin\{(' + '|'.join(SKIP_ENV) + r')\*?\}', raw)
        if m:
            skip_env = m.group(1)
        if skip_env:
            if re.search(r'\\end\{' + skip_env + r'\*?\}', raw):
                skip_env = None
            continue
        cm = COMMENT_RE.search(raw)
        code = raw[:cm.start()] if cm else raw
        comment = raw[cm.start() + 1:] if cm else ''
        tags = []
        tm = TAG_RE.search(comment)
        if tm:
            tags = [t for t in re.split(r'[\s,;]+', tm.group(1).strip()) if t]
        ignore = bool(IGNORE_RE.search(comment))
        if not code.strip():
            if tags:                       # comment-only line: tags apply to the next code line
                pending_tags += tags
            continue
        tags = pending_tags + tags
        pending_tags = []
        sm = SECTION_RE.search(code)
        if sm:
            section = sm.group(2)
        if DROP_LINE_RE.match(code):
            continue
        text = ARG_STRIP_RE.sub(' ', code)
        if tags:
            tag_lines[i] = (tags, text)
        pats = [NUM_RE] + ([INT_RE] if ints else [])
        seen = set()
        for pat in pats:
            for m in pat.finditer(text):
                if m.start() in seen:
                    continue
                seen.add(m.start())
                tok = m.group(1)
                after = text[m.end():]
                if UNIT_AFTER.match(after):
                    continue
                if pat is INT_RE and not PERCENT_AFTER.match(after) and int(tok) < 10:
                    continue
                before = text[:m.start()]
                sg = SIGN_BEFORE.search(before)
                sign = None
                if sg:
                    sign = '-' if sg.group('s') in '-\u2212\u2013' else '+'
                    # "a--b" or "0.1-0.2" are ranges, not signs
                    pre = before[:sg.start('s')].rstrip('$ {}\\!,')
                    if pre and (pre[-1].isdigit() or pre[-1] in '-\u2013\u2212'):
                        sign = None
                out.append(dict(line=i, section=section, token=tok, sign=sign,
                                percent=bool(PERCENT_AFTER.match(after)),
                                ctx=re.sub(r'\s+', ' ', text[max(0, m.start() - 45):m.end() + 25]).strip(),
                                tags=tags, ignore=ignore, text=text))
    return out, tag_lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tex', nargs='?', default=DEFAULT_TEX)
    ap.add_argument('--facts', default=str(ANALYSIS / 'facts.json'))
    ap.add_argument('--allow', default=str(ANALYSIS / 'claims_allowlist.txt'))
    ap.add_argument('--report', default=None, help='write the Markdown report here too')
    ap.add_argument('--json', default=None, help='write a machine-readable report here')
    ap.add_argument('--scope', default=None, help='comma-separated fact-key prefixes to match against '
                                                  '(e.g. sintel.vits,bonn.vits); default: all facts')
    ap.add_argument('--ints', action='store_true', help='also check integers >= 10 (and N%%)')
    ap.add_argument('--require-tags', action='store_true', help='strict: untagged numbers count as unmatched')
    ap.add_argument('--near', action='store_true', help='show nearest facts for unmatched numbers')
    ap.add_argument('--verbose', action='store_true', help='list matched numbers too')
    a = ap.parse_args()

    for p in (a.tex, a.facts):
        if not Path(p).exists():
            print(f'error: {p} not found', file=sys.stderr)
            return 2
    with open(a.facts, encoding='utf-8') as fh:
        facts = json.load(fh)
    if a.scope:
        pre = tuple(x.strip() for x in a.scope.split(',') if x.strip())
        facts = {k: v for k, v in facts.items() if k.startswith(pre)}
    F = FactIndex(facts)
    A = load_allow(a.allow)
    nums, tag_lines = scan(a.tex, a.ints)

    stats = dict(total=0, allowlisted=0, ignored=0, matched=0, matched_tagged=0, matched_specific=0,
                 matched_ambiguous=0, unmatched=0, untagged=0)
    unmatched, matched_rows, bad_tags = [], [], []
    for n in nums:
        stats['total'] += 1
        if n['ignore']:
            stats['ignored'] += 1
            continue
        if allowed(A, n['token'], n['line'], n['text']):
            stats['allowlisted'] += 1
            continue
        keys = F.matches(n['token'], n['sign'], n['percent'])
        n['keys'] = keys
        tagged_hit = [k for k in n['tags'] if k in keys]
        if not keys:
            stats['unmatched'] += 1
            unmatched.append(n)
        elif a.require_tags and not tagged_hit:
            stats['untagged'] += 1
            n['why'] = 'no matching `% fact:` tag'
            unmatched.append(n)
        else:
            stats['matched'] += 1
            n['strength'] = 'tagged' if tagged_hit else ('specific' if ndec(n['token']) >= 3 and len(keys) <= 3 else 'ambiguous')
            stats['matched_' + n['strength']] += 1
            matched_rows.append(n)

    # tag validation: key exists, and its value occurs on the tagged line
    n_tags = 0
    by_line = defaultdict(list)
    for n in nums:
        by_line[n['line']].append(n)
    for ln, (tags, _) in tag_lines.items():
        for k in tags:
            n_tags += 1
            if k not in F.raw:
                bad_tags.append((ln, k, 'unknown fact key'))
                continue
            if not any(k in F.matches(n['token'], n['sign'], n['percent']) for n in by_line.get(ln, [])):
                near = ', '.join(sorted({n['token'] for n in by_line.get(ln, [])})) or 'no numbers'
                bad_tags.append((ln, k, f'value {F.raw[k]:.6g} not found on the line (numbers there: {near})'))

    L = [f'# Claim check: {a.tex}', '',
         f'facts: {len(F.raw)} ({a.facts}); allowlist: {a.allow if Path(a.allow).exists() else "(none)"}', '',
         f'numbers checked: {stats["total"]}  matched: {stats["matched"]} '
         f'(tagged {stats["matched_tagged"]}, specific {stats["matched_specific"]}, ambiguous {stats["matched_ambiguous"]})  '
         f'allowlisted: {stats["allowlisted"]}  ignored: {stats["ignored"]}  '
         f'UNMATCHED: {stats["unmatched"] + stats["untagged"]}  fact tags: {n_tags} (bad: {len(bad_tags)})',
         '',
         'Match strength: tagged = covered by a `% fact:` tag; specific = untagged, >= 3 decimals and <= 3 facts '
         'share the value; ambiguous = anything else untagged (many facts share such values, so the match proves '
         'little: tag it, narrow with --scope, or use --require-tags).', '']
    if unmatched:
        hdr = '| line | section | number | context |' + (' nearest facts |' if a.near else '')
        L += ['## Unmatched numbers', '', hdr, '|---|---|---|---|' + ('---|' if a.near else '')]
        for n in unmatched:
            tok = (n['sign'] or '') + n['token'] + ('%' if n['percent'] else '')
            ctx = ((n['why'] + ': ' if n.get('why') else '') + n['ctx']).replace('|', r'\|')
            row = f'| {n["line"]} | {n["section"][:28]} | `{tok}` | `{ctx}` |'
            if a.near:
                row += ' ' + '; '.join(F.near(n['token'])) + ' |'
            L.append(row)
        L.append('')
    if bad_tags:
        L += ['## Bad fact tags', '', '| line | key | problem |', '|---|---|---|']
        L += [f'| {ln} | `{k}` | {why} |' for ln, k, why in bad_tags]
        L.append('')
    if a.verbose and matched_rows:
        L += ['## Matched', '', '| line | number | strength | facts (first 3) |', '|---|---|---|---|']
        for n in matched_rows:
            ks = n['keys']
            more = f' (+{len(ks) - 3})' if len(ks) > 3 else ''
            L.append(f'| {n["line"]} | `{(n["sign"] or "") + n["token"]}` | {n["strength"]} | '
                     f'{", ".join(ks[:3])}{more} |')
        L.append('')
    ok = not unmatched and not bad_tags
    L.append('RESULT: ' + ('OK' if ok else 'FAIL'))
    text = '\n'.join(L)
    print(text)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(text, encoding='utf-8')
    if a.json:
        Path(a.json).write_text(json.dumps(dict(
            stats=stats, unmatched=[{k: v for k, v in n.items() if k != 'text'} for n in unmatched],
            bad_tags=bad_tags), indent=1), encoding='utf-8')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
