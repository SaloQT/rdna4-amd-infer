// Stand-in for the GitHub CLI over a local bare "origin". PRs live in gh.json; a merge fast-forwards origin's main to
// the PR branch. Mode file gh.mode: ok | createfail | mergefail | conflict-once | movemain-once.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const here = process.env.RSI_TEST_DIR; /* the test work directory (run_tests.mjs sets it): origin.git, PR state, modes */
if (!here) { console.error('stub_gh: RSI_TEST_DIR is not set'); process.exit(2); }
const origin = path.join(here, 'origin.git');
const db = path.join(here, 'gh.json');
const modeFile = path.join(here, 'gh.mode');
const mode = fs.existsSync(modeFile) ? fs.readFileSync(modeFile, 'utf8').trim() : 'ok';
const st = fs.existsSync(db) ? JSON.parse(fs.readFileSync(db, 'utf8')) : { prs: [] };
const save = () => fs.writeFileSync(db, JSON.stringify(st, null, 1));
const once = () => fs.writeFileSync(modeFile, 'ok');
const g = (...a) => spawnSync('git', ['-C', origin, ...a], { encoding: 'utf8' });
const a = process.argv.slice(2);
fs.appendFileSync(path.join(here, 'gh.calls'), a.join(' ') + '\n');
const find = (x) => st.prs.find((p) => p.url === x || p.branch === x);
if (a[0] === 'auth') process.exit(0);
if (a[0] !== 'pr') process.exit(2);
if (a[1] === 'view') {
  if (mode === 'viewfail-after-merge' && st.prs.some((p) => p.state === 'MERGED')) { console.error('HTTP 502'); process.exit(1); }
  const p = find(a[2]);
  if (!p) { console.error('no pull requests found'); process.exit(1); }
  const q = a[a.indexOf('-q') + 1];
  console.log(q.includes('url') ? `${p.url} ${p.state}` : p.state);
  process.exit(0);
}
if (a[1] === 'create') {
  if (mode === 'createfail') { console.error('HTTP 502'); process.exit(1); }
  const br = a[a.indexOf('--head') + 1];
  if (g('rev-parse', '--verify', '-q', 'refs/heads/' + br).status !== 0) { console.error('branch not pushed'); process.exit(1); }
  const n = st.prs.length + 1;
  st.prs.push({ n, url: `https://example.invalid/pr/${n}`, branch: br, state: 'OPEN' });
  save();
  if (mode === 'movedocs-once') { /* someone merges a docs-only change meanwhile */
    const tmp = path.join(here, 'docs-clone');
    fs.rmSync(tmp, { recursive: true, force: true });
    spawnSync('git', ['clone', '-q', origin, tmp]);
    fs.appendFileSync(path.join(tmp, 'README.md'), '\ndocs change\n');
    spawnSync('git', ['-C', tmp, 'commit', '-q', '-am', 'docs only']);
    spawnSync('git', ['-C', tmp, 'push', '-q', 'origin', 'main']);
    fs.rmSync(tmp, { recursive: true, force: true });
    once();
  }
  if (mode === 'movemain-once') { /* someone lands an engine change meanwhile (another file than the PR's) */
    const tmp = path.join(here, 'eng-clone');
    fs.rmSync(tmp, { recursive: true, force: true });
    spawnSync('git', ['clone', '-q', origin, tmp]);
    fs.appendFileSync(path.join(tmp, 'engine', 'kernels', 'ie_inkling_r1_g02.inc'), '/* unrelated change */\n');
    spawnSync('git', ['-C', tmp, 'commit', '-q', '-am', 'unrelated change']);
    spawnSync('git', ['-C', tmp, 'push', '-q', 'origin', 'main']);
    fs.rmSync(tmp, { recursive: true, force: true });
    once();
  }
  console.log(`https://example.invalid/pr/${n}`);
  process.exit(0);
}
if (a[1] === 'merge') {
  const p = find(a[2]);
  if (!p || p.state !== 'OPEN') { console.error('no open PR'); process.exit(1); }
  if (mode === 'mergefail') { console.error('GraphQL: something went wrong'); process.exit(1); }
  if (mode === 'conflict-once') { once(); console.error('Pull request is not mergeable: the merge commit cannot be cleanly created'); process.exit(1); }
  const head = g('rev-parse', 'refs/heads/' + p.branch).stdout.trim();
  const base = g('rev-parse', 'main').stdout.trim();
  if (g('merge-base', '--is-ancestor', base, head).status === 0) g('update-ref', 'refs/heads/main', head);
  else { /* main moved: replay the branch's commits onto it, as a rebase-merge does */
    const tmp = path.join(here, 'merge-clone');
    fs.rmSync(tmp, { recursive: true, force: true });
    spawnSync('git', ['clone', '-q', origin, tmp]);
    const mb = spawnSync('git', ['-C', tmp, 'merge-base', 'origin/main', 'origin/' + p.branch], { encoding: 'utf8' }).stdout.trim();
    const cp = spawnSync('git', ['-C', tmp, 'cherry-pick', `${mb}..origin/${p.branch}`], { encoding: 'utf8' });
    if (cp.status !== 0) { fs.rmSync(tmp, { recursive: true, force: true }); console.error('not mergeable: conflict'); process.exit(1); }
    spawnSync('git', ['-C', tmp, 'push', '-q', 'origin', 'HEAD:main']);
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  g('update-ref', '-d', 'refs/heads/' + p.branch);
  p.state = 'MERGED';
  save();
  process.exit(0);
}
if (a[1] === 'close') {
  const p = find(a[2]);
  if (p) { p.state = 'CLOSED'; g('update-ref', '-d', 'refs/heads/' + p.branch); save(); }
  process.exit(0);
}
process.exit(2);
