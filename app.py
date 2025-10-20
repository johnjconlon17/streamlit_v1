# app.py  — IR efficiency vs ORIGINAL and vs CURRENT baselines
import streamlit as st
import sqlite3
import random
import pandas as pd
import numpy as np
from itertools import permutations
from contextlib import closing
import time

# ─────────────────────────────────────────────
# Page config FIRST
# ─────────────────────────────────────────────
st.set_page_config(page_title="Bilateral Trading Game", page_icon="🔁", layout="wide")

# ─────────────────────────────────────────────
# Admin gate (URL param only)
# ─────────────────────────────────────────────
params = st.query_params
is_admin_mode = "admin" in params and str(params["admin"]).lower() in ["1", "true", "yes"]
role_options = ["Student"] + (["Social Planner"] if is_admin_mode else [])

# ─────────────────────────────────────────────
# Config + auto-refresh defaults
# ─────────────────────────────────────────────
GOODS = [
    "Gizmo", "Whatsit", "Thingamabob", "Doohickey", "Widget",
    "Contraption", "Gadget", "Whatchamacallit", "Doodad", "Thingy",
    "Gubbins", "Apparatus", "Mechanism", "Rigamarole", "Oddment"
]
DB_PATH = "class_trade.db"

if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "refresh_interval" not in st.session_state:
    st.session_state["refresh_interval"] = 5

# ─────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=3000;")
    return conn

def init_db():
    conn = get_conn()
    with closing(conn.cursor()) as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                grp TEXT NOT NULL,
                UNIQUE(name, grp)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS endowments (
                user_id INTEGER PRIMARY KEY,
                good TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        # NEW: remember each user's original good
        cur.execute("""
            CREATE TABLE IF NOT EXISTS initial_endowments (
                user_id INTEGER PRIMARY KEY,
                good TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                user_id INTEGER NOT NULL,
                good TEXT NOT NULL,
                utility INTEGER NOT NULL,
                PRIMARY KEY (user_id, good),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grp TEXT NOT NULL,
                proposer_id INTEGER NOT NULL,
                recipient_id INTEGER NOT NULL,
                proposer_good TEXT NOT NULL,
                recipient_good TEXT NOT NULL,
                status TEXT NOT NULL,     -- 'pending', 'accepted', 'declined', 'cancelled'
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def fetchone(q, p=()):
    c = get_conn(); cur = c.cursor()
    cur.execute(q, p); r = cur.fetchone()
    cur.close(); return r

def fetchall(q, p=()):
    c = get_conn(); cur = c.cursor()
    cur.execute(q, p); r = cur.fetchall()
    cur.close(); return r

def execute(q, p=()):
    c = get_conn(); cur = c.cursor()
    cur.execute(q, p); c.commit()
    cur.close()

# resets
def reset_all_data():
    execute("DELETE FROM trades")
    execute("DELETE FROM preferences")
    execute("DELETE FROM endowments")
    execute("DELETE FROM initial_endowments")
    execute("DELETE FROM users")
    execute("VACUUM")

def reset_group(grp):
    uids = [u[0] for u in get_group_users(grp)]
    execute("DELETE FROM trades WHERE grp=?", (grp,))
    for uid in uids:
        execute("DELETE FROM preferences WHERE user_id=?", (uid,))
        execute("DELETE FROM endowments WHERE user_id=?", (uid,))
        execute("DELETE FROM initial_endowments WHERE user_id=?", (uid,))
        execute("DELETE FROM users WHERE id=?", (uid,))
    execute("VACUUM")

# ─────────────────────────────────────────────
# Core model logic
# ─────────────────────────────────────────────
def get_group_users(grp):
    return fetchall("SELECT id, name FROM users WHERE grp=? ORDER BY name", (grp,))

def active_goods_in_group(grp):
    rows = fetchall("""
        SELECT e.good
        FROM endowments e
        JOIN users u ON u.id = e.user_id
        WHERE u.grp=?
    """, (grp,))
    return sorted({r[0] for r in rows})

def pick_start_good_unique(grp):
    used = set(active_goods_in_group(grp))
    available = [g for g in GOODS if g not in used]
    return random.choice(available if available else GOODS)

def get_or_create_user(name, grp):
    row = fetchone("SELECT id FROM users WHERE name=? AND grp=?", (name, grp))
    if row:
        return row[0]
    execute("INSERT INTO users (name, grp) VALUES (?,?)", (name, grp))
    uid = fetchone("SELECT id FROM users WHERE name=? AND grp=?", (name, grp))[0]
    start_good = pick_start_good_unique(grp)
    execute("INSERT INTO endowments VALUES (?,?)", (uid, start_good))
    execute("INSERT OR REPLACE INTO initial_endowments VALUES (?,?)", (uid, start_good))
    for g in GOODS:
        u = random.randint(1, 10)
        execute("INSERT INTO preferences VALUES (?,?,?)", (uid, g, u))
    return uid

def get_user_endowment(uid):
    r = fetchone("SELECT good FROM endowments WHERE user_id=?", (uid,))
    return r[0] if r else None

def get_initial_endowment(uid):
    r = fetchone("SELECT good FROM initial_endowments WHERE user_id=?", (uid,))
    if r:
        return r[0]
    # Backfill if table was added after users existed
    cur = get_user_endowment(uid)
    if cur:
        execute("INSERT OR REPLACE INTO initial_endowments VALUES (?,?)", (uid, cur))
        return cur
    return None

def get_user_prefs_df(uid):
    r = fetchall("SELECT good, utility FROM preferences WHERE user_id=? ORDER BY good", (uid,))
    return pd.DataFrame(r, columns=["Good", "Utility"])

def current_allocation_for_group(grp):
    r = fetchall("""
        SELECT u.id, u.name, e.good
        FROM users u JOIN endowments e ON e.user_id=u.id
        WHERE u.grp=?
        ORDER BY u.name
    """, (grp,))
    return pd.DataFrame(r, columns=["user_id", "Name", "Good"])

def original_allocation_for_group(grp):
    r = fetchall("""
        SELECT u.id, u.name, ie.good
        FROM users u JOIN initial_endowments ie ON ie.user_id=u.id
        WHERE u.grp=?
        ORDER BY u.name
    """, (grp,))
    # If some users were missing (older games), backfill from current
    if not r:
        cur = current_allocation_for_group(grp)
        for uid, nm, gd in cur.itertuples(index=False):
            execute("INSERT OR REPLACE INTO initial_endowments VALUES (?,?)", (int(uid), str(gd)))
        r = [(int(uid), str(nm), str(gd)) for uid, nm, gd in cur.itertuples(index=False)]
    return pd.DataFrame(r, columns=["user_id", "Name", "Good"])

def preferences_matrix(grp, users_df, items):
    mats = []
    for uid in users_df["user_id"]:
        prefs = dict(fetchall("SELECT good, utility FROM preferences WHERE user_id=?", (uid,)))
        mats.append([prefs[g] for g in items])
    return np.array(mats, int)

def propose_trade(grp, pid, rid):
    if pid == rid:
        return "Cannot trade with yourself."
    pg, rg = get_user_endowment(pid), get_user_endowment(rid)
    if not (pg and rg):
        return "Endowment missing."
    row = fetchone(
        """SELECT id FROM trades WHERE grp=? AND proposer_id=? AND recipient_id=?
           AND proposer_good=? AND recipient_good=? AND status='pending'""",
        (grp, pid, rid, pg, rg)
    )
    if row:
        return "Trade already pending."
    execute(
        """INSERT INTO trades (grp, proposer_id, recipient_id, proposer_good, recipient_good, status)
           VALUES (?,?,?,?,?,'pending')""",
        (grp, pid, rid, pg, rg)
    )
    return "Trade proposed."

def incoming_trades(uid):
    return fetchall(
        """SELECT t.id, u.name, t.proposer_good, t.recipient_good, t.status, t.ts
           FROM trades t JOIN users u ON u.id=t.proposer_id
           WHERE t.recipient_id=? AND t.status='pending'""",
        (uid,)
    )

def outgoing_trades(uid):
    return fetchall(
        """SELECT t.id, v.name, t.proposer_good, t.recipient_good, t.status, t.ts
           FROM trades t JOIN users v ON v.id=t.recipient_id
           WHERE t.proposer_id=? AND t.status='pending'""",
        (uid,)
    )

def update_trade_status(tid, status):
    execute("UPDATE trades SET status=? WHERE id=?", (status, tid))

def accept_trade(tid):
    r = fetchone(
        "SELECT grp, proposer_id, recipient_id, proposer_good, recipient_good, status FROM trades WHERE id=?",
        (tid,)
    )
    if not r:
        return "Trade not found."
    grp, pid, rid, pg, rg, status = r
    if status != "pending":
        return "Trade no longer pending."
    if get_user_endowment(pid) != pg or get_user_endowment(rid) != rg:
        update_trade_status(tid, "declined")
        return "Offer invalid."
    execute("UPDATE endowments SET good=? WHERE user_id=?", (rg, pid))
    execute("UPDATE endowments SET good=? WHERE user_id=?", (pg, rid))
    update_trade_status(tid, "accepted")
    return "Trade accepted and executed."

def decline_trade(tid):
    update_trade_status(tid, "declined")

def cancel_outgoing(tid, uid):
    r = fetchone("SELECT proposer_id, status FROM trades WHERE id=?", (tid,))
    if r and r[0] == uid and r[1] == "pending":
        update_trade_status(tid, "cancelled")

def total_current_utility(grp):
    r = fetchall(
        """SELECT p.utility FROM users u
           JOIN endowments e ON e.user_id=u.id
           JOIN preferences p ON p.user_id=u.id AND p.good=e.good
           WHERE u.grp=?""",
        (grp,)
    )
    return sum(x[0] for x in r)

def optimal_assignment(grp):
    df = current_allocation_for_group(grp)
    n = len(df)
    if n == 0:
        return 0, df, []
    items = list(df["Good"])
    util = preferences_matrix(grp, df, items)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-util)
        best = int(util[r, c].sum())
        assign = [(df.iloc[i]["Name"], items[c[i]], int(util[i, c[i]])) for i in range(n)]
        return best, df, assign
    except Exception:
        remaining = set(range(n))
        chosen = {}
        total = 0
        for i in range(n):
            j = max(remaining, key=lambda k: util[i, k])
            chosen[i] = j; remaining.remove(j)
            total += int(util[i, j])
        assign = [(df.iloc[i]["Name"], items[chosen[i]], int(util[i, chosen[i]])) for i in range(n)]
        return total, df, assign

# ─────────────────────────────────────────────
# IR-Pareto tools parameterized by a baseline
# ─────────────────────────────────────────────
def _planner_state(grp):
    """Return (users_df, items_labels, util_matrix, label_to_index) at CURRENT items ordering."""
    users_df = current_allocation_for_group(grp)   # user_id, Name, Good
    items = list(users_df["Good"])                 # labels of current items (columns of util)
    util = preferences_matrix(grp, users_df, items)
    label_to_index = {label: j for j, label in enumerate(items)}
    return users_df, items, util, label_to_index

def _baseline_self_vector(util, label_to_index, baseline_labels):
    """base_self[i] = utility for user i at baseline allocation (by item label)."""
    return np.array([util[i, label_to_index[baseline_labels[i]]] for i in range(len(baseline_labels))])

def closest_ir_efficient_improvement_given_baseline(grp, baseline_labels):
    """
    Greedy TTC-style over the graph of acceptable edges: util[i,j] >= base_self[i],
    'strict' if >. Repeatedly execute the strictly-improving cycle with the FEWEST
    strictly-better participants, until none remain.
    Returns (total_utility, strictly_better_count, assignment[(Name, Good)]).
    """
    users_df, items, util, label_to_index = _planner_state(grp)
    n = len(users_df)
    if n == 0:
        return 0, 0, []
    base_self = _baseline_self_vector(util, label_to_index, baseline_labels)

    cur_items = items[:]  # labels by user index
    strictly_better = set()

    def build_targets():
        # point to highest acceptable column j (>= base_self[i]); tie-break stay on current item index
        targets = []
        for i in range(n):
            best_v = -10**9
            best_j = label_to_index[cur_items[i]]  # staying index
            for j in range(len(items)):
                v = util[i, j]
                if v < base_self[i]:  # not acceptable
                    continue
                if v > best_v or (v == best_v and j == best_j):
                    best_v = v; best_j = j
            targets.append(best_j)
        return targets

    def strict_edge(i, j):
        return util[i, j] > base_self[i]

    guard = 0
    while True:
        guard += 1
        if guard > n * 5:
            break
        targets = build_targets()

        # find cycles in functional graph
        visited = [0]*n
        cycles = []
        for s in range(n):
            if visited[s]: continue
            seen = {}
            cur = s
            while not visited[cur]:
                seen[cur] = len(seen)
                visited[cur] = 1
                cur = targets[cur]
                if cur in seen:
                    idx0 = seen[cur]
                    cyc = [k for k, idx in sorted(seen.items(), key=lambda kv: kv[1]) if idx >= idx0]
                    cycles.append(cyc)
                    break

        # keep cycles with at least one strict edge (true IR improvement)
        improving = []
        for cyc in cycles:
            if any(strict_edge(i, targets[i]) for i in cyc if targets[i] != label_to_index[cur_items[i]]):
                improving.append(cyc)
        if not improving:
            break

        # choose the cycle with the FEWEST strictly-better participants
        def strict_count(cyc):
            return sum(1 for i in cyc if targets[i] != label_to_index[cur_items[i]] and strict_edge(i, targets[i]))
        improving.sort(key=lambda cyc: (strict_count(cyc), len(cyc)))
        chosen = improving[0]

        # execute chosen cycle (each i gets item at targets[i])
        new_items = cur_items[:]
        for i in chosen:
            new_items[i] = items[targets[i]]  # targets index refers to column j → label items[j]
            if strict_edge(i, targets[i]):
                strictly_better.add(i)
        cur_items = new_items

    # total utility for final cur_items
    total = int(sum(util[i, label_to_index[cur_items[i]]] for i in range(n)))
    assign = [(users_df.iloc[i]["Name"], cur_items[i]) for i in range(n)]
    return total, int(len(strictly_better)), assign

def util_max_ir_improvement_given_baseline(grp, baseline_labels):
    """
    Maximize total utility subject to util[i,j] >= base_self[i].
    """
    users_df, items, util, label_to_index = _planner_state(grp)
    n = len(users_df)
    if n == 0:
        return 0, 0, []
    base_self = _baseline_self_vector(util, label_to_index, baseline_labels)

    allowed = util >= base_self.reshape(-1, 1)
    restricted = util.astype(float).copy()
    restricted[~allowed] = -1e6  # disallow

    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-restricted)  # maximize
        total = int(util[r, c].sum())
        strictly = int(sum(1 for i in range(n) if util[i, c[i]] > base_self[i]))
        assign = [(users_df.iloc[i]["Name"], items[c[i]]) for i in range(n)]
        return total, strictly, assign
    except Exception:
        # greedy fallback respecting IR
        remaining = set(range(n))
        chosen = {}
        total = 0
        strictly = 0
        for i in range(n):
            choices = [j for j in remaining if util[i, j] >= base_self[i]]
            if not choices:
                j = next(iter(remaining))
            else:
                j = max(choices, key=lambda jj: util[i, jj])
            chosen[i] = j; remaining.remove(j)
            total += int(util[i, j]); strictly += int(util[i, j] > base_self[i])
        assign = [(users_df.iloc[i]["Name"], items[chosen[i]]) for i in range(n)]
        return int(total), strictly, assign

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
init_db()

with st.sidebar:
    st.header("Bilateral Trading Game")
    role = st.radio("Select role", role_options, index=0)

    # Prefill from prior login if available
    prev_name, prev_grp = (st.session_state.get("who") or ("", ""))

    grp = st.text_input("Group code", value=prev_grp, placeholder="e.g., A1 or econ101-1", key="grp_input")

    if role == "Student":
        name = st.text_input("Your name", value=prev_name, placeholder="First Last", key="name_input")
        start_btn = st.button("Enter / Join Group")

    # Auto-refresh controls
    default_interval = 3 if role == "Social Planner" else 8
    auto = st.toggle("Auto-refresh", value=st.session_state["auto_refresh"])
    secs = st.number_input("Refresh every (s)", 2, 60, st.session_state["refresh_interval"], step=1)
    if auto != st.session_state["auto_refresh"] or secs != st.session_state["refresh_interval"]:
        st.session_state["auto_refresh"] = auto
        st.session_state["refresh_interval"] = int(secs)
        st.session_state["next_refresh_time"] = time.time() + st.session_state["refresh_interval"]

colA, colB = st.columns(2)

# ── Student Role ─────────────────────────────
if role == "Student":
    grp_clean = (grp or "").strip()
    name_clean = (name or "").strip() if "name" in locals() else ""

    if "user_id" not in st.session_state:
        if not grp_clean or not name_clean:
            st.info("Enter group and name in the sidebar to begin.")
        elif start_btn or True:
            uid = get_or_create_user(name_clean, grp_clean)
            st.session_state["user_id"] = uid
            st.session_state["who"] = (name_clean, grp_clean)
            st.rerun()

    if "user_id" in st.session_state:
        uid = st.session_state["user_id"]
        name_clean, grp_clean = st.session_state["who"]

        with colA:
            st.subheader("Your Endowment")
            good = get_user_endowment(uid)
            st.metric("You currently hold", good)

            st.subheader("Your Preferences (active goods)")
            prefs_df = get_user_prefs_df(uid)
            active = active_goods_in_group(grp_clean)
            if active:
                prefs_df = prefs_df[prefs_df["Good"].isin(active)].sort_values("Good")
            st.dataframe(prefs_df, width="stretch", hide_index=True)

            st.subheader("Propose a Trade")
            others = [(u[1], u[0]) for u in get_group_users(grp_clean) if u[0] != uid]
            if not others:
                st.warning("No one else yet.")
            else:
                partner_names = [x[0] for x in others]
                target = st.selectbox("Partner", partner_names, key="partner_select")
                tid = dict(others)[target]
                their = get_user_endowment(tid)
                st.write("Your item:", good)
                st.write(target + "'s item:", their)
                if st.button("Send offer"):
                    st.success(propose_trade(grp_clean, uid, tid))
                    st.rerun()

            st.subheader("Outgoing Offers")
            for t in outgoing_trades(uid):
                id_, to, pg, rg, status, _ = t
                with st.container(border=True):
                    st.write("To " + to + ": " + pg + " ↔ " + rg + " (" + status + ")")
                    if st.button("Cancel", key="c" + str(id_)):
                        cancel_outgoing(id_, uid)
                        st.rerun()

        with colB:
            st.subheader("Incoming Offers")
            inc = incoming_trades(uid)
            for t in inc:
                id_, frm, pg, rg, _, _ = t
                with st.container(border=True):
                    st.write("From " + frm + ": they give " + pg + ", you give " + rg)
                    a, b = st.columns(2)
                    if a.button("Accept", key="a" + str(id_)):
                        st.success(accept_trade(id_))
                        st.rerun()
                    if b.button("Decline", key="d" + str(id_)):
                        decline_trade(id_)
                        st.rerun()

            st.subheader("Group Members")
            st.dataframe(
                current_allocation_for_group(grp_clean)[["Name", "Good"]],
                width="stretch",
                hide_index=True
            )

            if st.button("Refresh now"):
                st.rerun()

# ── Social Planner Role ──────────────────────
elif role == "Social Planner":
    if not grp:
        st.info("Enter a group code in the sidebar.")
    else:
        df = current_allocation_for_group(grp)
        if df.empty:
            st.warning("No students yet.")
        else:
            st.title("Social Planner Dashboard")

            with st.expander("Current Allocation", expanded=True):
                st.dataframe(df[["Name", "Good"]], width="stretch", hide_index=True)

            cur_total = total_current_utility(grp)
            max_total, _, assign = optimal_assignment(grp)
            c1, c2 = st.columns(2)
            c1.metric("Current Total Utility", cur_total)
            c2.metric("Max Possible Total Utility", max_total)

            rows = fetchall(
                """SELECT u.name, e.good, p.utility
                   FROM users u
                   JOIN endowments e ON e.user_id=u.id
                   JOIN preferences p ON p.user_id=u.id AND p.good=e.good
                   WHERE u.grp=? ORDER BY u.name""",
                (grp,)
            )
            st.subheader("Per-Student Utilities")
            st.dataframe(pd.DataFrame(rows, columns=["Name", "Good", "Utility"]),
                         width="stretch", hide_index=True)

            # ── Efficiency Dashboard (IR vs ORIGINAL and vs CURRENT) ─────────
            st.divider()
            st.subheader("Efficiency Dashboard (IR-constrained)")

            # Baselines
            users_df, items_now, util_now, label_to_index = _planner_state(grp)
            original_df = original_allocation_for_group(grp)

            # Build baseline label vectors aligned with users_df order
            # (users_df and original_df are both ordered by name)
            base_original = list(original_df["Good"])
            base_current  = list(users_df["Good"])  # current identity

            # 1) Closest Efficient IR-Improvement (vs ORIGINAL)
            ceo_total, ceo_strict, ceo_assign = closest_ir_efficient_improvement_given_baseline(grp, base_original)
            # 2) Utility-Max IR-Improvement (vs ORIGINAL)
            umo_total, umo_strict, umo_assign = util_max_ir_improvement_given_baseline(grp, base_original)
            # 3) Closest Efficient IR-Improvement (vs CURRENT)
            cec_total, cec_strict, cec_assign = closest_ir_efficient_improvement_given_baseline(grp, base_current)
            # 4) Utility-Max IR-Improvement (vs CURRENT)
            umc_total, umc_strict, umc_assign = util_max_ir_improvement_given_baseline(grp, base_current)

            # Present in two rows
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.caption("Closest Efficient IR-Improvement **vs Original**")
                st.metric("People strictly better off", ceo_strict)
                st.metric("Total utility in allocation", ceo_total, delta=ceo_total - cur_total)
                with st.expander("Show assignment"):
                    if ceo_assign:
                        st.dataframe(pd.DataFrame(ceo_assign, columns=["Name", "Assigned Good"]),
                                     width="stretch", hide_index=True)
                    else:
                        st.caption("No IR-improving cycles vs original (already IR-efficient).")

            with r1c2:
                st.caption("Utility-Max IR-Improvement **vs Original**")
                st.metric("People strictly better off", umo_strict)
                st.metric("Total utility in allocation", umo_total, delta=umo_total - cur_total)
                with st.expander("Show assignment"):
                    if umo_assign:
                        st.dataframe(pd.DataFrame(umo_assign, columns=["Name", "Assigned Good"]),
                                     width="stretch", hide_index=True)
                    else:
                        st.caption("No IR-feasible improvement vs original.")

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.caption("Closest Efficient IR-Improvement **vs Current**")
                st.metric("People strictly better off", cec_strict)
                st.metric("Total utility in allocation", cec_total, delta=cec_total - cur_total)
                with st.expander("Show assignment"):
                    if cec_assign:
                        st.dataframe(pd.DataFrame(cec_assign, columns=["Name", "Assigned Good"]),
                                     width="stretch", hide_index=True)
                    else:
                        st.caption("No IR-improving cycles vs current (already IR-efficient).")

            with r2c2:
                st.caption("Utility-Max IR-Improvement **vs Current**")
                st.metric("People strictly better off", umc_strict)
                st.metric("Total utility in allocation", umc_total, delta=umc_total - cur_total)
                with st.expander("Show assignment"):
                    if umc_assign:
                        st.dataframe(pd.DataFrame(umc_assign, columns=["Name", "Assigned Good"]),
                                     width="stretch", hide_index=True)
                    else:
                        st.caption("No IR-feasible improvement vs current.")

            st.caption("All four allocations enforce individual rationality (IR) relative to the stated baseline. "
                       "‘Closest’ executes the smallest strictly-improving cycles until no more IR-improvements remain; "
                       "‘Utility-Max’ solves a Hungarian assignment restricted to IR-feasible edges.")

            # ── Admin Controls ───────────────────────────────────────────────
            st.divider(); st.subheader("Admin Controls")
            with st.expander("Danger Zone — Reset Data", expanded=False):
                cA, cB = st.columns(2)
                with cA:
                    allc = st.checkbox("Confirm reset ALL", key="rall")
                    if st.button("⚠️ Reset ALL Data", disabled=not allc):
                        reset_all_data(); st.success("All data cleared."); st.rerun()
                with cB:
                    one = st.checkbox("Confirm reset group", key="rgrp")
                    if st.button("♻️ Reset This Group", disabled=not one):
                        reset_group(grp); st.success("Group cleared."); st.rerun()

            if st.button("Refresh now"):
                st.rerun()

# ─────────────────────────────────────────────
# Global timed rerun (fires even when no one clicks)
# ─────────────────────────────────────────────
if st.session_state.get("auto_refresh", False):
    time.sleep(int(st.session_state.get("refresh_interval", 5)))
    st.rerun()
