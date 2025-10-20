# app.py  (no password gate)  — with Efficiency Dashboard
import streamlit as st
import sqlite3
import random
import pandas as pd
import numpy as np
from itertools import permutations
from contextlib import closing
from collections import defaultdict
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
    execute("DELETE FROM users")
    execute("VACUUM")

def reset_group(grp):
    uids = [u[0] for u in get_group_users(grp)]
    execute("DELETE FROM trades WHERE grp=?", (grp,))
    for uid in uids:
        execute("DELETE FROM preferences WHERE user_id=?", (uid,))
        execute("DELETE FROM endowments WHERE user_id=?", (uid,))
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
    if available:
        return random.choice(available)
    return random.choice(GOODS)

def get_or_create_user(name, grp):
    row = fetchone("SELECT id FROM users WHERE name=? AND grp=?", (name, grp))
    if row:
        return row[0]
    execute("INSERT INTO users (name, grp) VALUES (?,?)", (name, grp))
    uid = fetchone("SELECT id FROM users WHERE name=? AND grp=?", (name, grp))[0]
    start_good = pick_start_good_unique(grp)
    execute("INSERT INTO endowments VALUES (?,?)", (uid, start_good))
    for g in GOODS:
        u = random.randint(1, 10)
        execute("INSERT INTO preferences VALUES (?,?,?)", (uid, g, u))
    return uid

def get_user_endowment(uid):
    r = fetchone("SELECT good FROM endowments WHERE user_id=?", (uid,))
    return r[0] if r else None

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
        # Greedy fallback (works for any n)
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
# Pareto tools for Planner Dashboard
# ─────────────────────────────────────────────
def _planner_state(grp):
    """Return (users_df, items, util, name_index, good_index, current_indices)"""
    users_df = current_allocation_for_group(grp)  # user_id, Name, Good
    items = list(users_df["Good"])                # current goods by holder index
    util = preferences_matrix(grp, users_df, items)
    name_index = {i: users_df.iloc[i]["Name"] for i in range(len(users_df))}
    good_index = {j: items[j] for j in range(len(items))}
    # current assignment is identity: user i holds item i
    current_indices = list(range(len(users_df)))
    return users_df, items, util, name_index, good_index, current_indices

def is_pareto_efficient_current(grp):
    """Check if current allocation is Pareto efficient (w.r.t. individual rationality relative to current)."""
    users_df, items, util, *_ = _planner_state(grp)
    n = len(users_df)
    if n <= 1:
        return True
    # edge i->j if util[i,j] >= util[i,i]; strict if >
    better_or_equal = (util >= np.diag(util))
    strict = (util > np.diag(util))
    # Build mapping by pointing each i to an acceptable j with highest utility (tie-break own item)
    # Then see if any cycle contains at least one strict edge.
    targets = []
    for i in range(n):
        acceptable_idxs = np.where(better_or_equal[i])[0]
        # prefer keeping item if tied
        best_val = -1
        best_j = i
        for j in acceptable_idxs:
            v = util[i, j]
            if v > best_val or (v == best_val and j == i):
                best_val = v; best_j = j
        targets.append(best_j)
    # Detect cycles and whether any has at least one strict improvement
    visited = [0]*n
    for i in range(n):
        if visited[i]: continue
        path_index = {}
        cur = i
        while True:
            if visited[cur]: break
            path_index[cur] = len(path_index)
            visited[cur] = 1
            cur = targets[cur]
            if cur in path_index:
                # found cycle: nodes with index >= path_index[cur]
                cycle_nodes = [k for k, idx in path_index.items() if idx >= path_index[cur]]
                # strict if any i in cycle has util[i, targets[i]] > util[i,i]
                if any(strict[k, targets[k]] for k in cycle_nodes if targets[k] != k):
                    return False
                break
    return True

def closest_efficient_pareto_improvement(grp):
    """
    Greedy TTC-style: in each round, form 'best acceptable' pointers and
    execute the single cycle with the FEWEST strictly-better participants.
    Repeat until no strictly-improving cycles remain.
    Return: (total_utility, strictly_better_count, final_assignment list of (Name, Good))
    All comparisons are vs the ORIGINAL current allocation.
    """
    users_df, items, util, name_index, good_index, _ = _planner_state(grp)
    n = len(users_df)
    if n == 0:
        return 0, 0, []
    base_util_self = np.diag(util).copy()

    # Working copies
    cur_items = items[:]  # items[j] is the label of item at index j
    # Track which item index each user i currently points to when choosing best acceptable
    def build_targets():
        better_or_equal = (util >= np.diag(util))
        targets = []
        for i in range(n):
            acceptable = np.where(better_or_equal[i])[0]
            # choose highest utility acceptable (tie-break: own item to reduce churn)
            best_v = -1
            best_j = i
            for j in acceptable:
                v = util[i, j]
                if v > best_v or (v == best_v and j == i):
                    best_v = v; best_j = j
            targets.append(best_j)
        return targets

    # To interpret strictness vs ORIGINAL current allocation:
    def strictly_better_in_edge(i, j):
        return util[i, j] > base_util_self[i]

    strictly_better_seen = set()
    iteration_guard = 0
    while True:
        iteration_guard += 1
        if iteration_guard > n * 5:
            break  # safety

        targets = build_targets()

        # Find all cycles in the functional graph i -> targets[i]
        visited = [0]*n
        cycles = []
        for s in range(n):
            if visited[s]: continue
            path_index = {}
            cur = s
            while not visited[cur]:
                path_index[cur] = len(path_index)
                visited[cur] = 1
                cur = targets[cur]
                if cur in path_index:
                    # cycle nodes:
                    idx0 = path_index[cur]
                    ordered = sorted(path_index.items(), key=lambda kv: kv[1])
                    cyc = [node for node, idx in ordered if idx >= idx0]
                    cycles.append(cyc)
                    break

        # Filter to strictly-improving cycles (at least one strict edge vs ORIGINAL base)
        improving_cycles = []
        for cyc in cycles:
            has_strict = False
            for i in cyc:
                j = targets[i]
                if i != j and strictly_better_in_edge(i, j):
                    has_strict = True; break
            if has_strict:
                improving_cycles.append(cyc)

        if not improving_cycles:
            # no more improvement cycles → Pareto-efficient
            break

        # Pick the cycle with the FEWEST strictly-better participants (greedy "closest" step)
        def cycle_strict_count(cyc):
            return sum(1 for i in cyc if targets[i] != i and strictly_better_in_edge(i, targets[i]))
        improving_cycles.sort(key=lambda cyc: (cycle_strict_count(cyc), len(cyc)))
        chosen = improving_cycles[0]

        # Execute the chosen cycle: move items along the pointers on this cycle
        # For cycle i1->i2->...->ik->i1, each node gets the item of its target.
        new_items = cur_items[:]
        for idx in range(len(chosen)):
            i = chosen[idx]
            j = targets[i]
            # assign i the item currently at index j
            new_items[i] = cur_items[j]
        cur_items = new_items

        # Update util "self" diagonal to reflect who strictly improved vs ORIGINAL
        for i in chosen:
            j = targets[i]
            if strictly_better_in_edge(i, j):
                strictly_better_seen.add(i)

        # Recompute util columns order to reflect new item positions
        # Note: util[i, j] indexes by item index j; when items permute, util w.r.t. the current j stays valid.
        # We don't need to reorder util; j is always an item index (position), not a label.

        # Continue loop until no improving cycles remain

    # Compute final totals for this allocation
    # Build (Name, Good) for final assignment
    final_assignment = [(users_df.iloc[i]["Name"], cur_items[i]) for i in range(n)]
    # Total utility in final allocation:
    # Need index of each item label in the ORIGINAL items list to find column index
    label_to_index = {label: idx for idx, label in enumerate(items)}
    total_final = 0
    for i in range(n):
        j = label_to_index[cur_items[i]]
        total_final += int(util[i, j])

    return int(total_final), int(len(strictly_better_seen)), final_assignment

def utility_maximizing_pareto_improvement(grp):
    """
    Maximize total utility subject to individual rationality vs current:
    util[i, j] >= util[i, i] for all i who receive j.
    Solve via Hungarian on a restricted matrix (disallow IR-violating edges by setting -inf).
    Return: (total_utility, strictly_better_count, assignment list of (Name, Good))
    """
    users_df, items, util, name_index, good_index, _ = _planner_state(grp)
    n = len(users_df)
    if n == 0:
        return 0, 0, []

    base_self = np.diag(util)
    allowed = util >= np.diag(util)  # IR constraints
    # Build restricted matrix: very negative cost for disallowed edges
    # We'll maximize util -> minimize negative util
    REJECT = -10**9
    restricted_util = util.copy().astype(float)
    restricted_util[~allowed] = -1e6  # big negative so they won't be chosen
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-restricted_util)  # maximize
        total = int(restricted_util[r, c].sum())
        # If any disallowed chosen (would be -1e6), then no IR-feasible improving assignment → fall back
        if total <= int(base_self.sum()):
            # No strictly improving IR-feasible solution, return current
            assign = [(users_df.iloc[i]["Name"], items[i]) for i in range(n)]
            strictly = 0
            return int(base_self.sum()), strictly, assign
        assign = []
        strictly = 0
        for i in range(n):
            good_label = items[c[i]]
            assign.append((users_df.iloc[i]["Name"], good_label))
            if util[i, c[i]] > base_self[i]:
                strictly += 1
        return int(sum(util[i, c[i]] for i in range(n))), int(strictly), assign
    except Exception:
        # Fallback greedy under IR
        remaining = set(range(n))
        chosen = {}
        total = 0
        strictly = 0
        for i in range(n):
            # among allowed items pick best remaining
            choices = [j for j in range(n) if j in remaining and util[i, j] >= base_self[i]]
            if not choices:
                # must keep own if available
                if i in remaining:
                    j = i
                else:
                    j = next(iter(remaining))
            else:
                j = max(choices, key=lambda jj: util[i, jj])
            chosen[i] = j; remaining.discard(j)
            total += int(util[i, j]); strictly += int(util[i, j] > base_self[i])
        assign = [(users_df.iloc[i]["Name"], items[chosen[i]]) for i in range(n)]
        return int(total), int(strictly), assign

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

    # --- Auto-refresh controls ---
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

    # First-time join path (NO st.stop)
    if "user_id" not in st.session_state:
        if not grp_clean or not name_clean:
            st.info("Enter group and name in the sidebar to begin.")
        elif start_btn or True:
            uid = get_or_create_user(name_clean, grp_clean)
            st.session_state["user_id"] = uid
            st.session_state["who"] = (name_clean, grp_clean)
            st.rerun()

    # Render student UI only if logged in
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
                    st.write(f"To {to}: {pg} ↔ {rg} ({status})")
                    if st.button("Cancel", key="c" + str(id_)):
                        cancel_outgoing(id_, uid)
                        st.rerun()

        with colB:
            st.subheader("Incoming Offers")
            inc = incoming_trades(uid)
            for t in inc:
                id_, frm, pg, rg, _, _ = t
                with st.container(border=True):
                    st.write(f"From {frm}: they give {pg}, you give {rg}")
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

            # ── Efficiency Dashboard ─────────────────────────────────────────
            st.divider()
            st.subheader("Efficiency Dashboard")

            # Pareto efficient now?
            pareto_now = is_pareto_efficient_current(grp)
            st.markdown("**Pareto Efficient (current allocation)?** " + ("Yes" if pareto_now else "No"))

            # Closest Pareto-efficient Pareto-improving allocation (greedy TTC)
            closest_total, closest_strict, closest_assign = closest_efficient_pareto_improvement(grp)
            # Utility-maximizing Pareto-improving allocation (restricted Hungarian)
            utilpi_total, utilpi_strict, utilpi_assign = utility_maximizing_pareto_improvement(grp)

            # Summaries
            c3, c4 = st.columns(2)
            with c3:
                st.caption("Closest Efficient Pareto-Improving Allocation (greedy TTC)")
                st.metric("People strictly better off vs current", closest_strict)
                st.metric("Total utility in that allocation", closest_total, delta=closest_total - cur_total)
                with st.expander("Show assignment"):
                    if closest_assign:
                        st.dataframe(pd.DataFrame(closest_assign, columns=["Name", "Assigned Good"]),
                                     width="stretch", hide_index=True)
                    else:
                        st.caption("No improvement available (already Pareto efficient).")

            with c4:
                st.caption("Utility-Maximizing Pareto-Improving Allocation (IR-constrained Hungarian)")
                st.metric("People strictly better off vs current", utilpi_strict)
                st.metric("Total utility in that allocation", utilpi_total, delta=utilpi_total - cur_total)
                with st.expander("Show assignment"):
                    if utilpi_assign:
                        st.dataframe(pd.DataFrame(utilpi_assign, columns=["Name", "Assigned Good"]),
                                     width="stretch", hide_index=True)
                    else:
                        st.caption("No improvement available (already Pareto efficient).")

            st.caption("Notes: The 'closest efficient' result uses a greedy trading-cycles heuristic; "
                       "the IR-constrained Hungarian maximizes total utility subject to no one being worse off.")

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
