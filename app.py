import streamlit as st
import sqlite3
import random
import pandas as pd
import numpy as np
from itertools import permutations
from contextlib import closing

# --- Admin gate (URL param) ---
params = st.query_params
is_admin_mode = "admin" in params and str(params["admin"][0]).lower() in ["1", "true", "yes"]

# Only show "Social Planner" when admin mode flag present
role_options = ["Student"] + (["Social Planner"] if is_admin_mode else [])


# --------- Config ---------
GOODS = ["Gizmo", "Whatsit", "Thingamabob", "Doohickey", "Widget", "Contraption", "Gadget", "Whatchamacallit"]
DB_PATH = "class_trade.db"
AUTOREFRESH_MS = 5000  # refresh UI every 5s so trades appear promptly

st.set_page_config(page_title="Bilateral Trading Game", page_icon="🔁", layout="wide")

import time

# ---- global defaults (so it works before sidebar renders) ----
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True   # default ON for everyone
if "refresh_interval" not in st.session_state:
    st.session_state["refresh_interval"] = 5  # seconds, change as you like

# ---- clock-driven rerun that survives st.stop() ----
if st.session_state["auto_refresh"]:
    if "next_refresh_time" not in st.session_state:
        st.session_state["next_refresh_time"] = time.time() + st.session_state["refresh_interval"]
    # If it's time, schedule the next tick and rerun immediately
    if time.time() >= st.session_state["next_refresh_time"]:
        st.session_state["next_refresh_time"] = time.time() + st.session_state["refresh_interval"]
        st.rerun()  # use st.rerun() on Streamlit 1.50+


# --------- DB helpers ---------
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def reset_all_data():
    # Wipe all rows from all tables (schema stays intact)
    execute("DELETE FROM trades")
    execute("DELETE FROM preferences")
    execute("DELETE FROM endowments")
    execute("DELETE FROM users")
    # Optional: reclaim space
    execute("VACUUM")

def reset_group(grp):
    # Optional: wipe just one group (keep others intact)
    uids = [u[0] for u in get_group_users(grp)]
    execute("DELETE FROM trades WHERE grp=?", (grp,))
    for uid in uids:
        execute("DELETE FROM preferences WHERE user_id=?", (uid,))
        execute("DELETE FROM endowments WHERE user_id=?", (uid,))
        execute("DELETE FROM users WHERE id=?", (uid,))
    execute("VACUUM")


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
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                user_id INTEGER NOT NULL,
                good TEXT NOT NULL,
                utility INTEGER NOT NULL,
                PRIMARY KEY (user_id, good),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
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
                ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(proposer_id) REFERENCES users(id),
                FOREIGN KEY(recipient_id) REFERENCES users(id)
            )
        """)
        conn.commit()

def fetchone(query, params=()):
    conn = get_conn()
    with closing(conn.cursor()) as cur:
        cur.execute(query, params)
        return cur.fetchone()

def fetchall(query, params=()):
    conn = get_conn()
    with closing(conn.cursor()) as cur:
        cur.execute(query, params)
        return cur.fetchall()

def execute(query, params=()):
    conn = get_conn()
    with closing(conn.cursor()) as cur:
        cur.execute(query, params)
        conn.commit()

# --------- Core model ops ---------
def get_or_create_user(name, grp):
    row = fetchone("SELECT id FROM users WHERE name=? AND grp=?", (name, grp))
    if row:
        return row[0]
    execute("INSERT INTO users (name, grp) VALUES (?,?)", (name, grp))
    user_id = fetchone("SELECT id FROM users WHERE name=? AND grp=?", (name, grp))[0]
    # init endowment and preferences
    start_good = random.choice(GOODS)
    execute("INSERT OR REPLACE INTO endowments (user_id, good) VALUES (?,?)", (user_id, start_good))
    for g in GOODS:
        u = random.randint(1, 10)
        execute("INSERT OR REPLACE INTO preferences (user_id, good, utility) VALUES (?,?,?)", (user_id, g, u))
    return user_id

def get_group_users(grp):
    return fetchall("SELECT id, name FROM users WHERE grp=? ORDER BY name COLLATE NOCASE", (grp,))

def get_user_endowment(user_id):
    row = fetchone("SELECT good FROM endowments WHERE user_id=?", (user_id,))
    return row[0] if row else None

def get_user_prefs_df(user_id):
    rows = fetchall("SELECT good, utility FROM preferences WHERE user_id=? ORDER BY good", (user_id,))
    return pd.DataFrame(rows, columns=["Good", "Utility"])

def current_allocation_for_group(grp):
    rows = fetchall("""
        SELECT u.id, u.name, e.good
        FROM users u
        JOIN endowments e ON e.user_id = u.id
        WHERE u.grp=?
        ORDER BY u.name COLLATE NOCASE
    """, (grp,))
    return pd.DataFrame(rows, columns=["user_id", "Name", "Good"])

def preferences_matrix(grp, users_df, items_list):
    # users_df: columns user_id, Name, Good (current)
    # items_list: list of goods (one entry per item instance)
    mats = []
    for uid in users_df["user_id"]:
        prefs = dict(fetchall("SELECT good, utility FROM preferences WHERE user_id=?", (uid,)))
        mats.append([prefs[g] for g in items_list])
    return np.array(mats, dtype=int)

def propose_trade(grp, proposer_id, recipient_id):
    if proposer_id == recipient_id:
        return "Cannot trade with yourself."
    # capture current goods to lock a specific swap
    pg = get_user_endowment(proposer_id)
    rg = get_user_endowment(recipient_id)
    if pg is None or rg is None:
        return "Endowment missing."
    # check if an identical pending trade already exists
    row = fetchone("""
        SELECT id FROM trades
        WHERE grp=? AND proposer_id=? AND recipient_id=? AND proposer_good=? AND recipient_good=? AND status='pending'
    """, (grp, proposer_id, recipient_id, pg, rg))
    if row:
        return "Trade already pending."
    execute("""
        INSERT INTO trades (grp, proposer_id, recipient_id, proposer_good, recipient_good, status)
        VALUES (?,?,?,?,?, 'pending')
    """, (grp, proposer_id, recipient_id, pg, rg))
    return "Trade proposed."

def incoming_trades(recipient_id):
    return fetchall("""
        SELECT t.id, u.name, t.proposer_good, t.recipient_good, t.status, t.ts
        FROM trades t
        JOIN users u ON u.id = t.proposer_id
        WHERE t.recipient_id=? AND t.status='pending'
        ORDER BY t.ts DESC
    """, (recipient_id,))

def outgoing_trades(proposer_id):
    return fetchall("""
        SELECT t.id, v.name, t.proposer_good, t.recipient_good, t.status, t.ts
        FROM trades t
        JOIN users v ON v.id = t.recipient_id
        WHERE t.proposer_id=? AND t.status='pending'
        ORDER BY t.ts DESC
    """, (proposer_id,))

def update_trade_status(trade_id, new_status):
    execute("UPDATE trades SET status=? WHERE id=?", (new_status, trade_id))

def accept_trade(trade_id):
    # swap goods if still valid
    row = fetchone("""
        SELECT grp, proposer_id, recipient_id, proposer_good, recipient_good, status
        FROM trades WHERE id=?
    """, (trade_id,))
    if not row:
        return "Trade not found."
    grp, pid, rid, pg, rg, status = row
    if status != "pending":
        return "Trade no longer pending."
    # verify current goods match recorded proposal
    cur_pg = get_user_endowment(pid)
    cur_rg = get_user_endowment(rid)
    if cur_pg != pg or cur_rg != rg:
        update_trade_status(trade_id, "declined")
        return "Offer invalid (goods changed)."
    # swap
    execute("UPDATE endowments SET good=? WHERE user_id=?", (rg, pid))
    execute("UPDATE endowments SET good=? WHERE user_id=?", (pg, rid))
    update_trade_status(trade_id, "accepted")
    return "Trade accepted and executed."

def decline_trade(trade_id):
    update_trade_status(trade_id, "declined")

def cancel_outgoing(trade_id, user_id):
    # allow proposer to cancel their pending trade
    row = fetchone("SELECT proposer_id, status FROM trades WHERE id=?", (trade_id,))
    if not row:
        return
    proposer_id, status = row
    if proposer_id == user_id and status == "pending":
        update_trade_status(trade_id, "cancelled")

# --------- Utility calculations ---------
def total_current_utility(grp):
    rows = fetchall("""
        SELECT u.id, e.good, p.utility
        FROM users u
        JOIN endowments e ON e.user_id = u.id
        JOIN preferences p ON p.user_id = u.id AND p.good = e.good
        WHERE u.grp=?
    """, (grp,))
    return sum(r[2] for r in rows)

def optimal_assignment(grp):
    # Builds the assignment that maximizes sum of utilities given the CURRENT multiset of items.
    users_df = current_allocation_for_group(grp)
    n = len(users_df)
    if n == 0:
        return 0, users_df, []

    items = list(users_df["Good"])  # one item per user, as-is
    util = preferences_matrix(grp, users_df, items)

    # Convert to cost matrix for Hungarian (maximize => minimize negative)
    try:
        from scipy.optimize import linear_sum_assignment
        cost = -util
        r, c = linear_sum_assignment(cost)
        max_total = int(util[r, c].sum())
        # c[j] is the item index assigned to user j (after aligning row order)
        assignment = []
        for idx_row, user_row in enumerate(users_df.itertuples(index=False)):
            user_name = user_row.Name
            assigned_item = items[c[idx_row]]
            assignment.append((user_name, assigned_item, int(util[idx_row, c[idx_row]])))
        return max_total, users_df, assignment
    except Exception:
        # Fallback: brute force if small; else greedy approximation.
        if n <= 8:
            best_sum = -1
            best_perm = None
            for perm in permutations(range(n)):
                s = 0
                for i in range(n):
                    s += util[i, perm[i]]
                if s > best_sum:
                    best_sum = s
                    best_perm = perm
            assignment = []
            for i, user_row in enumerate(users_df.itertuples(index=False)):
                assignment.append((user_row.Name, items[best_perm[i]], int(util[i, best_perm[i]])))
            return int(best_sum), users_df, assignment
        # Greedy: each step give the remaining item that boosts utility most
        remaining = set(range(n))
        chosen = {}
        best_sum = 0
        for i in range(n):
            # pick best remaining item for user i
            best_item = max(remaining, key=lambda j: util[i, j])
            best_sum += int(util[i, best_item])
            chosen[i] = best_item
            remaining.remove(best_item)
        assignment = []
        for i, user_row in enumerate(users_df.itertuples(index=False)):
            assignment.append((user_row.Name, items[chosen[i]], int(util[i, chosen[i]])))
        return int(best_sum), users_df, assignment

# --------- UI ---------
init_db()

st.session_state["auto_refresh"] = auto_refresh
st.session_state["refresh_interval"] = refresh_seconds

with st.sidebar:
    st.header("Bilateral Trading Game")
    role = st.radio("Select role", role_options, index=0)
    grp = st.text_input("Group code", placeholder="e.g., A1 or econ101-1").strip()
    auto_refresh = st.toggle("Auto-refresh", value=True, help="Rerun the app on a timer so trades appear for everyone.")
    refresh_seconds = st.number_input("Refresh every (seconds)", min_value=2, max_value=60, value=5, step=1)
    if role == "Student":
        name = st.text_input("Your name", placeholder="First Last").strip()
        start_btn = st.button("Enter / Join Group")

colA, colB = st.columns([1, 1])

if role == "Student":
    if not grp or not name:
        st.info("Enter group and name in the sidebar to begin.")
        st.stop()
    if start_btn or "user_id" not in st.session_state or st.session_state.get("who") != (name, grp):
        try:
            uid = get_or_create_user(name, grp)
            st.session_state["user_id"] = uid
            st.session_state["who"] = (name, grp)
        except Exception as e:
            st.error(f"Error creating or loading your profile: {e}")
            st.stop()
    uid = st.session_state["user_id"]

    with colA:
        st.subheader("Your Endowment")
        my_good = get_user_endowment(uid)
        st.metric(label="You currently hold", value=my_good if my_good else "None")
        st.caption("Trades are swaps of the single item each party currently holds.")

        st.subheader("Your Preferences")
        prefs_df = get_user_prefs_df(uid)
        st.dataframe(prefs_df, use_container_width=True, hide_index=True)

        st.subheader("Propose a Trade")
        users = get_group_users(grp)
        options = [(u[1], u[0]) for u in users if u[0] != uid]
        if len(options) == 0:
            st.warning("No one else in your group yet.")
        else:
            target_name = st.selectbox("Choose a partner", [o[0] for o in options])
            target_id = dict(options)[target_name]
            partner_good = get_user_endowment(target_id)
            st.write("Your item:", my_good)
            st.write(target_name + "'s item:", partner_good)
            if st.button("Send trade offer"):
                msg = propose_trade(grp, uid, target_id)
                if msg.endswith("proposed."):
                    st.success(msg)
                else:
                    st.warning(msg)

        st.subheader("Your Pending Outgoing Offers")
        out = outgoing_trades(uid)
        if len(out) == 0:
            st.caption("No outgoing offers.")
        else:
            for tid, rec_name, pg, rg, status, ts in out:
                with st.container(border=True):
                    st.write("To:", rec_name, "— Offered:", pg, "for", rg)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Status: " + status)
                    with c2:
                        if st.button("Cancel", key="cancel_" + str(tid)):
                            cancel_outgoing(tid, uid)
                            st.experimental_rerun()

    with colB:
        st.subheader("Incoming Offers")
        inc = incoming_trades(uid)
        if len(inc) == 0:
            st.caption("No incoming offers right now.")
        else:
            for tid, prop_name, pg, rg, status, ts in inc:
                with st.container(border=True):
                    st.write("From:", prop_name)
                    st.write("They give:", pg, "— You give:", rg)
                    a, b = st.columns(2)
                    with a:
                        if st.button("Accept", key="accept_" + str(tid)):
                            res = accept_trade(tid)
                            if res.startswith("Trade accepted"):
                                st.success(res)
                            else:
                                st.warning(res)
                            st.experimental_rerun()
                    with b:
                        if st.button("Decline", key="decline_" + str(tid)):
                            decline_trade(tid)
                            st.experimental_rerun()

        st.subheader("People in Your Group")
        roster = current_allocation_for_group(grp)[["Name", "Good"]]
        st.dataframe(roster, use_container_width=True, hide_index=True)

elif role == "Social Planner":
    if not grp:
        st.info("Enter a group code in the sidebar.")
        st.stop()

    st.title("Social Planner Dashboard")

    alloc_df = current_allocation_for_group(grp)
    if alloc_df.empty:
        st.warning("No students in this group yet.")
        st.stop()

    with st.expander("Current Allocation"):
        st.dataframe(alloc_df[["Name", "Good"]], use_container_width=True, hide_index=True)

    # Current utility
    cur_total = total_current_utility(grp)

    # Optimal (max) utility and suggested assignment
    max_total, users_df, assignment = optimal_assignment(grp)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Current Total Utility", cur_total)
    with c2:
        st.metric("Maximum Possible Total Utility", max_total)

    # Show per-student current utility
    rows = fetchall("""
        SELECT u.name, e.good, p.utility
        FROM users u
        JOIN endowments e ON e.user_id = u.id
        JOIN preferences p ON p.user_id = u.id AND p.good = e.good
        WHERE u.grp=?
        ORDER BY u.name COLLATE NOCASE
    """, (grp,))
    cur_util_df = pd.DataFrame(rows, columns=["Name", "Good", "Utility"])
    st.subheader("Per-Student Utilities (Current)")
    st.dataframe(cur_util_df, use_container_width=True, hide_index=True)

    st.subheader("One Optimal Assignment (given current items)")
    if len(assignment) == 0:
        st.caption("Insufficient data.")
    else:
        opt_df = pd.DataFrame(assignment, columns=["Name", "Assigned Good", "Utility"])
        st.dataframe(opt_df.sort_values("Name"), use_container_width=True, hide_index=True)

    st.caption("Notes: The maximum is computed over reassignments of the current item pool only. If SciPy is installed, the Hungarian algorithm is used; otherwise a fallback is used (exact up to 8 students, then greedy).")

    st.divider()
    st.subheader("Admin Controls")
    
    with st.expander("Danger Zone — Reset Data", expanded=False):
        st.caption("Use with care. This cannot be undone.")
    
        c1, c2 = st.columns(2)
    
        with c1:
            confirm_all = st.checkbox("I understand: Reset **ALL** groups", key="confirm_all_reset")
            if st.button("⚠️ Reset ALL Data", type="primary", disabled=not confirm_all):
                reset_all_data()
                st.success("All data cleared. Fresh start!")
                st.experimental_rerun()
    
        with c2:
            st.caption("Optionally reset just this group.")
            confirm_grp = st.checkbox("I understand: Reset **this** group only", key="confirm_grp_reset")
            if st.button("♻️ Reset This Group (" + grp + ")", disabled=not confirm_grp):
                reset_group(grp)
                st.success("Group " + grp + " cleared.")
                st.experimental_rerun()


# ----- timed rerun -----
if st.session_state.get("auto_refresh", False):
    import time
    time.sleep(int(st.session_state.get("refresh_interval", 5)))
    st.experimental_rerun()
