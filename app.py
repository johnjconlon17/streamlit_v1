# app.py — Multi-unit market with bundle trades (1x1, 2x1, 1x2, 2x2)
import streamlit as st
import sqlite3
import random
import pandas as pd
import numpy as np
from contextlib import closing
import json
import time
import math

# ─────────────────────────────────────────────
# Page config FIRST
# ─────────────────────────────────────────────
st.set_page_config(page_title="Bilateral Trading Game (Multi-Unit)", page_icon="🔁", layout="wide")

# ─────────────────────────────────────────────
# Role gate via URL param (no passwords)
# ─────────────────────────────────────────────
params = st.query_params
is_admin_mode = "admin" in params and str(params["admin"]).lower() in ["1", "true", "yes"]
role_options = ["Student"] + (["Social Planner"] if is_admin_mode else [])

# ─────────────────────────────────────────────
# Master goods list (we'll sample T per group at start)
# ─────────────────────────────────────────────
MASTER_GOODS = [
    "Gizmo", "Whatsit", "Thingamabob", "Doohickey", "Widget",
    "Contraption", "Gadget", "Whatchamacallit", "Doodad", "Thingy",
    "Gubbins", "Apparatus", "Mechanism", "Rigamarole", "Oddment",
    "Thingummy", "Whirligig", "Dinglehopper", "Curio", "Bric-a-brac"
]

DB_PATH = "class_trade_multi.db"

# ─────────────────────────────────────────────
# Light auto-refresh defaults
# ─────────────────────────────────────────────
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "refresh_interval" not in st.session_state:
    st.session_state["refresh_interval"] = 6

# ─────────────────────────────────────────────
# DB helpers / schema
# ─────────────────────────────────────────────
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=6.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=4000;")
    return conn

def _safe_commit(conn, retries=2):
    for i in range(retries + 1):
        try:
            conn.commit()
            return
        except Exception:
            if i == retries:
                raise
            time.sleep(0.05 * (i + 1))

def init_db():
    conn = get_conn()
    with closing(conn.cursor()) as cur:
        # Users
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                grp  TEXT NOT NULL,
                UNIQUE(name, grp)
            )
        """)
        # Group config / lifecycle
        cur.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                grp TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'waiting',  -- 'waiting' | 'started'
                k INTEGER DEFAULT 1,                     -- units per student
                t INTEGER DEFAULT 5,                     -- # distinct types in this group
                types_json TEXT DEFAULT '[]',            -- JSON array of active types
                started_ts DATETIME
            )
        """)
        # Preferences: utility per type (1..10)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                utility INTEGER NOT NULL,
                PRIMARY KEY (user_id, type),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        # Items: each row is one copy
        cur.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grp TEXT NOT NULL,
                type TEXT NOT NULL,
                owner_id INTEGER NOT NULL,
                FOREIGN KEY(owner_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        # Multi-unit trade proposals (bundle on each side)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grp TEXT NOT NULL,
                proposer_id INTEGER NOT NULL,
                recipient_id INTEGER NOT NULL,
                offer_json TEXT NOT NULL,    -- {"type": count, ...} proposer → recipient
                request_json TEXT NOT NULL,  -- {"type": count, ...} recipient → proposer
                status TEXT NOT NULL,        -- 'pending' | 'accepted' | 'declined' | 'cancelled'
                ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(proposer_id) REFERENCES users(id),
                FOREIGN KEY(recipient_id) REFERENCES users(id)
            )
        """)
        _safe_commit(conn)

def fetchone(q, p=()):
    c = get_conn()
    with closing(c.cursor()) as cur:
        cur.execute(q, p)
        return cur.fetchone()

def fetchall(q, p=()):
    c = get_conn()
    with closing(c.cursor()) as cur:
        cur.execute(q, p)
        return cur.fetchall()

def execute(q, p=()):
    c = get_conn()
    with closing(c.cursor()) as cur:
        cur.execute(q, p)
        c.commit()

# ─────────────────────────────────────────────
# Group lifecycle helpers
# ─────────────────────────────────────────────
def ensure_group_row(grp):
    if not grp:
        return
    row = fetchone("SELECT grp FROM groups WHERE grp=?", (grp,))
    if not row:
        execute("INSERT INTO groups (grp, status, k, t, types_json) VALUES (?, 'waiting', 1, 5, '[]')", (grp,))

def get_group_state(grp):
    row = fetchone("SELECT status, k, t, types_json FROM groups WHERE grp=?", (grp,))
    if not row:
        return ("waiting", 1, 5, [])
    status, k, t, tj = row
    types = json.loads(tj or "[]")
    return (status, int(k), int(t), types)

def set_group_config(grp, status=None, k=None, t=None, types=None):
    ensure_group_row(grp)
    cur_status, cur_k, cur_t, cur_types = get_group_state(grp)
    status = status if status is not None else cur_status
    k = k if k is not None else cur_k
    t = t if t is not None else cur_t
    types_json = json.dumps(types if types is not None else cur_types)
    execute("UPDATE groups SET status=?, k=?, t=?, types_json=? WHERE grp=?",
            (status, int(k), int(t), types_json, grp))

def reset_group(grp):
    execute("DELETE FROM trades WHERE grp=?", (grp,))
    execute("DELETE FROM items  WHERE grp=?", (grp,))
    # Keep users and preferences
    set_group_config(grp, status="waiting", types=[], k=1, t=5)

def get_group_users(grp):
    return fetchall("SELECT id, name FROM users WHERE grp=? ORDER BY name COLLATE NOCASE", (grp,))

# ─────────────────────────────────────────────
# Users / preferences
# ─────────────────────────────────────────────
def get_or_create_user(name, grp):
    ensure_group_row(grp)
    status, _, _, _ = get_group_state(grp)
    row = fetchone("SELECT id FROM users WHERE name=? AND grp=?", (name, grp))
    if row:
        return row[0]
    if status == "started":
        raise RuntimeError("Market already started for this group; new students cannot join until reset.")
    execute("INSERT INTO users (name, grp) VALUES (?,?)", (name, grp))
    uid = fetchone("SELECT id FROM users WHERE name=? AND grp=?", (name, grp))[0]
    for t in MASTER_GOODS:
        u = random.randint(1, 10)
        execute("INSERT OR REPLACE INTO preferences (user_id, type, utility) VALUES (?,?,?)", (uid, t, u))
    return uid

def prefs_df_for_user(uid, active_types):
    rows = fetchall("SELECT type, utility FROM preferences WHERE user_id=?", (uid,))
    df = pd.DataFrame(rows, columns=["Type", "Utility"])
    if active_types:
        df = df[df["Type"].isin(active_types)]
    return df.sort_values("Type")

# ─────────────────────────────────────────────
# Items / inventories
# ─────────────────────────────────────────────
def allocate_initial_items(grp, k, types):
    execute("DELETE FROM items WHERE grp=?", (grp,))
    users = get_group_users(grp)
    if not users or not types:
        return
    for uid, _ in users:
        for _ in range(int(k)):
            typ = random.choice(types)
            execute("INSERT INTO items (grp, type, owner_id) VALUES (?,?,?)", (grp, typ, uid))

def inventory_counts(grp, uid):
    rows = fetchall("SELECT type, COUNT(*) FROM items WHERE grp=? AND owner_id=? GROUP BY type", (grp, uid))
    return {t: int(c) for (t, c) in rows}

def inventory_items_by_type(grp, uid):
    rows = fetchall("SELECT id, type FROM items WHERE grp=? AND owner_id=?", (grp, uid))
    by = {}
    for iid, t in rows:
        by.setdefault(t, []).append(int(iid))
    return by  # {type: [item_ids...]}

def group_active_types(grp):
    _, _, _, types = get_group_state(grp)
    return types

# ─────────────────────────────────────────────
# Utility with zero marginal after first copy
# ─────────────────────────────────────────────
def bundle_utility(uid, types_set):
    rows = fetchall("SELECT type, utility FROM preferences WHERE user_id=?", (uid,))
    u_map = {t: int(v) for (t, v) in rows}
    return int(sum(u_map.get(t, 0) for t in types_set))

def user_types_set(grp, uid):
    rows = fetchall("SELECT DISTINCT type FROM items WHERE grp=? AND owner_id=?", (grp, uid))
    return {t for (t,) in rows}

def current_total_utility(grp):
    users = get_group_users(grp)
    return sum(bundle_utility(uid, user_types_set(grp, uid)) for uid, _ in users)

# ─────────────────────────────────────────────
# Trade helpers (bundles up to 2 types per side)
# ─────────────────────────────────────────────
def valid_bundle(counts):
    if not isinstance(counts, dict) or not counts:
        return False
    for v in counts.values():
        if not isinstance(v, int) or v < 0:
            return False
    return sum(counts.values()) > 0

def has_bundle(grp, uid, counts):
    if not counts:
        return False
    inv = inventory_counts(grp, uid)
    for t, c in counts.items():
        if c > 0 and inv.get(t, 0) < c:
            return False
    return True

def normalize_bundle(d):
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        if k and isinstance(v, (int, float)):
            n = int(v)
            if n > 0:
                out[k] = n
    return out

def propose_trade(grp, proposer_id, recipient_id, offer_counts, request_counts):
    offer = normalize_bundle(offer_counts)
    request = normalize_bundle(request_counts)
    if not valid_bundle(offer) or not valid_bundle(request):
        return "Invalid bundle."
    if proposer_id == recipient_id:
        return "Cannot trade with yourself."
    if not has_bundle(grp, proposer_id, offer):
        return "You no longer have the items you’re offering."
    if not has_bundle(grp, recipient_id, request):
        return "Partner no longer has the items you’re requesting."
    execute("""INSERT INTO trades (grp, proposer_id, recipient_id, offer_json, request_json, status)
               VALUES (?,?,?,?,?, 'pending')""",
            (grp, proposer_id, recipient_id, json.dumps(offer), json.dumps(request)))
    return "Trade proposed."

def list_incoming(grp, uid):
    return fetchall("""SELECT id, proposer_id, offer_json, request_json, ts
                       FROM trades WHERE grp=? AND recipient_id=? AND status='pending'
                       ORDER BY ts DESC""", (grp, uid))

def list_outgoing(grp, uid):
    return fetchall("""SELECT id, recipient_id, offer_json, request_json, ts
                       FROM trades WHERE grp=? AND proposer_id=? AND status='pending'
                       ORDER BY ts DESC""", (grp, uid))

def cancel_trade(grp, uid, trade_id):
    row = fetchone("SELECT proposer_id, status FROM trades WHERE id=? AND grp=?", (trade_id, grp))
    if row and row[0] == uid and row[1] == "pending":
        execute("UPDATE trades SET status='cancelled' WHERE id=?", (trade_id,))

def decline_trade(trade_id):
    execute("UPDATE trades SET status='declined' WHERE id=?", (trade_id,))

def accept_trade(grp, trade_id):
    row = fetchone("""SELECT proposer_id, recipient_id, offer_json, request_json, status
                      FROM trades WHERE id=? AND grp=?""", (trade_id, grp))
    if not row:
        return "Trade not found."
    proposer_id, recipient_id, offer_js, request_js, status = row
    if status != "pending":
        return "Trade no longer pending."
    offer = json.loads(offer_js); request = json.loads(request_js)

    if not has_bundle(grp, proposer_id, offer):
        decline_trade(trade_id)
        return "Offer invalid: proposer no longer has those items."
    if not has_bundle(grp, recipient_id, request):
        decline_trade(trade_id)
        return "Offer invalid: recipient no longer has requested items."

    # Move concrete item copies
    prop_items = inventory_items_by_type(grp, proposer_id)
    recp_items = inventory_items_by_type(grp, recipient_id)

    # proposer → recipient
    for t, c in offer.items():
        ids = prop_items.get(t, [])[:c]
        for iid in ids:
            execute("UPDATE items SET owner_id=? WHERE id=?", (recipient_id, iid))

    # recipient → proposer
    for t, c in request.items():
        ids = recp_items.get(t, [])[:c]
        for iid in ids:
            execute("UPDATE items SET owner_id=? WHERE id=?", (proposer_id, iid))

    execute("UPDATE trades SET status='accepted' WHERE id=?", (trade_id,))
    return "Trade executed."

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
init_db()

with st.sidebar:
    st.header("Bilateral Trading Game (Multi-Unit)")
    role = st.radio("Select role", role_options, index=0)

    prev_name, prev_grp = (st.session_state.get("who") or ("", ""))
    grp = st.text_input("Group code", value=prev_grp, placeholder="e.g., A1 or econ101-1")
    if role == "Student":
        name = st.text_input("Your name", value=prev_name, placeholder="First Last")
        start_btn = st.button("Enter / Join Group")

    auto = st.toggle("Auto-refresh", value=st.session_state["auto_refresh"])
    secs = st.number_input("Refresh every (s)", 2, 60, st.session_state["refresh_interval"], step=1)
    if auto != st.session_state["auto_refresh"] or secs != st.session_state["refresh_interval"]:
        st.session_state["auto_refresh"] = auto
        st.session_state["refresh_interval"] = int(secs)
        st.session_state["next_refresh_time"] = time.time() + st.session_state["refresh_interval"]

colA, colB = st.columns(2)

# ─────────────────────────────────────────────
# Student role
# ─────────────────────────────────────────────
if role == "Student":
    grp_clean = (grp or "").strip()
    name_clean = (name or "").strip() if "name" in locals() else ""

    if "user_id" not in st.session_state:
        if not grp_clean or not name_clean:
            st.info("Enter group and name in the sidebar to begin.")
        elif start_btn or True:
            try:
                uid = get_or_create_user(name_clean, grp_clean)
            except Exception as e:
                st.error(str(e))
                st.stop()
            st.session_state["user_id"] = uid
            st.session_state["who"] = (name_clean, grp_clean)
            st.rerun()

    if "user_id" in st.session_state:
        uid = st.session_state["user_id"]
        name_clean, grp_clean = st.session_state["who"]
        status, k_cfg, t_cfg, active_types = get_group_state(grp_clean)

        with colA:
            st.subheader("Your Inventory")
            inv = inventory_counts(grp_clean, uid)
            table_rows = [(t, inv.get(t, 0)) for t in (active_types or [])]
            st.dataframe(pd.DataFrame(sorted(table_rows), columns=["Type", "Count"]),
                         width="stretch", hide_index=True)

            st.subheader("Your Preferences (active types)")
            st.dataframe(prefs_df_for_user(uid, active_types), width="stretch", hide_index=True)

        with colB:
            st.subheader("Group Status")
            st.write("Status:", status.upper())
            roster = pd.DataFrame(get_group_users(grp_clean), columns=["user_id", "Name"])
            st.dataframe(roster[["Name"]], width="stretch", hide_index=True)

            if status == "waiting":
                st.info("Waiting period — trades disabled until the planner starts the market.")
            else:
                st.subheader("Propose a Trade (1-for-1, 2-for-1, 1-for-2, 2-for-2)")
                others = [(u[1], u[0]) for u in get_group_users(grp_clean) if u[0] != uid]
                if not others:
                    st.warning("No one else in your group yet.")
                else:
                    partner_names = [o[0] for o in others]
                    partner = st.selectbox("Choose partner", partner_names, index=None, placeholder="Select a partner")
                    partner_id = None if partner is None else dict(others)[partner]

                    inv_self = inventory_counts(grp_clean, uid)
                    inv_partner = inventory_counts(grp_clean, partner_id) if partner_id else {}

                    # Offer side (you give)
                    st.markdown("**You offer**")
                    offer_type1 = st.selectbox("Offer type 1", ["(none)"] + (active_types or []), index=0, key="off1")
                    offer_cnt1  = st.number_input("Count", 0, 2, 0, key="off1c")
                    offer_type2 = st.selectbox("Offer type 2", ["(none)"] + (active_types or []), index=0, key="off2")
                    offer_cnt2  = st.number_input("Count ", 0, 2, 0, key="off2c")

                    # Request side (you receive)
                    st.markdown("**You request**")
                    req_type1 = st.selectbox("Request type 1", ["(none)"] + (active_types or []), index=0, key="req1")
                    req_cnt1  = st.number_input("Count  ", 0, 2, 0, key="req1c")
                    req_type2 = st.selectbox("Request type 2", ["(none)"] + (active_types or []), index=0, key="req2")
                    req_cnt2  = st.number_input("Count   ", 0, 2, 0, key="req2c")

                    offer = {}
                    if offer_type1 != "(none)": offer[offer_type1] = offer_cnt1
                    if offer_type2 != "(none)": offer[offer_type2] = offer.get(offer_type2, 0) + offer_cnt2
                    request = {}
                    if req_type1 != "(none)": request[req_type1] = req_cnt1
                    if req_type2 != "(none)": request[req_type2] = request.get(req_type2, 0) + req_cnt2

                    # Show quick availability hints (only if a partner is chosen)
                    if partner_id is not None:
                        if offer:
                            missing = [t for t, c in offer.items() if c > inv_self.get(t, 0)]
                            if missing:
                                st.warning("You don't have enough of: " + ", ".join(missing))
                        if request:
                            missing = [t for t, c in request.items() if c > inv_partner.get(t, 0)]
                            if missing:
                                st.warning(partner + " doesn't have enough of: " + ", ".join(missing))

                    send_disabled = partner_id is None or not offer or not request
                    if st.button("Send trade offer", disabled=send_disabled):
                        msg = propose_trade(grp_clean, uid, partner_id, offer, request)
                        if msg == "Trade proposed.":
                            st.success(msg)
                            st.rerun()
                        else:
                            st.warning(msg)

                st.subheader("Incoming Offers")
                inc = list_incoming(grp_clean, uid)
                if not inc:
                    st.caption("No incoming offers.")
                else:
                    for tid, pid, offer_js, req_js, ts in inc:
                        prop_name = fetchone("SELECT name FROM users WHERE id=?", (pid,))[0]
                        offer_d = json.loads(offer_js); req_d = json.loads(req_js)
                        with st.container(border=True):
                            st.write("From:", prop_name)
                            st.write("They give you:", req_d)
                            st.write("You give them:", offer_d)
                            a, b = st.columns(2)
                            if a.button("Accept", key="acc" + str(tid)):
                                res = accept_trade(grp_clean, tid)
                                if res.startswith("Trade executed"):
                                    st.success(res)
                                else:
                                    st.warning(res)
                                st.rerun()
                            if b.button("Decline", key="dec" + str(tid)):
                                decline_trade(tid); st.rerun()

                st.subheader("Your Pending Outgoing Offers")
                out = list_outgoing(grp_clean, uid)
                if not out:
                    st.caption("No outgoing offers.")
                else:
                    for tid, rid, offer_js, req_js, ts in out:
                        rec_name = fetchone("SELECT name FROM users WHERE id=?", (rid,))[0]
                        with st.container(border=True):
                            st.write("To:", rec_name)
                            st.write("You offer:", json.loads(offer_js))
                            st.write("You request:", json.loads(req_js))
                            if st.button("Cancel", key="can" + str(tid)):
                                cancel_trade(grp_clean, uid, tid); st.rerun()

# ─────────────────────────────────────────────
# Planner role
# ─────────────────────────────────────────────
elif role == "Social Planner":
    if not grp:
        st.info("Enter a group code in the sidebar.")
    else:
        ensure_group_row(grp)
        status, k_cfg, t_cfg, active_types = get_group_state(grp)

        st.title("Social Planner")
        st.write("Group:", grp, "· Status:", status.upper())

        with st.expander("Roster"):
            roster = pd.DataFrame(get_group_users(grp), columns=["user_id", "Name"])
            st.dataframe(roster[["Name"]], width="stretch", hide_index=True)

        st.subheader("Configuration")
        k_new = st.number_input("Units per student (K)", 1, 10, k_cfg)
        max_t = min(len(MASTER_GOODS), 15)
        t_new = st.number_input("Number of distinct types (T)", 2, max_t, min(t_cfg, max_t))
        st.caption("T types are sampled from a master list; duplicates in inventories are allowed.")

        if st.button("Save config (does not start market)"):
            set_group_config(grp, k=int(k_new), t=int(t_new))
            st.success("Config saved.")

        st.subheader("Start / Reset")
        if status == "waiting":
            if st.button("Start Market"):
                types = random.sample(MASTER_GOODS, int(t_new))
                set_group_config(grp, status="started", k=int(k_new), t=int(t_new), types=types)
                allocate_initial_items(grp, int(k_new), types)
                st.success("Market started with " + str(k_new) + " units per student and " + str(t_new) + " types.")
                st.rerun()
        else:
            st.caption("Market is running. New students cannot join until reset.")
            if st.button("♻️ Reset Group"):
                reset_group(grp)
                st.success("Group reset to waiting.")
                st.rerun()

        st.subheader("Active Types & Totals")
        status, k_cfg, t_cfg, active_types = get_group_state(grp)
        st.write("Active types:", ", ".join(active_types) if active_types else "(none)")

        if status == "started":
            users = get_group_users(grp)
            rows = []
            for uid, nm in users:
                counts = inventory_counts(grp, uid)
                types_set = {t for t in active_types if counts.get(t, 0) > 0}
                rows.append(
                    {"Name": nm, **{t: counts.get(t, 0) for t in active_types},
                     "Distinct types": len(types_set),
                     "Utility": bundle_utility(uid, types_set)}
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.metric("Total utility (set utility)", current_total_utility(grp))

# ─────────────────────────────────────────────
# Global timed rerun
# ─────────────────────────────────────────────
if st.session_state.get("auto_refresh", False):
    time.sleep(int(st.session_state.get("refresh_interval", 6)))
    st.rerun()
