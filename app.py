# app.py — Multi-unit market with Pareto-swap suggestion + sentence trade UI
import streamlit as st
import sqlite3
import random
import pandas as pd
import numpy as np
from contextlib import closing
import json
import time

st.set_page_config(page_title="Bilateral Trading Game (Multi-Unit)", page_icon="🔁", layout="wide")

# Role via URL param
params = st.query_params
is_admin_mode = "admin" in params and str(params["admin"]).lower() in ["1", "true", "yes"]
role_options = ["Student"] + (["Social Planner"] if is_admin_mode else [])

MASTER_GOODS = [
    "Gizmo", "Whatsit", "Thingamabob", "Doohickey", "Widget",
    "Contraption", "Gadget", "Whatchamacallit", "Doodad", "Thingy",
    "Gubbins", "Apparatus", "Mechanism", "Rigamarole", "Oddment",
    "Thingummy", "Whirligig", "Dinglehopper", "Curio", "Bric-a-brac"
]
DB_PATH = "class_trade_multi.db"

if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "refresh_interval" not in st.session_state:
    st.session_state["refresh_interval"] = 6

# ---------- DB ----------
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=6.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=4000;")
    return conn

def init_db():
    conn = get_conn()
    with closing(conn.cursor()) as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                grp  TEXT NOT NULL,
                UNIQUE(name, grp)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                grp TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'waiting',  -- 'waiting' | 'started'
                k INTEGER DEFAULT 1,
                t INTEGER DEFAULT 5,
                types_json TEXT DEFAULT '[]',
                started_ts DATETIME
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                utility INTEGER NOT NULL,
                PRIMARY KEY (user_id, type),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grp TEXT NOT NULL,
                type TEXT NOT NULL,
                owner_id INTEGER NOT NULL,
                FOREIGN KEY(owner_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grp TEXT NOT NULL,
                proposer_id INTEGER NOT NULL,
                recipient_id INTEGER NOT NULL,
                offer_json TEXT NOT NULL,
                request_json TEXT NOT NULL,
                status TEXT NOT NULL,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(proposer_id) REFERENCES users(id),
                FOREIGN KEY(recipient_id) REFERENCES users(id)
            )
        """)
        conn.commit()

def fetchone(q,p=()):
    c=get_conn()
    with closing(c.cursor()) as cur:
        cur.execute(q,p); return cur.fetchone()

def fetchall(q,p=()):
    c=get_conn()
    with closing(c.cursor()) as cur:
        cur.execute(q,p); return cur.fetchall()

def execute(q,p=()):
    c=get_conn()
    with closing(c.cursor()) as cur:
        cur.execute(q,p); c.commit()

# ---------- Group helpers ----------
def ensure_group_row(grp):
    if not grp: return
    if not fetchone("SELECT grp FROM groups WHERE grp=?", (grp,)):
        execute("INSERT INTO groups (grp,status,k,t,types_json) VALUES (?, 'waiting', 1, 5, '[]')", (grp,))

def get_group_state(grp):
    row = fetchone("SELECT status,k,t,types_json FROM groups WHERE grp=?", (grp,))
    if not row: return ("waiting",1,5,[])
    status,k,t,tj = row
    return (status,int(k),int(t),json.loads(tj or "[]"))

def set_group_config(grp, status=None, k=None, t=None, types=None):
    ensure_group_row(grp)
    s,kk,tt,types_cur = get_group_state(grp)
    if status is None: status=s
    if k is None: k=kk
    if t is None: t=tt
    if types is None: types=types_cur
    execute("UPDATE groups SET status=?,k=?,t=?,types_json=? WHERE grp=?",
            (status,int(k),int(t),json.dumps(types),grp))

def reset_group(grp):
    execute("DELETE FROM trades WHERE grp=?", (grp,))
    execute("DELETE FROM items WHERE grp=?", (grp,))
    set_group_config(grp, status="waiting", k=1, t=5, types=[])

def get_group_users(grp):
    return fetchall("SELECT id,name FROM users WHERE grp=? ORDER BY name COLLATE NOCASE", (grp,))

# ---------- Users / prefs ----------
def get_or_create_user(name, grp):
    ensure_group_row(grp)
    status,_,_,_ = get_group_state(grp)
    row = fetchone("SELECT id FROM users WHERE name=? AND grp=?", (name,grp))
    if row: return row[0]
    if status=="started":
        raise RuntimeError("Market already started for this group; new students cannot join until reset.")
    execute("INSERT INTO users (name,grp) VALUES (?,?)", (name,grp))
    uid = fetchone("SELECT id FROM users WHERE name=? AND grp=?", (name,grp))[0]
    for t in MASTER_GOODS:
        u = random.randint(1,10)
        execute("INSERT OR REPLACE INTO preferences (user_id,type,utility) VALUES (?,?,?)", (uid,t,u))
    return uid

def prefs_df_for_user(uid, active_types):
    rows = fetchall("SELECT type,utility FROM preferences WHERE user_id=?", (uid,))
    df = pd.DataFrame(rows, columns=["Type","Utility"])
    if active_types: df = df[df["Type"].isin(active_types)]
    return df.sort_values("Type")

def pref_of(uid, typ):
    r = fetchone("SELECT utility FROM preferences WHERE user_id=? AND type=?", (uid,typ))
    return int(r[0]) if r else 0

# ---------- Items / inventories ----------
def allocate_initial_items(grp, k, types):
    execute("DELETE FROM items WHERE grp=?", (grp,))
    users = get_group_users(grp)
    if not users or not types: return
    for uid,_ in users:
        for _ in range(int(k)):
            execute("INSERT INTO items (grp,type,owner_id) VALUES (?,?,?)", (grp, random.choice(types), uid))

def inventory_counts(grp, uid):
    rows = fetchall("SELECT type, COUNT(*) FROM items WHERE grp=? AND owner_id=? GROUP BY type", (grp,uid))
    return {t:int(c) for t,c in rows}

def inventory_items_by_type(grp, uid):
    rows = fetchall("SELECT id,type FROM items WHERE grp=? AND owner_id=?", (grp,uid))
    by={}
    for iid,t in rows: by.setdefault(t,[]).append(int(iid))
    return by

def group_active_types(grp):
    return get_group_state(grp)[3]

def user_types_set(grp, uid):
    rows = fetchall("SELECT DISTINCT type FROM items WHERE grp=? AND owner_id=?", (grp,uid))
    return {t for (t,) in rows}

def current_total_utility(grp):
    return sum(pref_of(uid,t) for uid,_ in get_group_users(grp) for t in user_types_set(grp,uid))

# ---------- Trades (bundle as dict) ----------
def normalize_bundle(d):
    out={}
    if isinstance(d,dict):
        for k,v in d.items():
            try:
                n=int(v)
            except Exception:
                n=0
            if k and n>0: out[k]=n
    return out

def has_bundle(grp, uid, counts):
    inv = inventory_counts(grp, uid)
    for t,c in counts.items():
        if inv.get(t,0) < int(c): return False
    return True

def propose_trade(grp, proposer_id, recipient_id, offer_counts, request_counts):
    offer = normalize_bundle(offer_counts)
    request = normalize_bundle(request_counts)
    if not offer or not request: return "Invalid bundle."
    if proposer_id==recipient_id: return "Cannot trade with yourself."
    if not has_bundle(grp, proposer_id, offer): return "You no longer have the items you’re offering."
    if not has_bundle(grp, recipient_id, request): return "Partner no longer has the items you’re requesting."
    execute("""INSERT INTO trades (grp,proposer_id,recipient_id,offer_json,request_json,status)
               VALUES (?,?,?,?,?, 'pending')""",
            (grp, proposer_id, recipient_id, json.dumps(offer), json.dumps(request)))
    return "Trade proposed."

def list_incoming(grp, uid):
    return fetchall("""SELECT id, proposer_id, offer_json, request_json, ts
                       FROM trades WHERE grp=? AND recipient_id=? AND status='pending'
                       ORDER BY ts DESC""", (grp,uid))

def list_outgoing(grp, uid):
    return fetchall("""SELECT id, recipient_id, offer_json, request_json, ts
                       FROM trades WHERE grp=? AND proposer_id=? AND status='pending'
                       ORDER BY ts DESC""", (grp,uid))

def cancel_trade(grp, uid, trade_id):
    row = fetchone("SELECT proposer_id,status FROM trades WHERE id=? AND grp=?", (trade_id,grp))
    if row and row[0]==uid and row[1]=="pending":
        execute("UPDATE trades SET status='cancelled' WHERE id=?", (trade_id,))

def decline_trade(trade_id):
    execute("UPDATE trades SET status='declined' WHERE id=?", (trade_id,))

def accept_trade(grp, trade_id):
    row = fetchone("""SELECT proposer_id,recipient_id,offer_json,request_json,status
                      FROM trades WHERE id=? AND grp=?""", (trade_id,grp))
    if not row: return "Trade not found."
    proposer_id, recipient_id, offer_js, request_js, status = row
    if status!="pending": return "Trade no longer pending."
    offer = json.loads(offer_js); request=json.loads(request_js)
    if not has_bundle(grp, proposer_id, offer):
        decline_trade(trade_id); return "Offer invalid: proposer no longer has those items."
    if not has_bundle(grp, recipient_id, request):
        decline_trade(trade_id); return "Offer invalid: recipient no longer has requested items."
    prop_items = inventory_items_by_type(grp, proposer_id)
    recp_items = inventory_items_by_type(grp, recipient_id)
    for t,c in offer.items():
        for iid in prop_items.get(t,[])[:int(c)]:
            execute("UPDATE items SET owner_id=? WHERE id=?", (recipient_id, iid))
    for t,c in request.items():
        for iid in recp_items.get(t,[])[:int(c)]:
            execute("UPDATE items SET owner_id=? WHERE id=?", (proposer_id, iid))
    execute("UPDATE trades SET status='accepted' WHERE id=?", (trade_id,))
    return "Trade executed."

# ---------- Pareto-swap suggestion (one cycle) ----------
def _best_desired_type(grp, uid, active_types):
    have = user_types_set(grp, uid)
    candidates = [t for t in active_types if t not in have]
    if not candidates: return None
    # choose highest utility among types that exist in group (owned by someone else)
    top_t = None; top_u = -1
    for t in candidates:
        any_copy = fetchone("SELECT 1 FROM items WHERE grp=? AND type=? AND owner_id<>? LIMIT 1", (grp,t,uid))
        if not any_copy: continue
        u = pref_of(uid, t)
        if u > top_u:
            top_u = u; top_t = t
    return top_t

def _pick_item_copy_of_type(grp, typ, exclude_uid):
    return fetchone("SELECT id, owner_id FROM items WHERE grp=? AND type=? AND owner_id<>? LIMIT 1", (grp,typ,exclude_uid))

def find_pareto_improving_cycle(grp, attempts=200):
    """
    Try to find ONE Pareto-improving cycle (IR w.r.t current set-utility).
    Returns transfers list: [(item_id, type, from_uid, to_uid), ...] or [] if none.
    """
    users = get_group_users(grp)
    if not users: return []
    uids = [uid for uid,_ in users]
    active = group_active_types(grp)
    if not active: return []

    # Precompute inv counts and types set for IR checks
    inv_counts = {uid: inventory_counts(grp, uid) for uid in uids}
    types_set  = {uid: {t for t,c in inv_counts[uid].items() if c>0} for uid in uids}

    # Some quick exit: if no one desires anything new, it's efficient
    desires_exist = False
    for uid in uids:
        if _best_desired_type(grp, uid, active):
            desires_exist = True; break
    if not desires_exist: return []

    # Attempt multiple times (randomization via DB order suffices)
    for _ in range(attempts):
        # Build agent->item pointer map
        agent_to_item = {}
        item_owner = {}
        item_type  = {}
        desirers = []
        for uid in uids:
            t = _best_desired_type(grp, uid, active)
            if not t: continue
            row = _pick_item_copy_of_type(grp, t, uid)
            if not row: continue
            iid, owner = int(row[0]), int(row[1])
            agent_to_item[uid] = iid
            item_owner[iid] = owner
            item_type[iid] = t
            desirers.append(uid)

        if not agent_to_item: return []

        # Try to extract a cycle from each desirer
        for start in desirers:
            # Path alternates: agent, item, agent, item, ...
            agents_path = [start]
            items_path  = []
            idx_of_agent = {start: 0}
            ok_cycle = None

            while True:
                cur_agent = agents_path[-1]
                iid = agent_to_item.get(cur_agent)
                if iid is None: break
                items_path.append(iid)
                owner = item_owner[iid]
                if owner in idx_of_agent:
                    s = idx_of_agent[owner]
                    cyc_agents = agents_path[s:]            # agents in cycle order
                    cyc_items  = items_path[s:]             # each agent gets corresponding item
                    # IR check
                    strict_any = False
                    ir_ok = True
                    m = len(cyc_agents)
                    for k in range(m):
                        a = cyc_agents[k]
                        gain_item = cyc_items[k]
                        gain_type = item_type[gain_item]
                        # lost item is previous in cycle
                        lost_item = cyc_items[(k-1)%m]
                        lost_type = item_type[lost_item]
                        # Did they currently have gain_type?
                        already_have = (gain_type in types_set[a])
                        # Is lost_type unique?
                        lost_unique = (inv_counts[a].get(lost_type,0) == 1)
                        # IR condition
                        if lost_unique:
                            if pref_of(a, gain_type) < pref_of(a, lost_type):
                                ir_ok = False; break
                        # Strict improvement?
                        if not already_have:
                            if (not lost_unique) or (pref_of(a, gain_type) > pref_of(a, lost_type)):
                                strict_any = True
                    if ir_ok and strict_any:
                        # Build transfer list: each item moves from its owner to the agent who pointed to it
                        transfers = []
                        for k in range(m):
                            to_uid = cyc_agents[k]
                            iid_k  = cyc_items[k]
                            frm_uid= item_owner[iid_k]
                            transfers.append( (iid_k, item_type[iid_k], frm_uid, to_uid) )
                        return transfers
                    # else: break out of this path; try another start / attempt
                    break
                else:
                    agents_path.append(owner)
                    idx_of_agent[owner] = len(agents_path)-1

    return []  # none found → Pareto efficient

def execute_transfers(grp, transfers):
    # transfers: [(item_id, type, from_uid, to_uid), ...]
    # validate current owners first
    for iid, _t, frm, _to in transfers:
        r = fetchone("SELECT owner_id FROM items WHERE id=? AND grp=?", (iid,grp))
        if not r or int(r[0]) != int(frm):
            return "Swap invalidated by new trades. Try again."
    for iid, _t, _frm, to in transfers:
        execute("UPDATE items SET owner_id=? WHERE id=?", (to, iid))
    return "Swap executed."

# ---------- UI ----------
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

# ---------- Student ----------
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
                st.error(str(e)); st.stop()
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
            rows = [(t, inv.get(t,0)) for t in (active_types or [])]
            st.dataframe(pd.DataFrame(sorted(rows), columns=["Type","Count"]),
                         width="stretch", hide_index=True)

            st.subheader("Your Preferences (active types)")
            st.dataframe(prefs_df_for_user(uid, active_types), width="stretch", hide_index=True)

        with colB:
            st.subheader("Group Status")
            st.write("Status:", status.upper())
            roster = pd.DataFrame(get_group_users(grp_clean), columns=["user_id","Name"])
            st.dataframe(roster[["Name"]], width="stretch", hide_index=True)

            if status == "waiting":
                st.info("Waiting period — trades disabled until the planner starts the market.")
            else:
                st.subheader("Propose a Trade")
                others = [(u[1],u[0]) for u in get_group_users(grp_clean) if u[0]!=uid]
                if not others:
                    st.warning("No one else in your group yet.")
                else:
                    partner_names = [o[0] for o in others]
                    partner = st.selectbox("Partner", partner_names, index=None, placeholder="Select a partner")
                    partner_id = None if partner is None else dict(others)[partner]

                    # Sentence UI: "I'd give [Name] [#] [items] for [#] of their [items]"
                    st.markdown("**I’d give**")
                    give_num = st.number_input("Quantity you give", 1, 2, 1, key="give_num")
                    give_type = st.selectbox("of your", (active_types or []), index=0 if active_types else None, key="give_type")

                    st.markdown("**for**")
                    get_num = st.number_input("Quantity you want", 1, 2, 1, key="get_num")
                    get_type = st.selectbox("of their", (active_types or []), index=0 if active_types else None, key="get_type")

                    if partner_id is not None and give_type and get_type:
                        my_inv = inventory_counts(grp_clean, uid)
                        their_inv = inventory_counts(grp_clean, partner_id)
                        if my_inv.get(give_type,0) < give_num:
                            st.warning("You don't have enough " + give_type + ".")
                        if their_inv.get(get_type,0) < get_num:
                            st.warning(partner + " doesn't have enough " + get_type + ".")

                    disabled = partner_id is None or (not give_type) or (not get_type)
                    if st.button("Send offer", disabled=disabled):
                        offer   = {give_type: int(give_num)}
                        request = {get_type:  int(get_num)}
                        msg = propose_trade(grp_clean, uid, partner_id, offer, request)
                        if msg == "Trade proposed.":
                            st.success(msg); st.rerun()
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
                            a,b = st.columns(2)
                            if a.button("Accept", key="acc"+str(tid)):
                                res = accept_trade(grp_clean, tid)
                                if res.startswith("Trade executed"):
                                    st.success(res)
                                else:
                                    st.warning(res)
                                st.rerun()
                            if b.button("Decline", key="dec"+str(tid)):
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
                            if st.button("Cancel", key="can"+str(tid)):
                                cancel_trade(grp_clean, uid, tid); st.rerun()

# ---------- Planner ----------
elif role == "Social Planner":
    if not grp:
        st.info("Enter a group code in the sidebar.")
    else:
        ensure_group_row(grp)
        status, k_cfg, t_cfg, active_types = get_group_state(grp)

        st.title("Social Planner")
        st.write("Group:", grp, "· Status:", status.upper())

        with st.expander("Roster"):
            roster = pd.DataFrame(get_group_users(grp), columns=["user_id","Name"])
            st.dataframe(roster[["Name"]], width="stretch", hide_index=True)

        st.subheader("Configuration")
        k_new = st.number_input("Units per student (K)", 1, 10, k_cfg)
        max_t = min(len(MASTER_GOODS), 20)
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
            rows=[]
            for uid,nm in users:
                counts = inventory_counts(grp, uid)
                types_set = {t for t in active_types if counts.get(t,0)>0}
                rows.append({"Name": nm, **{t: counts.get(t,0) for t in active_types},
                             "Distinct types": len(types_set),
                             "Utility": sum(pref_of(uid,t) for t in types_set)})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.metric("Total utility (set utility)", current_total_utility(grp))

            st.divider()
            st.subheader("Efficiency: suggest a Pareto-improving swap")
            st.caption("Finds one multilateral cycle (TTC-style) that makes someone strictly better off and nobody worse off (relative to the current allocation).")
            if st.button("Find Pareto-improving swap"):
                transfers = find_pareto_improving_cycle(grp)
                if not transfers:
                    st.success("No IR-improving cycle found — current allocation appears Pareto efficient.")
                else:
                    st.session_state["suggested_transfers"] = transfers
                    st.info("Proposed multilateral swap:")
            transfers = st.session_state.get("suggested_transfers")
            if transfers:
                pretty=[]
                for iid, typ, frm, to in transfers:
                    frm_name = fetchone("SELECT name FROM users WHERE id=?", (frm,))[0]
                    to_name  = fetchone("SELECT name FROM users WHERE id=?", (to, ))[0]
                    pretty.append({"From": frm_name, "To": to_name, "Type": typ, "Item ID": iid})
                st.dataframe(pd.DataFrame(pretty), width="stretch", hide_index=True)
                c1,c2 = st.columns(2)
                if c1.button("Execute this swap"):
                    res = execute_transfers(grp, transfers)
                    if res.startswith("Swap executed"):
                        st.success(res)
                        st.session_state["suggested_transfers"] = None
                        st.rerun()
                    else:
                        st.warning(res)
                if c2.button("Clear suggestion"):
                    st.session_state["suggested_transfers"] = None

# ---------- Global timed rerun ----------
if st.session_state.get("auto_refresh", False):
    time.sleep(int(st.session_state.get("refresh_interval", 6)))
    st.rerun()
