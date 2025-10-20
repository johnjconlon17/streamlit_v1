# app.py
# Streamlit Allocation Planner with URL-based Admin/Participant views
# + Participant metadata (name, group, section) inputs & URL-prefill
#
# requirements.txt (Streamlit Cloud):
# streamlit>=1.37
# pandas>=2.0
# numpy>=1.25
# scipy>=1.10
# pulp

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import pulp

st.set_page_config(page_title="Allocation Planner", layout="wide")


# ------------------------------------------------------------
# URL helpers (role + optional participant metadata prefill)
# ------------------------------------------------------------
def get_query_param(key: str, default: str = "") -> str:
    try:
        val = st.query_params.get(key, default)
    except Exception:
        val = st.experimental_get_query_params().get(key, [default])[0]
    return str(val)


def get_role_from_query() -> str:
    admin_val = get_query_param("admin", "0").lower()
    is_admin = admin_val in ("1", "true", "yes")
    return "planner" if is_admin else "participant"


role = get_role_from_query()
st.markdown("**View:** {}"
            .format("👑 Planner (admin=1)" if role == "planner" else "🧑‍🎓 Participant (admin=0)"))


# ------------------------------------------------------------
# App state helpers
# ------------------------------------------------------------
def ensure_state():
    """Initialize example data if none exists."""
    if "people" not in st.session_state:
        st.session_state.people = ["Alice", "Bob", "Carol"]
    if "goods" not in st.session_state:
        st.session_state.goods = ["Good A", "Good B", "Good C"]

    n = len(st.session_state.people)
    g = len(st.session_state.goods)

    if "utilities_df" not in st.session_state:
        vals = np.array([
            [8, 6, 2],
            [5, 7, 3],
            [4, 3, 9],
        ], dtype=float)[:n, :g]
        st.session_state.utilities_df = pd.DataFrame(
            vals,
            index=st.session_state.people,
            columns=st.session_state.goods
        )

    if "stock_df" not in st.session_state:
        st.session_state.stock_df = pd.DataFrame(
            {"Total Available": [6, 6, 6][:g]},
            index=st.session_state.goods
        )

    if "allocation_df" not in st.session_state:
        alloc = np.array([
            [2, 1, 0],
            [2, 2, 1],
            [2, 3, 5],
        ], dtype=float)[:n, :g]
        st.session_state.allocation_df = pd.DataFrame(
            alloc,
            index=st.session_state.people,
            columns=st.session_state.goods
        )


def reindex_all_frames():
    """Keep frames aligned after edits to people/goods."""
    people = st.session_state.people
    goods = st.session_state.goods

    st.session_state.utilities_df = st.session_state.utilities_df.reindex(
        index=people, columns=goods, fill_value=0.0
    )
    st.session_state.stock_df = st.session_state.stock_df.reindex(index=goods).fillna(0.0)
    st.session_state.allocation_df = st.session_state.allocation_df.reindex(
        index=people, columns=goods, fill_value=0.0
    )


# ------------------------------------------------------------
# Core math helpers
# ------------------------------------------------------------
def clamp_allocation_to_stock(allocation: pd.DataFrame, stock: pd.Series) -> pd.DataFrame:
    """Rescale down proportionally if current allocation exceeds stock by any good."""
    A = allocation.copy().astype(float)
    for good in A.columns:
        total = A[good].sum()
        cap = float(stock.get(good, np.inf))
        if total > cap and cap >= 0:
            if total > 0:
                A[good] = A[good] * (cap / total)
            else:
                A[good] = 0.0
    A[A < 0] = 0.0
    return A


def compute_utilities(utilities_df: pd.DataFrame, allocation_df: pd.DataFrame) -> pd.Series:
    """Utility_i = sum_g u[i,g] * x[i,g]."""
    U = (utilities_df * allocation_df).sum(axis=1)
    return U


def solve_pareto_improvement(
    utilities_df: pd.DataFrame,
    stock: pd.Series,
    current_alloc: pd.DataFrame,
    epsilon: float = 1e-6,
) -> Tuple[bool, pd.DataFrame, Optional[pd.DataFrame], Dict]:
    """
    Check if a Pareto-improving allocation exists (continuous).
    Maximize sum_i,g u[i,g]*x[i,g]
    s.t. sum_i x[i,g] <= stock[g]
         sum_g u[i,g]*x[i,g] >= current utility_i   (IR constraints)
    """
    people = list(utilities_df.index)
    goods = list(utilities_df.columns)

    u_now = compute_utilities(utilities_df, current_alloc)

    prob = pulp.LpProblem("ParetoImprove", pulp.LpMaximize)

    x_vars = {(i, g): pulp.LpVariable(f"x_{i}_{g}", lowBound=0) for i in people for g in goods}

    prob += pulp.lpSum(utilities_df.loc[i, g] * x_vars[(i, g)] for i in people for g in goods)

    for g in goods:
        prob += pulp.lpSum(x_vars[(i, g)] for i in people) <= float(stock[g])

    for i in people:
        lhs = pulp.lpSum(utilities_df.loc[i, g] * x_vars[(i, g)] for g in goods)
        prob += lhs >= float(u_now[i])

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    debug = {"status": pulp.LpStatus[status], "objective_value": pulp.value(prob.objective)}

    X = pd.DataFrame(0.0, index=people, columns=goods)
    for i in people:
        for g in goods:
            val = x_vars[(i, g)].value()
            X.loc[i, g] = val if val is not None else 0.0

    u_candidate = compute_utilities(utilities_df, X)
    weakly_better = (u_candidate + 1e-10 >= u_now).all()
    strictly = (u_candidate > u_now + 1e-8).any()

    if pulp.LpStatus[status] != "Optimal" or not weakly_better or not strictly:
        return True, current_alloc.copy(), None, debug  # Treat as efficient (continuous model)

    deltas = {}
    deltas["utility"] = pd.DataFrame({
        "Current U": u_now,
        "Proposed U": u_candidate,
        "ΔU": u_candidate - u_now
    })

    transfer = X - current_alloc
    deltas["transfers"] = transfer

    return False, X, deltas, debug


def summarize_swaps(transfer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert transfer matrix (proposed - current) into a tidy list of who gives what to whom.
    Positive => receiving; negative => giving. Greedy flow per good for readability.
    """
    people = list(transfer_df.index)
    goods = list(transfer_df.columns)

    rows = []
    for g in goods:
        gives = [(p, -transfer_df.loc[p, g]) for p in people if transfer_df.loc[p, g] < -1e-9]
        gets  = [(p,  transfer_df.loc[p, g]) for p in people if transfer_df.loc[p, g] >  1e-9]

        gi, ge = 0, 0
        gives = list(gives)
        gets = list(gets)
        while gi < len(gives) and ge < len(gets):
            giver, qty_give = gives[gi]
            receiver, qty_get = gets[ge]
            amt = min(qty_give, qty_get)
            rows.append({"Good": g, "From": giver, "To": receiver, "Quantity": float(amt)})
            qty_give -= amt
            qty_get  -= amt
            if qty_give <= 1e-12:
                gi += 1
            else:
                gives[gi] = (giver, qty_give)
            if qty_get <= 1e-12:
                ge += 1
            else:
                gets[ge] = (receiver, qty_get)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------
ensure_state()
reindex_all_frames()

people = st.session_state.people
goods = st.session_state.goods
utilities_df = st.session_state.utilities_df.copy().astype(float)
stock = st.session_state.stock_df["Total Available"].copy().astype(float)
current_alloc = st.session_state.allocation_df.copy().astype(float)


# ============================================================
# PLANNER (ADMIN) VIEW — only if ?admin=1/true/yes
# ============================================================
if role == "planner":
    st.title("Planner: Efficiency Dashboard & Controls")

    with st.expander("Edit People & Goods"):
        c1, c2 = st.columns([1, 1])
        with c1:
            new_people = st.text_area("People (one per line)", "\n".join(people))
            if st.button("Update People"):
                st.session_state.people = [p.strip() for p in new_people.splitlines() if p.strip()]
                reindex_all_frames()
                st.rerun()
        with c2:
            new_goods = st.text_area("Goods (one per line)", "\n".join(goods))
            if st.button("Update Goods"):
                st.session_state.goods = [g.strip() for g in new_goods.splitlines() if g.strip()]
                reindex_all_frames()
                st.rerun()

    # Refresh after edits
    people = st.session_state.people
    goods = st.session_state.goods
    utilities_df = st.session_state.utilities_df.copy().astype(float)
    stock = st.session_state.stock_df["Total Available"].copy().astype(float)
    current_alloc = st.session_state.allocation_df.copy().astype(float)

    st.header("Inputs")
    c_util, c_stock = st.columns([2, 1])
    with c_util:
        st.subheader("Per-Unit Utilities (people × goods)")
        st.dataframe(utilities_df, use_container_width=True)
    with c_stock:
        st.subheader("Total Stock by Good")
        st.dataframe(st.session_state.stock_df, use_container_width=True)

    st.subheader("Current Allocation (people × goods)")
    st.caption("Edit as needed. If totals exceed stock for any good, we scale down proportionally for feasibility checks.")
    st.dataframe(current_alloc, use_container_width=True, height=220)

    clamped = clamp_allocation_to_stock(current_alloc, stock)
    if not clamped.equals(current_alloc):
        st.info("Current allocation exceeded stock for ≥1 good; using a scaled version for feasibility/efficiency checks below.")
    A_for_check = clamped

    u_now = compute_utilities(utilities_df, A_for_check)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.metric("Social Welfare (Σ utilities)", f"{u_now.sum():.3f}")
    with c2:
        st.metric("Min Individual Utility", f"{u_now.min():.3f}")
    with c3:
        st.metric("Max Individual Utility", f"{u_now.max():.3f}")

    is_eff, proposed_alloc, deltas, debug = solve_pareto_improvement(
        utilities_df, stock, A_for_check, epsilon=1e-6
    )

    st.header("Efficiency Dashboard")
    if is_eff:
        st.success("✅ Current allocation is Pareto efficient (continuous quantities).")
        with st.expander("Debug (LP)"):
            st.json(debug)
    else:
        st.error("❗ Current allocation is **not** Pareto efficient (continuous).")
        st.subheader("One Pareto-Improving Reallocation (continuous)")

        st.markdown("**Utilities**")
        st.dataframe(deltas["utility"], use_container_width=True)

        st.markdown("**Proposed Allocation**")
        st.dataframe(proposed_alloc, use_container_width=True)

        st.markdown("**Transfers (Proposed − Current)**")
        transfer_df = deltas["transfers"]
        st.dataframe(transfer_df, use_container_width=True)

        st.markdown("**Implied Swaps (multilateral allowed)**")
        swap_df = summarize_swaps(transfer_df)
        if swap_df.empty:
            st.write("No net transfers needed (difference is negligible).")
        else:
            st.dataframe(swap_df, use_container_width=True)

        with st.expander("Debug (LP)"):
            st.json(debug)

    st.divider()
    st.caption("Note: LP uses continuous quantities. For indivisible items or shortest-move improvements, we can switch to MILP or alternative objectives.")


# ============================================================
# PARTICIPANT (STUDENT) VIEW — default if admin not set
# ============================================================
else:
    st.title("Participant: Trade Proposals")

    # --- Participant metadata (with URL prefill support) ---
    prefill_name = get_query_param("name", "")
    prefill_group = get_query_param("group", "")
    prefill_section = get_query_param("section", "")

    meta_cols = st.columns(3)
    with meta_cols[0]:
        p_name = st.text_input("Your name", value=prefill_name, placeholder="e.g., Jane Doe", key="p_name")
    with meta_cols[1]:
        p_group = st.text_input("Group / Team", value=prefill_group, placeholder="e.g., Blue", key="p_group")
    with meta_cols[2]:
        p_section = st.text_input("Section (optional)", value=prefill_section, placeholder="e.g., A", key="p_section")

    st.caption("Build a natural-language trade proposal. This does **not** execute trades.")

    # Use current session data (read-only for participants, but visible)
    people = st.session_state.people
    goods = st.session_state.goods
    utilities_df = st.session_state.utilities_df.copy().astype(float)
    stock = st.session_state.stock_df["Total Available"].copy().astype(float)
    current_alloc = st.session_state.allocation_df.copy().astype(float)

    colp = st.columns(2)
    with colp[0]:
        giver = st.selectbox("I am (the proposer)", people, key="tp_giver")
        recipient = st.selectbox("Proposing to", [p for p in people if p != giver] or people, key="tp_recipient")
    with colp[1]:
        give_good = st.selectbox("I’d give (Good)", goods, key="tp_give_good")
        give_qty = st.number_input("Quantity to give", min_value=0.0, step=1.0, value=1.0, key="tp_give_qty")

    colq = st.columns(2)
    with colq[0]:
        get_good = st.selectbox("…in exchange for (Good)", goods, key="tp_get_good")
    with colq[1]:
        get_qty = st.number_input("Quantity to receive", min_value=0.0, step=1.0, value=1.0, key="tp_get_qty")

    st.caption("Availability check (based on current allocation):")
    avail_cols = st.columns(2)
    with avail_cols[0]:
        st.write(f"{giver} currently holds:")
        st.dataframe(current_alloc.loc[[giver]].T.rename(columns={giver: "Qty"}))
    with avail_cols[1]:
        st.write(f"{recipient} currently holds:")
        st.dataframe(current_alloc.loc[[recipient]].T.rename(columns={recipient: "Qty"}))

    if give_qty > 0 and get_qty > 0:
        base_sentence = f"I'd give {recipient} {give_qty:g} {give_good} in exchange for {get_qty:g} {get_good}."
    else:
        base_sentence = "Please choose positive quantities to form a proposal."

    st.subheader("Your Proposal")
    # Include the participant’s metadata in a tidy summary block:
    meta_lines = []
    if p_name.strip():
        meta_lines.append("Name: " + p_name.strip())
    if p_group.strip():
        meta_lines.append("Group: " + p_group.strip())
    if p_section.strip():
        meta_lines.append("Section: " + p_section.strip())
    if meta_lines:
        st.write(", ".join(meta_lines))

    st.write(base_sentence)

    # Simple coherence checks
    problems = []
    if give_qty > current_alloc.loc[giver, give_good] + 1e-12:
        problems.append(f"{giver} only has {current_alloc.loc[giver, give_good]:g} of {give_good} to give.")
    if get_qty > current_alloc.loc[recipient, get_good] + 1e-12:
        problems.append(f"{recipient} only has {current_alloc.loc[recipient, get_good]:g} of {get_good} to trade.")
    if give_good == get_good and give_qty == get_qty and giver != recipient and give_qty > 0:
        problems.append("Trading the same good for the same quantity is a wash.")

    if problems:
        st.warning(" ".join(problems))
    else:
        st.info("Looks coherent given current holdings (this does not execute the trade).")

    # Copy-friendly text block
    st.markdown("**Copy this text to submit:**")
    copy_lines = []
    if meta_lines:
        copy_lines.append("; ".join(meta_lines))
    copy_lines.append(base_sentence)
    st.code("\n".join(copy_lines), language="text")

    st.divider()
    st.caption("Tip: Instructors use the planner view by appending “?admin=1”. You can prefill fields like “?name=Jane%20Doe&group=Blue&section=A”.")
