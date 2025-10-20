# app.py
# Streamlit Pareto Efficiency Dashboard + Trade Proposal UI
# Requires: streamlit, pandas, numpy, pulp
#   pip install streamlit pandas numpy pulp

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import pulp

st.set_page_config(page_title="Allocation Planner", layout="wide")

# ------------------------------
# Helpers
# ------------------------------

def ensure_state():
    """Initialize example data if none exists."""
    if "people" not in st.session_state:
        st.session_state.people = ["Alice", "Bob", "Carol"]
    if "goods" not in st.session_state:
        st.session_state.goods = ["Good A", "Good B", "Good C"]

    n = len(st.session_state.people)
    g = len(st.session_state.goods)

    if "utilities_df" not in st.session_state:
        # Per-unit utility values (editable in UI)
        # rows = people, cols = goods
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
        # Total available units for each good (editable in UI)
        st.session_state.stock_df = pd.DataFrame(
            {"Total Available": [6, 6, 6][:g]},
            index=st.session_state.goods
        )

    if "allocation_df" not in st.session_state:
        # Current allocation: rows = people, cols = goods
        # (editable)
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
    Test if a Pareto-improving allocation exists (continuous).
    If exists, return improved allocation.
    We maximize the sum of utilities subject to:
      - Good totals <= stock
      - Each person's utility >= current utility (+ tiny epsilon on objective via "improve at least one")
    Then we *also* check if everybody is weakly >= current and someone strictly > current by >= 1e-8.

    Returns:
      (is_efficient, improved_alloc, deltas, debug_info)
      - is_efficient: True if NO improvement exists (current is Pareto efficient in the continuous sense).
      - improved_alloc: if improvement exists, the found allocation; else a copy of current_alloc
      - deltas: per-person Δutility and per-good transfers if improvement exists (None otherwise)
      - debug_info: LP status and diagnostic details
    """
    people = list(utilities_df.index)
    goods = list(utilities_df.columns)

    # Current utilities
    u_now = compute_utilities(utilities_df, current_alloc)

    # LP
    prob = pulp.LpProblem("ParetoImprove", pulp.LpMaximize)

    # Decision vars x[i,g] >= 0
    x_vars = {
        (i, g): pulp.LpVariable(f"x_{i}_{g}", lowBound=0)
        for i in people for g in goods
    }

    # Objective: maximize sum_i,g u[i,g]*x[i,g]
    prob += pulp.lpSum(utilities_df.loc[i, g] * x_vars[(i, g)] for i in people for g in goods)

    # Goods availability
    for g in goods:
        prob += pulp.lpSum(x_vars[(i, g)] for i in people) <= float(stock[g])  # respect total stock

    # IR constraints (weak Pareto improvement)
    for i in people:
        lhs = pulp.lpSum(utilities_df.loc[i, g] * x_vars[(i, g)] for g in goods)
        prob += lhs >= float(u_now[i])  # everyone at least as well off

    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    debug = {
        "status": pulp.LpStatus[status],
        "objective_value": pulp.value(prob.objective)
    }

    # Build proposed allocation
    X = pd.DataFrame(0.0, index=people, columns=goods)
    for i in people:
        for g in goods:
            X.loc[i, g] = x_vars[(i, g)].value() if x_vars[(i, g)].value() is not None else 0.0

    # Check improvement condition: all >= current, and someone strictly greater
    u_candidate = compute_utilities(utilities_df, X)
    weakly_better = (u_candidate + 1e-10 >= u_now).all()
    strictly = (u_candidate > u_now + 1e-8).any()

    if pulp.LpStatus[status] != "Optimal" or not weakly_better or not strictly:
        # No Pareto improvement found: treat as efficient (in this continuous model)
        return True, current_alloc.copy(), None, debug

    # Otherwise: improvement exists
    # Produce delta summaries
    deltas = {}

    deltas["utility"] = pd.DataFrame({
        "Current U": u_now,
        "Proposed U": u_candidate,
        "ΔU": u_candidate - u_now
    })

    # Transfers per good = proposed minus current => who gives/receives
    transfer = X - current_alloc
    deltas["transfers"] = transfer

    return False, X, deltas, debug


def summarize_swaps(transfer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert transfer matrix (proposed - current) into a tidy list of who gives what to whom.
    Positive means receiving; negative means giving.
    We construct a flow decomposition for each good.
    """
    people = list(transfer_df.index)
    goods = list(transfer_df.columns)

    rows = []
    for g in goods:
        # build supply (negative) and demand (positive) lists
        gives = [(p, -transfer_df.loc[p, g]) for p in people if transfer_df.loc[p, g] < -1e-9]
        gets = [(p, transfer_df.loc[p, g]) for p in people if transfer_df.loc[p, g] > 1e-9]

        # Greedy match supply->demand for a readable swap list
        gi, ge = 0, 0
        gives = list(gives)
        gets = list(gets)
        while gi < len(gives) and ge < len(gets):
            giver, qty_give = gives[gi]
            receiver, qty_get = gets[ge]
            amt = min(qty_give, qty_get)
            rows.append({
                "Good": g,
                "From": giver,
                "To": receiver,
                "Quantity": float(amt)
            })
            qty_give -= amt
            qty_get -= amt
            if qty_give <= 1e-12:
                gi += 1
            else:
                gives[gi] = (giver, qty_give)
            if qty_get <= 1e-12:
                ge += 1
            else:
                gets[ge] = (receiver, qty_get)

    return pd.DataFrame(rows)


# ------------------------------
# UI
# ------------------------------

ensure_state()

st.title("Planner: Efficiency Dashboard & Trade Proposals")

with st.expander("Edit People & Goods"):
    c1, c2 = st.columns([1, 1])
    with c1:
        new_people = st.text_area("People (one per line)", "\n".join(st.session_state.people))
        if st.button("Update People"):
            st.session_state.people = [p.strip() for p in new_people.splitlines() if p.strip()]
    with c2:
        new_goods = st.text_area("Goods (one per line)", "\n".join(st.session_state.goods))
        if st.button("Update Goods"):
            st.session_state.goods = [g.strip() for g in new_goods.splitlines() if g.strip()]

# Reindex frames if people/goods changed
people = st.session_state.people
goods = st.session_state.goods

# Reindex utilities
st.session_state.utilities_df = st.session_state.utilities_df.reindex(index=people, columns=goods, fill_value=0.0)
# Reindex stock
st.session_state.stock_df = st.session_state.stock_df.reindex(index=goods).fillna(0.0)
# Reindex allocation
st.session_state.allocation_df = st.session_state.allocation_df.reindex(index=people, columns=goods, fill_value=0.0)

st.header("Inputs")

c_util, c_stock = st.columns([2, 1])
with c_util:
    st.subheader("Per-Unit Utilities (people × goods)")
    st.dataframe(st.session_state.utilities_df, use_container_width=True)
with c_stock:
    st.subheader("Total Stock by Good")
    st.dataframe(st.session_state.stock_df, use_container_width=True)

st.subheader("Current Allocation (people × goods)")
st.caption("You can edit this table directly. Totals per good should not exceed stock; if they do, we'll scale down proportionally (for feasibility checks).")
edited_alloc = st.dataframe(
    st.session_state.allocation_df,
    use_container_width=True,
    height=200
)
# Provide a quick “apply edits” button by reading back from session (Streamlit doesn't auto-return edited df directly here).
if st.button("Apply any manual edits to allocation"):
    st.experimental_rerun()

# Clamp to stock (soft enforcement)
A_current = clamp_allocation_to_stock(st.session_state.allocation_df, st.session_state.stock_df["Total Available"])
if not A_current.equals(st.session_state.allocation_df):
    st.info("Current allocation exceeded stock on at least one good; scaled proportionally to fit totals.")
    st.session_state.allocation_df = A_current.copy()

# ------------------------------
# Efficiency Dashboard (Planner)
# ------------------------------
st.header("Efficiency Dashboard (Planner)")

utilities_df = st.session_state.utilities_df.copy().astype(float)
stock = st.session_state.stock_df["Total Available"].copy().astype(float)
current_alloc = st.session_state.allocation_df.copy().astype(float)

u_now = compute_utilities(utilities_df, current_alloc)
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.metric("Social Welfare (sum of utilities)", f"{u_now.sum():.3f}")
with c2:
    st.metric("Min Individual Utility", f"{u_now.min():.3f}")
with c3:
    st.metric("Max Individual Utility", f"{u_now.max():.3f}")

is_eff, proposed_alloc, deltas, debug = solve_pareto_improvement(
    utilities_df, stock, current_alloc, epsilon=1e-6
)

if is_eff:
    st.success("✅ Current allocation is Pareto efficient (under continuous quantities).")
    with st.expander("Debug (LP)"):
        st.json(debug)
else:
    st.error("❗ Current allocation is **not** Pareto efficient (continuous).")
    st.subheader("One Pareto-Improving Reallocation (continuous)")

    # Show utilities change
    st.markdown("**Utilities**")
    st.dataframe(deltas["utility"], use_container_width=True)

    # Show proposed allocation
    st.markdown("**Proposed Allocation**")
    st.dataframe(proposed_alloc, use_container_width=True)

    # Transfers view
    st.markdown("**Transfers (Proposed − Current)**")
    transfer_df = deltas["transfers"]
    st.dataframe(transfer_df, use_container_width=True)

    # Swap summary (readable multilateral trades)
    st.markdown("**Implied Swaps (multilateral allowed)**")
    swap_df = summarize_swaps(transfer_df)
    if swap_df.empty:
        st.write("No net transfers needed (difference is negligible).")
    else:
        st.dataframe(swap_df, use_container_width=True)

    with st.expander("Debug (LP)"):
        st.json(debug)

# ------------------------------
# Participant Trade Proposal UI
# ------------------------------
st.header("Participant Trade Proposals")

st.caption("Build a natural-language proposal with dropdowns. This does **not** change the allocation; it just formats a proposal you can collect from participants.")

people = st.session_state.people
goods = st.session_state.goods

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

# Availability hints
st.caption("Availability check (based on current allocation):")
avail_cols = st.columns(2)
with avail_cols[0]:
    st.write(f"{giver} currently holds:")
    st.dataframe(current_alloc.loc[[giver]].T.rename(columns={giver: "Qty"}))
with avail_cols[1]:
    st.write(f"{recipient} currently holds:")
    st.dataframe(current_alloc.loc[[recipient]].T.rename(columns={recipient: "Qty"}))

# Build sentence
if give_qty > 0 and get_qty > 0:
    proposal = f"I'd give {recipient} {give_qty:g} {give_good} in exchange for {get_qty:g} {get_good}."
else:
    proposal = "Please choose positive quantities to form a proposal."

st.subheader("Proposal")
st.write(proposal)

# Optional validation messages
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

st.divider()
st.caption("Notes: Efficiency test uses a continuous LP relaxation. If you need indivisible/integer items or minimal-distance improvements, we can switch to an MILP or shortest-move formulation.")
