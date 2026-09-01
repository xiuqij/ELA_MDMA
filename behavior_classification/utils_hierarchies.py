import os
import pandas as pd
import numpy as np

def build_wl_matrix(df_group, mouse_ids):
    """
    df_group: subset of rows for one video × box × date.
    mouse_ids: list of all mice you expect in the box (e.g. ["mouseRED","mouseBLUE","mouseGREEN","mouseYELLOW"]).
    Returns:
        wl_mat: n×n matrix where wl_mat[i,j] = wins of i over j
        idx_to_mouse: index→mouse dict (row/column order)
    """
    # initialize
    wl_mat = pd.DataFrame(
        0,
        index=mouse_ids,
        columns=mouse_ids,
        dtype=float
    )

    for _, row in df_group.iterrows():
        winner = row["chaser"]
        loser  = row["chased"]
        if winner in wl_mat.index and loser in wl_mat.columns and winner != loser:
            wl_mat.loc[winner, loser] += 1
    
    return wl_mat.values, {i: m for i, m in enumerate(mouse_ids)}


def davids_score_from_matrix(wl_mat):
    """
    David's score: guaranteed normDS ∈ [0, N-1]
    """
    n = wl_mat.shape[0]
    
    # Total wins for simple ranking fallback
    total_wins = wl_mat.sum()
    
    if total_wins < 4:  # Too sparse: simple win count ranking
        DS = wl_mat.sum(axis=1)
    else:
        # Full David's score
        total = wl_mat + wl_mat.T
        nonzero = total > 0
        
        P = np.zeros_like(wl_mat, dtype=float)
        P[nonzero] = wl_mat[nonzero] / total[nonzero]
        
        # Symmetric handling: undecided pairs get 0.5 exactly
        P[~nonzero] = 0.5
        
        w1 = P.sum(axis=1)
        l1 = (1 - P).sum(axis=1)
        w2 = P @ w1
        l2 = (1 - P) @ l1
        
        DS = w1 + w2 - l1 - l2
    
    # Clip to exact theoretical bounds
    normDS = np.clip((DS + n * (n - 1) / 2.0) / n, 0, n - 1)
    
    return DS, normDS
    